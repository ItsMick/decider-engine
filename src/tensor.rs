//! FP32 and experimental INT8 kernels with disjoint parallel output slices.
use crate::int8::Kernel;
use crate::weights::Data;
use rayon::prelude::*;

pub(crate) fn dot_scores(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, b_tail) = b.as_chunks::<8>();
    let mut lanes = [0.0; 8];
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for i in 0..8 {
            if cfg!(target_arch = "aarch64") {
                lanes[i] = a[i].mul_add(b[i], lanes[i]);
            } else {
                lanes[i] += a[i] * b[i];
            }
        }
    }
    let mut sum = lanes.iter().sum::<f32>();
    for (a, b) in a_tail.iter().zip(b_tail) {
        sum += a * b;
    }
    sum
}

/// Projection columns computed together. On AArch64, 3x3 tiles of `dot_tile`
/// measured 1.3-1.9x faster than one `dot_scores` at a time on Neoverse-N1 and
/// Apple CPUs; on x86-64 tiling was slower, so it keeps single dot products.
const TILE: usize = if cfg!(target_arch = "aarch64") { 3 } else { 1 };

/// Per-worker conversion buffers for packed weight and adapter rows.
struct Scratch<const T: usize> {
    weights: [Vec<f32>; T],
    adapter: Vec<f32>,
}
impl<const T: usize> Scratch<T> {
    fn new() -> Self {
        Self {
            weights: std::array::from_fn(|_| Vec::new()),
            adapter: Vec::new(),
        }
    }
}

/// Nine dot products, rows of `xs` against rows of `ws`. Each keeps the exact
/// operation order of `dot_scores` (eight lanes, their sum, then the tail), so
/// results are bit-identical; nine independent accumulator chains let the CPU
/// overlap them, and each loaded chunk is reused three times.
#[inline(never)]
pub(crate) fn dot_tile(xs: [&[f32]; 3], ws: [&[f32]; 3]) -> [[f32; 3]; 3] {
    let len = xs[0].len();
    assert!(xs.iter().chain(&ws).all(|v| v.len() == len));
    let chunks = len / 8;
    let xr: [&[[f32; 8]]; 3] = std::array::from_fn(|i| xs[i].as_chunks::<8>().0);
    let wr: [&[[f32; 8]]; 3] = std::array::from_fn(|j| ws[j].as_chunks::<8>().0);
    let mut lanes = [[0.0f32; 8]; 9];
    for k in 0..chunks {
        let a: [[f32; 8]; 3] = std::array::from_fn(|i| xr[i][k]);
        let b: [[f32; 8]; 3] = std::array::from_fn(|j| wr[j][k]);
        for i in 0..3 {
            for j in 0..3 {
                let acc = &mut lanes[i * 3 + j];
                for q in 0..8 {
                    if cfg!(target_arch = "aarch64") {
                        acc[q] = a[i][q].mul_add(b[j][q], acc[q]);
                    } else {
                        acc[q] += a[i][q] * b[j][q];
                    }
                }
            }
        }
    }
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut sum = lanes[i * 3 + j].iter().sum::<f32>();
            for t in chunks * 8..len {
                sum += xs[i][t] * ws[j][t];
            }
            sum
        })
    })
}

/// Operations applied while transposing completed projection values.
#[derive(Clone, Copy)]
pub(crate) enum Epilogue {
    Write,
    Add,
    SwiGlu,
}
impl Epilogue {
    pub(crate) fn store(self, dst: &mut f32, value: f32) {
        match self {
            Self::Write => *dst = value,
            Self::Add => *dst += value,
            Self::SwiGlu => *dst = crate::qwen::silu(*dst) * value,
        }
    }
}

pub(crate) struct Adapter<'a> {
    pub inputs: &'a [f32],
    pub weights: &'a Data,
    pub rank: usize,
    pub scale: f32,
}
impl Adapter<'_> {
    fn combine(&self, base: f32, row: usize, weights: &[f32]) -> f32 {
        // Preserve the standalone B projection's bias-zero addition and the
        // original FP32 multiply/add order; do not use a fused multiply-add.
        let delta = dot_scores(
            &self.inputs[row * self.rank..(row + 1) * self.rank],
            weights,
        ) + 0.0;
        base + self.scale * delta
    }
}

pub(crate) struct Projection<'a> {
    pub input: usize,
    pub output: usize,
    pub weights: &'a Data,
    pub bias: Option<&'a [f32]>,
    pub adapter: Option<Adapter<'a>>,
}
impl Projection<'_> {
    /// Base and LoRA share the final write, which may add a residual or
    /// consume an existing gate. Larger batches retain separate dot passes.
    pub(crate) fn apply(&self, x: &[f32], result: &mut [f32], epilogue: Epilogue) {
        self.apply_quantized(x, None, result, epilogue);
    }

    /// `apply` with `x` already quantized for INT8 computation, when available.
    pub(crate) fn apply_quantized(
        &self,
        x: &[f32],
        quantized: Option<&QuantizedInputs>,
        result: &mut [f32],
        epilogue: Epilogue,
    ) {
        let (input, output) = (self.input, self.output);
        let rows = x.len() / input;
        assert_eq!(x.len(), rows * input);
        assert_eq!(self.weights.len(), output * input);
        assert_eq!(result.len(), rows * output);
        assert!(self.bias.is_none_or(|b| b.len() == output));
        if let Some(adapter) = &self.adapter {
            assert_eq!(adapter.inputs.len(), rows * adapter.rank);
            assert_eq!(adapter.weights.len(), output * adapter.rank);
        }
        if rows == 0 {
            return;
        }
        if let Some(adapter) = self.adapter.as_ref().filter(|_| rows >= 64) {
            let base = Projection {
                input,
                output,
                weights: self.weights,
                bias: self.bias,
                adapter: None,
            }
            .columns_quantized(x, quantized);
            let delta = Projection {
                input: adapter.rank,
                output,
                weights: adapter.weights,
                bias: None,
                adapter: None,
            }
            .columns(adapter.inputs);
            write_columns(
                &base,
                Some((&delta, adapter.scale)),
                rows,
                output,
                result,
                epilogue,
            );
        } else {
            let columns = self.columns_quantized(x, quantized);
            write_columns(&columns, None, rows, output, result, epilogue);
        }
    }

    fn columns(&self, x: &[f32]) -> Vec<f32> {
        self.columns_quantized(x, None)
    }

    fn columns_quantized(&self, x: &[f32], quantized: Option<&QuantizedInputs>) -> Vec<f32> {
        if let Data::Quantized {
            bits: 8,
            compute_int8: true,
            ..
        } = self.weights
        {
            if let Some(columns) = linear_int8(x, self, quantized) {
                return columns;
            }
        }
        #[cfg(feature = "fast-gemm")]
        if x.len() / self.input >= 64 && self.input >= 1024 && self.output >= 128 {
            return self.columns_gemm(x);
        }
        if cfg!(feature = "cache-blocking")
            && x.len() / self.input >= 64
            && self.input >= 256
            && self.output >= 24
        {
            self.columns_blocked::<24, TILE, 24>(x)
        } else {
            self.columns_tiled::<TILE>(x)
        }
    }

    /// Single-dot-product columns when `T` is 1; 3x3 `dot_tile` blocks when `T`
    /// is 3. Both produce bit-identical results.
    fn columns_tiled<const T: usize>(&self, x: &[f32]) -> Vec<f32> {
        self.columns_blocked::<T, T, { usize::MAX }>(x)
    }

    /// A safe view onto a packed weight panel; the optional Rust GEMM backend
    /// owns its packing and SIMD dispatch. Its reduction order can differ.
    #[cfg(feature = "fast-gemm")]
    fn columns_gemm(&self, x: &[f32]) -> Vec<f32> {
        use ndarray::{linalg::general_mat_mul, ArrayView2, ArrayViewMut2, ShapeBuilder};
        let (input, output) = (self.input, self.output);
        let rows = x.len() / input;
        let a = ArrayView2::from_shape((rows, input), x).expect("validated input shape");
        let mut columns = vec![0.0; rows * output];
        columns
            .par_chunks_mut(rows * 128)
            .enumerate()
            .for_each_init(
                || (Vec::new(), Vec::new()),
                |(weights_scratch, adapter_scratch), (block, dst)| {
                    let first = block * 128;
                    let count = dst.len() / rows;
                    let packed = self
                        .weights
                        .row(first * input, count * input, weights_scratch);
                    let b = ArrayView2::from_shape((count, input), packed)
                        .expect("validated weight panel");
                    let mut c = ArrayViewMut2::from_shape((rows, count).f(), &mut *dst)
                        .expect("disjoint column panel");
                    general_mat_mul(1.0, &a, &b.t(), 0.0, &mut c);
                    for (j, column) in dst.chunks_exact_mut(rows).enumerate() {
                        let bias = self.bias.map_or(0.0, |bias| bias[first + j]);
                        let adapter = self.adapter.as_ref().map(|adapter| {
                            (
                                adapter,
                                adapter.weights.row(
                                    (first + j) * adapter.rank,
                                    adapter.rank,
                                    adapter_scratch,
                                ),
                            )
                        });
                        for (row, value) in column.iter_mut().enumerate() {
                            let base = *value + bias;
                            *value = adapter.map_or(base, |(a, w)| a.combine(base, row, w));
                        }
                    }
                },
            );
        columns
    }

    /// Keep a row block hot while projecting it against C output columns.
    /// Every individual dot still uses the original eight-lane reduction.
    fn columns_blocked<const C: usize, const T: usize, const R: usize>(
        &self,
        x: &[f32],
    ) -> Vec<f32> {
        assert!(C > 0 && R > 0 && (T == 1 || T == 3) && C.is_multiple_of(T));
        let (input, output) = (self.input, self.output);
        let rows = x.len() / input;
        let tokens: Vec<&[f32]> = x.chunks_exact(input).collect();
        // Each task owns C adjacent full columns, so writes remain disjoint.
        let compute = |group: usize, dst: &mut [f32], scratch: &mut Scratch<C>| {
            let first = group * C;
            let count = dst.len() / rows;
            let weights: Vec<&[f32]> = scratch.weights[..count]
                .iter_mut()
                .enumerate()
                .map(|(j, buf)| self.weights.row((first + j) * input, input, buf))
                .collect();
            for row_start in (0..rows).step_by(R) {
                let end = row_start.saturating_add(R).min(rows);
                for col in (0..count).step_by(T) {
                    let width = T.min(count - col);
                    let mut row = row_start;
                    if T == 3 && width == 3 {
                        while row + 3 <= end {
                            let tile = dot_tile(
                                [tokens[row], tokens[row + 1], tokens[row + 2]],
                                [weights[col], weights[col + 1], weights[col + 2]],
                            );
                            for (i, values) in tile.iter().enumerate() {
                                for (j, value) in values.iter().enumerate() {
                                    dst[(col + j) * rows + row + i] = *value;
                                }
                            }
                            row += 3;
                        }
                    }
                    for row in row..end {
                        for j in col..col + width {
                            dst[j * rows + row] = dot_scores(tokens[row], weights[j]);
                        }
                    }
                }
            }
            for (j, column) in dst.chunks_mut(rows).enumerate() {
                let adapter = self.adapter.as_ref().map(|a| {
                    (
                        a,
                        a.weights
                            .row((first + j) * a.rank, a.rank, &mut scratch.adapter),
                    )
                });
                let b = self.bias.map_or(0.0, |b| b[first + j]);
                for (row, value) in column.iter_mut().enumerate() {
                    let base = *value + b;
                    *value = adapter.map_or(base, |(a, w)| a.combine(base, row, w));
                }
            }
        };
        let mut columns = vec![0.0; rows * output];
        if rows * output >= 512 {
            columns
                .par_chunks_mut(rows * C)
                .enumerate()
                .for_each_init(Scratch::new, |scratch, (group, dst)| {
                    compute(group, dst, scratch)
                });
        } else {
            let mut scratch = Scratch::new();
            for (group, dst) in columns.chunks_mut(rows * C).enumerate() {
                compute(group, dst, &mut scratch);
            }
        }
        columns
    }
}

fn write_columns(
    columns: &[f32],
    adapter: Option<(&[f32], f32)>,
    rows: usize,
    output: usize,
    result: &mut [f32],
    epilogue: Epilogue,
) {
    for (row, dst) in result.chunks_exact_mut(output).enumerate() {
        for (column, value) in dst.iter_mut().enumerate() {
            let index = column * rows + row;
            let base = columns[index];
            let projected = adapter.map_or(base, |(delta, scale)| base + scale * delta[index]);
            epilogue.store(value, projected);
        }
    }
}

pub(crate) fn linear(
    x: &[f32],
    rows: usize,
    input: usize,
    weights: &Data,
    output: usize,
    bias: Option<&[f32]>,
    result: &mut [f32],
) {
    assert_eq!(x.len(), rows * input);
    Projection {
        input,
        output,
        weights,
        bias,
        adapter: None,
    }
    .apply(x, result, Epilogue::Write);
}

pub(crate) struct QuantizedInputs {
    values: Vec<i8>,
    scales: Vec<f32>,
}
impl QuantizedInputs {
    pub(crate) fn new(x: &[f32], width: usize) -> Option<Self> {
        // Preserve FP32 propagation for non-finite inputs rather than silently
        // turning NaNs or infinities into integer zeroes or saturated values.
        if !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        let groups = width.div_ceil(64);
        let rows = x.len() / width;
        let mut result = Self {
            values: vec![0; x.len()],
            scales: vec![0.0; rows * groups],
        };
        // Rows are quantized independently, so they can run in parallel.
        let quantize = |((row, values), scales): ((&[f32], &mut [i8]), &mut [f32])| {
            for ((group, values), scale) in row.chunks(64).zip(values.chunks_mut(64)).zip(scales) {
                let max = group.iter().map(|v| v.abs()).fold(0.0, f32::max);
                *scale = if max == 0.0 {
                    1.0
                } else {
                    (max / 127.0).max(f32::MIN_POSITIVE)
                };
                for (q, v) in values.iter_mut().zip(group) {
                    *q = (v / *scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        };
        if x.len() >= 1 << 16 && rayon::current_num_threads() > 1 {
            x.par_chunks(width)
                .zip(result.values.par_chunks_mut(width))
                .zip(result.scales.par_chunks_mut(groups))
                .for_each(quantize);
        } else {
            x.chunks(width)
                .zip(result.values.chunks_mut(width))
                .zip(result.scales.chunks_mut(groups))
                .for_each(quantize);
        }
        Some(result)
    }
}

/// An activation read by several projections. Its INT8 quantization for
/// `int8-compute` is made once, on first use, and shared by them all.
pub(crate) struct SharedInput<'a> {
    pub x: &'a [f32],
    width: usize,
    quantized: std::cell::OnceCell<Option<QuantizedInputs>>,
}
impl<'a> SharedInput<'a> {
    pub(crate) fn new(x: &'a [f32], width: usize) -> Self {
        Self {
            x,
            width,
            quantized: std::cell::OnceCell::new(),
        }
    }
    /// `None` when the input has non-finite values and must stay FP32.
    pub(crate) fn quantized(&self) -> Option<&QuantizedInputs> {
        self.quantized
            .get_or_init(|| QuantizedInputs::new(self.x, self.width))
            .as_ref()
    }
}

fn linear_int8(
    x: &[f32],
    projection: &Projection<'_>,
    quantized: Option<&QuantizedInputs>,
) -> Option<Vec<f32>> {
    let (input, output) = (projection.input, projection.output);
    let rows = x.len() / input;
    let Data::Quantized {
        values,
        scales,
        width,
        bits: 8,
        ..
    } = projection.weights
    else {
        unreachable!("integer projection requires INT8 weights");
    };
    assert_eq!(*width, input);
    let owned;
    let inputs = match quantized {
        Some(shared) => {
            assert_eq!(shared.values.len(), x.len());
            shared
        }
        None => {
            owned = QuantizedInputs::new(x, input)?;
            &owned
        }
    };
    let groups = input.div_ceil(64);
    let kernel = Kernel::detect();
    if kernel.tiles(input) {
        return Some(linear_int8_tiled(
            inputs, projection, values, scales, &kernel,
        ));
    }
    let compute = |column: usize, dst: &mut [f32], scratch: &mut Vec<f32>| {
        let w = &values[column * input..(column + 1) * input];
        let w_scales = &scales[column * groups..(column + 1) * groups];
        let adapter = projection
            .adapter
            .as_ref()
            .map(|a| (a, a.weights.row(column * a.rank, a.rank, scratch)));
        for (row, (value, token)) in dst
            .iter_mut()
            .zip(inputs.values.chunks_exact(input))
            .enumerate()
        {
            let mut sum = 0.0;
            let x_scales = &inputs.scales[row * groups..(row + 1) * groups];
            for (((a, b), xs), ws) in token
                .chunks(64)
                .zip(w.chunks(64))
                .zip(x_scales)
                .zip(w_scales)
            {
                sum += kernel.dot(a, b) as f32 * (*xs * *ws);
            }
            let base = sum + projection.bias.map_or(0.0, |b| b[column]);
            *value = adapter.map_or(base, |(a, w)| a.combine(base, row, w));
        }
    };
    let mut columns = vec![0.0; rows * output];
    if rows * output >= 512 {
        columns
            .par_chunks_mut(rows)
            .enumerate()
            .for_each_init(Vec::new, |scratch, (column, dst)| {
                compute(column, dst, scratch)
            });
    } else {
        let mut scratch = Vec::new();
        for (column, dst) in columns.chunks_mut(rows).enumerate() {
            compute(column, dst, &mut scratch);
        }
    }
    Some(columns)
}

/// `linear_int8` with native dot products: 4x4 blocks of rows and weight rows
/// share loads and group reductions. Every output equals `Kernel::scaled_dot`
/// exactly, followed by the same bias and adapter steps as the per-output path.
fn linear_int8_tiled(
    inputs: &QuantizedInputs,
    projection: &Projection<'_>,
    values: &[u8],
    scales: &[f32],
    kernel: &Kernel,
) -> Vec<f32> {
    use crate::int8::{TILE_COLUMNS as TC, TILE_ROWS as TR};
    let (input, output) = (projection.input, projection.output);
    let groups = input.div_ceil(64);
    let rows = inputs.values.len() / input;
    let x: Vec<&[i8]> = inputs.values.chunks_exact(input).collect();
    let x_scales: Vec<&[f32]> = inputs.scales.chunks_exact(groups).collect();
    let compute = |block: usize, dst: &mut [f32], scratch: &mut Vec<f32>| {
        let first = block * TC;
        let count = dst.len() / rows;
        let w: Vec<&[u8]> = (first..first + count)
            .map(|c| &values[c * input..(c + 1) * input])
            .collect();
        let w_scales: Vec<&[f32]> = (first..first + count)
            .map(|c| &scales[c * groups..(c + 1) * groups])
            .collect();
        let mut row = 0;
        if count == TC {
            while row + TR <= rows {
                let tile = kernel.scaled_tile(
                    std::array::from_fn(|r| x[row + r]),
                    std::array::from_fn(|r| x_scales[row + r]),
                    std::array::from_fn(|c| w[c]),
                    std::array::from_fn(|c| w_scales[c]),
                );
                for (r, sums) in tile.iter().enumerate() {
                    for (c, sum) in sums.iter().enumerate() {
                        dst[c * rows + row + r] = *sum;
                    }
                }
                row += TR;
            }
        }
        for row in row..rows {
            for c in 0..count {
                dst[c * rows + row] = kernel.scaled_dot(x[row], x_scales[row], w[c], w_scales[c]);
            }
        }
        for (c, column) in dst.chunks_mut(rows).enumerate() {
            let adapter = projection
                .adapter
                .as_ref()
                .map(|a| (a, a.weights.row((first + c) * a.rank, a.rank, scratch)));
            let bias = projection.bias.map_or(0.0, |b| b[first + c]);
            for (row, value) in column.iter_mut().enumerate() {
                let base = *value + bias;
                *value = adapter.map_or(base, |(a, w)| a.combine(base, row, w));
            }
        }
    };
    let mut columns = vec![0.0; rows * output];
    if rows * output >= 512 {
        columns
            .par_chunks_mut(rows * TC)
            .enumerate()
            .for_each_init(Vec::new, |scratch, (block, dst)| {
                compute(block, dst, scratch)
            });
    } else {
        let mut scratch = Vec::new();
        for (block, dst) in columns.chunks_mut(rows * TC).enumerate() {
            compute(block, dst, &mut scratch);
        }
    }
    columns
}

pub(crate) fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Precision;

    #[test]
    #[ignore = "synthetic timing; run with --release and a fixed RAYON_NUM_THREADS"]
    fn benchmark_int8_projections() {
        use std::{hint::black_box, time::Instant};
        let (input, output) = (2048usize, 2048usize);
        let mut weights = Data::Quantized {
            values: (0..input * output)
                .map(|i| ((i * 71 % 255) as i16 - 127) as u8)
                .collect(),
            scales: vec![1.0 / 127.0; output * input.div_ceil(64)],
            width: input,
            elements: input * output,
            bits: 8,
            compute_int8: false,
        };
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        println!(
            "ARM dotprod detected: {}",
            std::arch::is_aarch64_feature_detected!("dotprod")
        );
        println!("Rayon workers: {}", rayon::current_num_threads());
        for rows in [1, 32, 128] {
            let x: Vec<f32> = (0..rows * input)
                .map(|i| ((i * 43 % 255) as f32 - 127.0) / 127.0)
                .collect();
            let mut result = vec![0.0; rows * output];
            for compute in [false, true] {
                if let Data::Quantized { compute_int8, .. } = &mut weights {
                    *compute_int8 = compute;
                }
                linear(&x, rows, input, &weights, output, None, &mut result);
                let start = Instant::now();
                for _ in 0..3 {
                    linear(
                        black_box(&x),
                        rows,
                        input,
                        &weights,
                        output,
                        None,
                        &mut result,
                    );
                    black_box(&result);
                }
                println!(
                    "rows={rows} int8_compute={compute} mean_ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0 / 3.0
                );
            }
        }
    }

    #[test]
    fn integer_projection_matches_independent_float_reference() {
        // Exercise sequential/parallel output, full SIMD blocks, tails, multiple
        // groups, zero inputs, signed bytes, distinct group scales and biases.
        for input in [1usize, 15, 16, 17, 63, 64, 65, 131] {
            for rows in [1, 19] {
                let output = 31;
                let groups = input.div_ceil(64);
                let x: Vec<f32> = (0..rows * input)
                    .map(|i| {
                        if i < input {
                            0.0
                        } else {
                            ((i * 43 % 255) as f32 - 127.0) / 8.0
                        }
                    })
                    .collect();
                let values: Vec<u8> = (0..output * input).map(|i| (i * 71 + 128) as u8).collect();
                let scales: Vec<f32> = (0..output * groups)
                    .map(|i| (i % 7 + 1) as f32 / 32.0)
                    .collect();
                let bias: Vec<f32> = (0..output).map(|i| i as f32 / 16.0).collect();
                let weights = Data::Quantized {
                    values,
                    scales,
                    width: input,
                    elements: output * input,
                    bits: 8,
                    compute_int8: true,
                };
                let mut actual = vec![0.0; rows * output];
                linear(&x, rows, input, &weights, output, Some(&bias), &mut actual);
                let quantized = QuantizedInputs::new(&x, input).unwrap();
                let mut scratch = Vec::new();
                for column in 0..output {
                    let w = weights.row(column * input, input, &mut scratch);
                    for row in 0..rows {
                        // Independent FP64 multiplication of reconstructed
                        // inputs and weights, without using integer dot code.
                        let expected = f64::from(bias[column])
                            + w.iter()
                                .enumerate()
                                .map(|(k, w)| {
                                    f64::from(*w)
                                        * f64::from(quantized.values[row * input + k])
                                        * f64::from(quantized.scales[row * groups + k / 64])
                                })
                                .sum::<f64>();
                        assert!(
                            (f64::from(actual[row * output + column]) - expected).abs()
                                < 1e-3 + expected.abs() * 1e-6,
                            "width {input}, row {row}, column {column}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn integer_projection_tiles_match_per_output_scaled_dots() {
        let kernel = Kernel::detect();
        let rank = 3;
        for (rows, input, output) in [
            (1usize, 64usize, 4usize),
            (4, 64, 4),
            (5, 128, 6),
            (9, 192, 31),
            (67, 128, 8),
            (7, 100, 9),
        ] {
            let groups = input.div_ceil(64);
            let x: Vec<f32> = (0..rows * input)
                .map(|i| ((i * 43 % 255) as f32 - 127.0) / 9.0)
                .collect();
            let values: Vec<u8> = (0..output * input).map(|i| (i * 71 + 128) as u8).collect();
            let scales: Vec<f32> = (0..output * groups)
                .map(|i| (i % 7 + 1) as f32 / 32.0)
                .collect();
            let weights = Data::Quantized {
                values: values.clone(),
                scales: scales.clone(),
                width: input,
                elements: output * input,
                bits: 8,
                compute_int8: true,
            };
            let bias: Vec<f32> = (0..output).map(|i| i as f32 / 16.0 - 0.5).collect();
            let mid: Vec<f32> = (0..rows * rank)
                .map(|i| (i % 9) as f32 / 4.0 - 1.0)
                .collect();
            let adapter_weights =
                Data::F32((0..output * rank).map(|i| (i % 7) as f32 / 8.0).collect());
            let projection = Projection {
                input,
                output,
                weights: &weights,
                bias: Some(&bias),
                adapter: Some(Adapter {
                    inputs: &mid,
                    weights: &adapter_weights,
                    rank,
                    scale: 0.375,
                }),
            };
            let actual = projection.columns(&x);
            let quantized = QuantizedInputs::new(&x, input).unwrap();
            let mut scratch = Vec::new();
            for column in 0..output {
                let a = projection.adapter.as_ref().unwrap();
                let aw = a.weights.row(column * rank, rank, &mut scratch).to_vec();
                for row in 0..rows {
                    let sum = kernel.scaled_dot(
                        &quantized.values[row * input..(row + 1) * input],
                        &quantized.scales[row * groups..(row + 1) * groups],
                        &values[column * input..(column + 1) * input],
                        &scales[column * groups..(column + 1) * groups],
                    );
                    let expected = a.combine(sum + bias[column], row, &aw);
                    assert_eq!(
                        actual[column * rows + row].to_bits(),
                        expected.to_bits(),
                        "{rows}x{input}->{output} row {row} column {column}"
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_input_quantization_matches_serial_exactly() {
        let pools: Vec<_> = [1, 4]
            .into_iter()
            .map(|t| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
            })
            .collect();
        // Above and below the parallel threshold, with a partial last group.
        for (rows, width) in [(3, 100), (70, 1000), (40, 2048)] {
            let x: Vec<f32> = (0..rows * width)
                .map(|i| match i % 17 {
                    0 => 0.0,
                    1 => -1e-38,
                    k => ((i * 31 % 499) as f32 - 249.0) / k as f32,
                })
                .collect();
            let results: Vec<_> = pools
                .iter()
                .map(|p| p.install(|| QuantizedInputs::new(&x, width).unwrap()))
                .collect();
            assert_eq!(results[0].values, results[1].values);
            assert!(results[0]
                .scales
                .iter()
                .zip(&results[1].scales)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert_eq!(results[0].scales.len(), rows * width.div_ceil(64));
        }
    }

    #[test]
    fn dynamic_input_quantization_bounds_error_and_preserves_nonfinite_fallback() {
        let width = 131;
        let mut x: Vec<_> = (0..width * 3)
            .map(|i| ((i % 79) as f32 - 39.0) / 11.0)
            .collect();
        x[..64].fill(0.0);
        x[64..128].fill(f32::from_bits(1));
        let q = QuantizedInputs::new(&x, width).unwrap();
        assert_eq!(q.values.len(), x.len());
        assert_eq!(q.scales.len(), 9);
        for (row, token) in x.chunks(width).enumerate() {
            for (group, inputs) in token.chunks(64).enumerate() {
                let scale = q.scales[row * 3 + group];
                assert!(scale.is_finite() && scale > 0.0);
                for (offset, original) in inputs.iter().enumerate() {
                    let restored = f32::from(q.values[row * width + group * 64 + offset]) * scale;
                    assert!((restored - original).abs() <= scale / 2.0 + 1e-6);
                }
            }
        }
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(QuantizedInputs::new(&[nonfinite], 1).is_none());
            let weights = Data::Quantized {
                values: vec![1],
                scales: vec![1.0],
                width: 1,
                elements: 1,
                bits: 8,
                compute_int8: true,
            };
            let mut result = [0.0];
            linear(&[nonfinite], 1, 1, &weights, 1, None, &mut result);
            assert!(!result[0].is_finite());
        }
    }

    #[test]
    fn dot_tile_matches_dot_scores_bit_for_bit() {
        for len in (0..=40).chain([63, 64, 65, 131, 2048]) {
            let v = |seed: usize| -> Vec<f32> {
                (0..len)
                    .map(|i| match (i + seed) % 11 {
                        0 => -0.0,
                        1 => 1e-30,
                        2 => -3.5e8,
                        k => ((i * 37 + seed * 101) % 997) as f32 / 61.0 - 8.0 + k as f32 / 7.0,
                    })
                    .collect()
            };
            let xs = [v(1), v(2), v(3)];
            let ws = [v(4), v(5), v(6)];
            let tile = dot_tile([&xs[0], &xs[1], &xs[2]], [&ws[0], &ws[1], &ws[2]]);
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(
                        tile[i][j].to_bits(),
                        dot_scores(&xs[i], &ws[j]).to_bits(),
                        "len {len}, ({i}, {j})"
                    );
                }
            }
        }
    }

    #[test]
    fn tiled_columns_match_single_dot_products_in_every_layout() {
        // The pre-tiling kernel: one dot product per output, then bias and adapter.
        fn reference(p: &Projection<'_>, x: &[f32]) -> Vec<f32> {
            let rows = x.len() / p.input;
            let mut out = vec![0.0; rows * p.output];
            let (mut ws, mut aw) = (Vec::new(), Vec::new());
            for column in 0..p.output {
                let w = p.weights.row(column * p.input, p.input, &mut ws).to_vec();
                let adapter = p
                    .adapter
                    .as_ref()
                    .map(|a| (a, a.weights.row(column * a.rank, a.rank, &mut aw).to_vec()));
                let b = p.bias.map_or(0.0, |b| b[column]);
                for (row, token) in x.chunks_exact(p.input).enumerate() {
                    let base = dot_scores(token, &w) + b;
                    out[column * rows + row] = adapter
                        .as_ref()
                        .map_or(base, |(a, w)| a.combine(base, row, w));
                }
            }
            out
        }
        let rank = 5;
        for (rows, input, output) in [
            (1, 9, 1),
            (2, 17, 2),
            (3, 8, 3),
            (4, 33, 4),
            (7, 64, 5),
            (19, 131, 31),
            (67, 40, 7),
            (23, 257, 23),
            (24, 256, 24),
            (25, 257, 25),
            (63, 257, 47),
            (64, 256, 48),
            (65, 257, 49),
        ] {
            let x: Vec<f32> = (0..rows * input)
                .map(|i| ((i * 43 % 113) as f32 - 56.0) / 32.0)
                .collect();
            let mid: Vec<f32> = (0..rows * rank)
                .map(|i| (i % 9) as f32 / 4.0 - 1.0)
                .collect();
            let bias: Vec<f32> = (0..output).map(|i| i as f32 / 16.0 - 0.25).collect();
            let adapter_weights = Data::F32(
                (0..output * rank)
                    .map(|i| (i % 7) as f32 / 8.0 - 0.4)
                    .collect(),
            );
            let bf16 = |n: usize| {
                Data::Bf16(
                    (0..n)
                        .map(|i| ((((i * 17 % 89) as f32 - 44.0) / 32.0).to_bits() >> 16) as u16)
                        .collect(),
                )
            };
            for (weights, label) in [
                (
                    Data::F32(
                        (0..output * input)
                            .map(|i| (i % 13) as f32 / 5.0 - 1.2)
                            .collect(),
                    ),
                    "f32",
                ),
                (bf16(output * input), "bf16"),
                (
                    bf16(output * input).quantize(input, Precision::Int8),
                    "int8",
                ),
                (
                    bf16(output * input).quantize(input, Precision::Int4),
                    "int4",
                ),
            ] {
                for (with_bias, with_adapter) in [(false, false), (true, false), (true, true)] {
                    let projection = Projection {
                        input,
                        output,
                        weights: &weights,
                        bias: with_bias.then_some(&bias[..]),
                        adapter: with_adapter.then(|| Adapter {
                            inputs: &mid,
                            weights: &adapter_weights,
                            rank,
                            scale: 0.375,
                        }),
                    };
                    let expected = reference(&projection, &x);
                    // Both tile widths on every architecture, plus the selected one.
                    for actual in [
                        projection.columns(&x),
                        projection.columns_tiled::<1>(&x),
                        projection.columns_tiled::<3>(&x),
                        projection.columns_blocked::<24, 1, 24>(&x),
                        projection.columns_blocked::<24, 3, 24>(&x),
                        projection.columns_blocked::<48, 3, 16>(&x),
                    ] {
                        assert!(
                            actual
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{label} {rows}x{input}->{output} bias={with_bias} adapter={with_adapter}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "real projection cache-blocking sweep; use release and fixed workers"]
    fn benchmark_projection_blocking() {
        use std::{hint::black_box, time::Instant};
        println!(
            "blocking arch={} workers={}",
            std::env::consts::ARCH,
            rayon::current_num_threads()
        );
        let tokens: Vec<usize> = std::env::var("PROFILE_TOKENS")
            .map(|v| v.split(',').map(|n| n.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![64, 512]);
        for (input, output) in [(2048, 6144), (6144, 2048)] {
            let weights = Data::Bf16(
                (0..input * output)
                    .map(|i| ((((i * 17 % 89) as f32 - 44.0) / 32.0).to_bits() >> 16) as u16)
                    .collect(),
            );
            let p = Projection {
                input,
                output,
                weights: &weights,
                bias: None,
                adapter: None,
            };
            for &rows in &tokens {
                let x: Vec<f32> = (0..rows * input)
                    .map(|i| (i % 31) as f32 / 16.0 - 0.9)
                    .collect();
                let expected = p.columns_tiled::<TILE>(&x);
                let time = |name: &str, f: &mut dyn FnMut() -> Vec<f32>| {
                    assert_eq!(f(), expected, "{name}");
                    let start = Instant::now();
                    for _ in 0..2 {
                        black_box(f());
                    }
                    println!(
                        "blocking shape={rows}x{input}->{output} plan={name} ms={:.3}",
                        start.elapsed().as_secs_f64() * 500.0
                    );
                };
                time("original", &mut || p.columns_tiled::<TILE>(&x));
                time("c12r24", &mut || p.columns_blocked::<12, TILE, 24>(&x));
                time("c24r24", &mut || p.columns_blocked::<24, TILE, 24>(&x));
                time("c48r24", &mut || p.columns_blocked::<48, TILE, 24>(&x));
                time("c48r12", &mut || p.columns_blocked::<48, TILE, 12>(&x));
                time("c96r24", &mut || p.columns_blocked::<96, TILE, 24>(&x));
            }
        }
    }

    #[cfg(feature = "fast-gemm")]
    #[test]
    fn gemm_columns_match_reference_across_panels_and_layouts() {
        let rank = 5;
        for (rows, input, output) in [
            (1, 9, 1),
            (2, 33, 3),
            (25, 257, 25),
            (63, 1024, 129),
            (64, 1024, 128),
            (65, 1025, 131),
        ] {
            let x: Vec<f32> = (0..rows * input)
                .map(|i| (i % 31) as f32 / 16.0 - 0.9)
                .collect();
            let bias: Vec<f32> = (0..output).map(|i| (i % 7) as f32 / 13.0).collect();
            let mid: Vec<f32> = (0..rows * rank)
                .map(|i| (i % 11) as f32 / 19.0 - 0.2)
                .collect();
            let adapter_weights = Data::F32(
                (0..output * rank)
                    .map(|i| (i % 9) as f32 / 23.0 - 0.1)
                    .collect(),
            );
            let values: Vec<f32> = (0..output * input)
                .map(|i| ((i * 17 % 89) as f32 - 44.0) / (32.0 * (input as f32).sqrt()))
                .collect();
            let bf16 = || Data::Bf16(values.iter().map(|v| (v.to_bits() >> 16) as u16).collect());
            for weights in [
                Data::F32(values.clone()),
                bf16(),
                bf16().quantize(input, Precision::Int8),
                bf16().quantize(input, Precision::Int4),
            ] {
                for (with_bias, with_adapter) in [(false, false), (true, false), (true, true)] {
                    let p = Projection {
                        input,
                        output,
                        weights: &weights,
                        bias: with_bias.then_some(&bias[..]),
                        adapter: with_adapter.then_some(Adapter {
                            inputs: &mid,
                            weights: &adapter_weights,
                            rank,
                            scale: 0.375,
                        }),
                    };
                    let expected = p.columns_tiled::<1>(&x);
                    for workers in [1, 4] {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(workers)
                            .build()
                            .unwrap();
                        let actual = pool.install(|| p.columns_gemm(&x));
                        for (got, want) in actual.iter().zip(&expected) {
                            assert!(
                                (got - want).abs() <= 1e-5 + 5e-6 * want.abs(),
                                "{rows}x{input}->{output}: {got} != {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn parallel_projection_handles_bias_and_non_aligned_dimensions() {
        let (rows, input, output) = (19, 13, 31);
        let x: Vec<_> = (0..rows * input).map(|i| (i % 11) as f32 - 5.0).collect();
        let w: Vec<_> = (0..output * input).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<_> = (0..output).map(|i| i as f32 * 0.5).collect();
        let mut y = vec![0.0; rows * output];
        linear(
            &x,
            rows,
            input,
            &Data::F32(w.clone()),
            output,
            Some(&b),
            &mut y,
        );
        for row in 0..rows {
            for col in 0..output {
                let expected = b[col]
                    + (0..input)
                        .map(|k| x[row * input + k] * w[col * input + k])
                        .sum::<f32>();
                assert_eq!(y[row * output + col], expected);
            }
        }
    }
}
