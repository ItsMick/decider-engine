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

/// Operations applied while transposing completed projection values.
#[derive(Clone, Copy)]
pub(crate) enum Epilogue {
    Write,
    Add,
    SwiGlu,
}
impl Epilogue {
    fn store(self, dst: &mut f32, value: f32) {
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
            .columns(x);
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
            let columns = self.columns(x);
            write_columns(&columns, None, rows, output, result, epilogue);
        }
    }

    fn columns(&self, x: &[f32]) -> Vec<f32> {
        let (input, output) = (self.input, self.output);
        let rows = x.len() / input;
        if let Data::Quantized {
            bits: 8,
            compute_int8: true,
            ..
        } = self.weights
        {
            if let Some(columns) = linear_int8(x, self) {
                return columns;
            }
        }
        let compute = |column: usize, dst: &mut [f32], scratch: &mut (Vec<f32>, Vec<f32>)| {
            let w = self.weights.row(column * input, input, &mut scratch.0);
            let adapter = self
                .adapter
                .as_ref()
                .map(|a| (a, a.weights.row(column * a.rank, a.rank, &mut scratch.1)));
            let b = self.bias.map_or(0.0, |b| b[column]);
            for (row, (value, token)) in dst.iter_mut().zip(x.chunks_exact(input)).enumerate() {
                let base = dot_scores(token, w) + b;
                *value = adapter.map_or(base, |(a, w)| a.combine(base, row, w));
            }
        };
        let mut columns = vec![0.0; rows * output];
        if rows * output >= 512 {
            columns.par_chunks_mut(rows).enumerate().for_each_init(
                || (Vec::new(), Vec::new()),
                |scratch, (column, dst)| compute(column, dst, scratch),
            );
        } else {
            let mut scratch = (Vec::new(), Vec::new());
            for (column, dst) in columns.chunks_mut(rows).enumerate() {
                compute(column, dst, &mut scratch);
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

struct QuantizedInputs {
    values: Vec<i8>,
    scales: Vec<f32>,
}
impl QuantizedInputs {
    fn new(x: &[f32], width: usize) -> Option<Self> {
        // Preserve FP32 propagation for non-finite inputs rather than silently
        // turning NaNs or infinities into integer zeroes or saturated values.
        if !x.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut result = Self {
            values: Vec::with_capacity(x.len()),
            scales: Vec::with_capacity(x.len() / width * width.div_ceil(64)),
        };
        for row in x.chunks(width) {
            for group in row.chunks(64) {
                let max = group.iter().map(|v| v.abs()).fold(0.0, f32::max);
                let scale = if max == 0.0 {
                    1.0
                } else {
                    (max / 127.0).max(f32::MIN_POSITIVE)
                };
                result.scales.push(scale);
                result.values.extend(
                    group
                        .iter()
                        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8),
                );
            }
        }
        Some(result)
    }
}

fn linear_int8(x: &[f32], projection: &Projection<'_>) -> Option<Vec<f32>> {
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
    let inputs = QuantizedInputs::new(x, input)?;
    let groups = input.div_ceil(64);
    let kernel = Kernel::detect();
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
