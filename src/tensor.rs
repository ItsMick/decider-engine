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

/// Weight rows are reused across tokens; transpose the temporary output once.
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
    assert_eq!(weights.len(), output * input);
    assert_eq!(result.len(), rows * output);
    assert!(bias.is_none_or(|b| b.len() == output));
    if rows == 0 {
        return;
    }
    if let Data::Quantized {
        bits: 8,
        compute_int8: true,
        ..
    } = weights
    {
        if linear_int8(x, rows, input, weights, output, bias, result) {
            return;
        }
    }
    let compute = |column: usize, dst: &mut [f32], scratch: &mut Vec<f32>| {
        let w = weights.row(column * input, input, scratch);
        let b = bias.map_or(0.0, |b| b[column]);
        for (value, token) in dst.iter_mut().zip(x.chunks_exact(input)) {
            *value = dot_scores(token, w) + b;
        }
    };
    let mut columns = vec![0.0; result.len()];
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
    for (row, dst) in result.chunks_exact_mut(output).enumerate() {
        for (column, value) in dst.iter_mut().enumerate() {
            *value = columns[column * rows + row];
        }
    }
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

fn linear_int8(
    x: &[f32],
    rows: usize,
    input: usize,
    weights: &Data,
    output: usize,
    bias: Option<&[f32]>,
    result: &mut [f32],
) -> bool {
    let Data::Quantized {
        values,
        scales,
        width,
        bits: 8,
        ..
    } = weights
    else {
        unreachable!("integer projection requires INT8 weights");
    };
    assert_eq!(*width, input);
    let Some(inputs) = QuantizedInputs::new(x, input) else {
        return false;
    };
    let groups = input.div_ceil(64);
    let kernel = Kernel::detect();
    let compute = |column: usize, dst: &mut [f32]| {
        let w = &values[column * input..(column + 1) * input];
        let w_scales = &scales[column * groups..(column + 1) * groups];
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
            *value = sum + bias.map_or(0.0, |b| b[column]);
        }
    };
    let mut columns = vec![0.0; result.len()];
    if rows * output >= 512 {
        columns
            .par_chunks_mut(rows)
            .enumerate()
            .for_each(|(column, dst)| compute(column, dst));
    } else {
        for (column, dst) in columns.chunks_mut(rows).enumerate() {
            compute(column, dst);
        }
    }
    for (row, dst) in result.chunks_exact_mut(output).enumerate() {
        for (column, value) in dst.iter_mut().enumerate() {
            *value = columns[column * rows + row];
        }
    }
    true
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
