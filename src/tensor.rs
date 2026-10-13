//! Small portable FP32 kernels. Each parallel job owns a disjoint output slice.
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
