//! Opt-in NEON kernels, compiled only with `--features arm-neon` on little-endian
//! AArch64. Unsafe code here is limited to loads and stores within fixed-size
//! arrays; every kernel reproduces its portable counterpart bit for bit.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

// NEON is part of the AArch64 baseline, which the entry points below rely on.
#[cfg(not(target_feature = "neon"))]
compile_error!("the arm-neon feature requires an AArch64 target with NEON");

/// Load four floats.
#[inline(always)]
fn load(values: &[f32; 4]) -> float32x4_t {
    // SAFETY: `values` is an initialized array of exactly four f32 values, and
    // the load permits unaligned addresses.
    unsafe { vld1q_f32(values.as_ptr()) }
}

/// Store four floats.
#[inline(always)]
fn store(values: float32x4_t) -> [f32; 4] {
    let mut out = [0.0; 4];
    // SAFETY: `out` has room for exactly four f32 values.
    unsafe { vst1q_f32(out.as_mut_ptr(), values) };
    out
}

/// `tensor::dot_tile` with explicit NEON registers. The portable version leaves
/// its accumulators on the stack each iteration; these 18 vector accumulators
/// (nine outputs of eight lanes) stay in registers.
///
/// Each output reproduces `dot_scores` on AArch64: lane `q` accumulates
/// `x[q] * w[q]` with one fused multiply-add per 8-float chunk (`vfmaq_f32`
/// rounds once, like `f32::mul_add`), then the eight lanes are summed in order
/// and the tail is added with a separate multiply and add.
pub(crate) fn dot_tile(xs: [&[f32]; 3], ws: [&[f32]; 3]) -> [[f32; 3]; 3] {
    // SAFETY: NEON is enabled for every build of this module (checked above).
    unsafe { dot_tile_neon(xs, ws) }
}

#[target_feature(enable = "neon")]
fn dot_tile_neon(xs: [&[f32]; 3], ws: [&[f32]; 3]) -> [[f32; 3]; 3] {
    let len = xs[0].len();
    assert!(xs.iter().chain(&ws).all(|v| v.len() == len));
    let chunks = len / 8;
    let [x0, x1, x2] = xs.map(|x| x.as_chunks::<4>().0);
    let [w0, w1, w2] = ws.map(|w| w.as_chunks::<4>().0);
    // acc[i][j][h]: output (i, j), lanes 0-3 (h = 0) or 4-7 (h = 1).
    let mut acc = [[[vdupq_n_f32(0.0); 2]; 3]; 3];
    for k in 0..chunks {
        let (lo, hi) = (2 * k, 2 * k + 1);
        let x = [
            [load(&x0[lo]), load(&x0[hi])],
            [load(&x1[lo]), load(&x1[hi])],
            [load(&x2[lo]), load(&x2[hi])],
        ];
        let w = [
            [load(&w0[lo]), load(&w0[hi])],
            [load(&w1[lo]), load(&w1[hi])],
            [load(&w2[lo]), load(&w2[hi])],
        ];
        for i in 0..3 {
            for j in 0..3 {
                for h in 0..2 {
                    acc[i][j][h] = vfmaq_f32(acc[i][j][h], x[i][h], w[j][h]);
                }
            }
        }
    }
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let [l0, l1, l2, l3] = store(acc[i][j][0]);
            let [l4, l5, l6, l7] = store(acc[i][j][1]);
            let mut sum = [l0, l1, l2, l3, l4, l5, l6, l7].iter().sum::<f32>();
            for t in chunks * 8..len {
                sum += xs[i][t] * ws[j][t];
            }
            sum
        })
    })
}
