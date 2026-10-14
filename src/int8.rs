//! Checked integer dot products. This is the only module permitting unsafe code.
//! The ARM implementation reads complete 16-byte chunks from bounded slices;
//! its feature-gated entry is reached only after runtime CPU detection.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

pub(crate) struct Kernel {
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    dotprod: bool,
}

impl Kernel {
    #[inline]
    pub(crate) fn accelerated(&self) -> bool {
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        {
            self.dotprod
        }
        #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
        {
            false
        }
    }

    pub(crate) fn detect() -> Self {
        Self {
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        }
    }

    /// At most 64 signed byte products, so even -128 inputs cannot overflow i32.
    #[inline]
    pub(crate) fn dot(&self, a: &[i8], b: &[u8]) -> i32 {
        assert_eq!(a.len(), b.len());
        assert!(a.len() <= 64);
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if self.dotprod {
            // SAFETY: detect() established dotprod support; equal slice lengths
            // and the 64-element bound were checked above. No mutable aliasing.
            return unsafe { arm_dot(a, b) };
        }
        scalar_dot(a, b)
    }
}

/// Rows of a tiled group-scaled product.
pub(crate) const TILE_ROWS: usize = 4;
/// Columns of a tiled group-scaled product.
pub(crate) const TILE_COLUMNS: usize = 4;

impl Kernel {
    /// FP32 sum over 64-byte groups of `dot(x group, w group) * (x scale * w scale)`,
    /// accumulated in group order. This is the reference for `scaled_tile`.
    pub(crate) fn scaled_dot(&self, x: &[i8], x_scales: &[f32], w: &[u8], w_scales: &[f32]) -> f32 {
        let mut sum = 0.0;
        for (((a, b), xs), ws) in x.chunks(64).zip(w.chunks(64)).zip(x_scales).zip(w_scales) {
            sum += self.dot(a, b) as f32 * (*xs * *ws);
        }
        sum
    }

    /// Whether `scaled_tile` uses native dot products for rows of `width` bytes.
    pub(crate) fn tiles(&self, width: usize) -> bool {
        self.accelerated() && width > 0 && width.is_multiple_of(64)
    }

    /// `scaled_dot` for every pair of 4 input rows and 4 weight rows. With native
    /// dot products, groups of all 16 outputs are computed together; integer
    /// group dots are exact, and each output's FP32 scaling and accumulation
    /// happen in the same order, so results are bit-identical to `scaled_dot`.
    pub(crate) fn scaled_tile(
        &self,
        xs: [&[i8]; TILE_ROWS],
        x_scales: [&[f32]; TILE_ROWS],
        ws: [&[u8]; TILE_COLUMNS],
        w_scales: [&[f32]; TILE_COLUMNS],
    ) -> [[f32; TILE_COLUMNS]; TILE_ROWS] {
        let width = xs[0].len();
        let groups = width.div_ceil(64);
        assert!(xs.iter().all(|x| x.len() == width) && ws.iter().all(|w| w.len() == width));
        assert!(x_scales.iter().chain(&w_scales).all(|s| s.len() == groups));
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if self.tiles(width) {
            // SAFETY: detect() established dotprod support. Every row holds
            // `groups` complete 64-byte groups and every scale slice `groups`
            // values, checked above; the kernel only reads whole 16-byte chunks
            // of those rows through bounds-checked indexing.
            return unsafe { arm_scaled_tile(xs, x_scales, ws, w_scales) };
        }
        std::array::from_fn(|r| {
            std::array::from_fn(|c| self.scaled_dot(xs[r], x_scales[r], ws[c], w_scales[c]))
        })
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "dotprod")]
unsafe fn arm_scaled_tile(
    xs: [&[i8]; TILE_ROWS],
    x_scales: [&[f32]; TILE_ROWS],
    ws: [&[u8]; TILE_COLUMNS],
    w_scales: [&[f32]; TILE_COLUMNS],
) -> [[f32; TILE_COLUMNS]; TILE_ROWS] {
    use std::arch::aarch64::{
        int32x4_t, vaddq_f32, vcvtq_f32_s32, vdotq_s32, vdupq_n_f32, vdupq_n_s32, vld1q_f32,
        vld1q_s8, vmulq_f32, vmulq_n_f32, vpaddq_s32, vst1q_f32,
    };
    let x: [&[[i8; 16]]; TILE_ROWS] = xs.map(|x| x.as_chunks::<16>().0);
    let w: [&[[u8; 16]]; TILE_COLUMNS] = ws.map(|w| w.as_chunks::<16>().0);
    let mut sums = [vdupq_n_f32(0.0); TILE_ROWS];
    for group in 0..x_scales[0].len() {
        let mut acc: [[int32x4_t; TILE_COLUMNS]; TILE_ROWS] =
            [[vdupq_n_s32(0); TILE_COLUMNS]; TILE_ROWS];
        for chunk in group * 4..group * 4 + 4 {
            // SAFETY: each array holds exactly 16 initialized bytes; the loads
            // permit unaligned pointers and stay within these immutable arrays.
            let a = x.map(|row| unsafe { vld1q_s8(row[chunk].as_ptr()) });
            let b = w.map(|row| unsafe { vld1q_s8(row[chunk].as_ptr().cast::<i8>()) });
            for r in 0..TILE_ROWS {
                for c in 0..TILE_COLUMNS {
                    acc[r][c] = vdotq_s32(acc[r][c], a[r], b[c]);
                }
            }
        }
        // Lane c of `w_group` is weight row c's scale for this group.
        let w_group = w_scales.map(|s| s[group]);
        // SAFETY: `w_group` is an initialized array of four f32 values.
        let w_group = unsafe { vld1q_f32(w_group.as_ptr()) };
        for r in 0..TILE_ROWS {
            // Exact integer reduction to one group dot per column, in lanes 0..4.
            let dots = vpaddq_s32(
                vpaddq_s32(acc[r][0], acc[r][1]),
                vpaddq_s32(acc[r][2], acc[r][3]),
            );
            // Per lane, as `scaled_dot`: dot as f32 * (x scale * w scale), then add.
            let scale = vmulq_n_f32(w_group, x_scales[r][group]);
            sums[r] = vaddq_f32(sums[r], vmulq_f32(vcvtq_f32_s32(dots), scale));
        }
    }
    sums.map(|s| {
        let mut out = [0.0; TILE_COLUMNS];
        // SAFETY: `out` has room for exactly four f32 values.
        unsafe { vst1q_f32(out.as_mut_ptr(), s) };
        out
    })
}

fn scalar_dot(a: &[i8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| i32::from(*a) * i32::from(*b as i8))
        .sum()
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "dotprod")]
unsafe fn arm_dot(a: &[i8], b: &[u8]) -> i32 {
    use std::arch::aarch64::{vaddvq_s32, vdotq_s32, vdupq_n_s32, vld1q_s8};
    let (a_chunks, a_tail) = a.as_chunks::<16>();
    let (b_chunks, b_tail) = b.as_chunks::<16>();
    let mut acc = vdupq_n_s32(0);
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        // SAFETY: each array contains exactly 16 initialized bytes. The loads
        // permit unaligned pointers and only read within these immutable arrays.
        let (a, b) = unsafe { (vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr().cast::<i8>())) };
        acc = vdotq_s32(acc, a, b);
    }
    vaddvq_s32(acc) + scalar_dot(a_tail, b_tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_kernel_matches_integer_reference_for_all_block_lengths() {
        let kernel = Kernel::detect();
        for len in 0..=64 {
            // Include -128, 127, mixed signs, and unaligned starts and tails.
            let a: Vec<i8> = (0..=len).map(|i| (i * 71 + 128) as i8).collect();
            let b: Vec<u8> = (0..=len).map(|i| (i * 37 + 128) as u8).collect();
            let expected: i32 = a[1..]
                .iter()
                .zip(&b[1..])
                .map(|(a, b)| i32::from(*a) * i32::from(*b as i8))
                .sum();
            assert_eq!(kernel.dot(&a[1..], &b[1..]), expected);
            assert_eq!(scalar_dot(&a[1..], &b[1..]), expected);
        }
        assert_eq!(kernel.dot(&[-128; 64], &[128; 64]), 1_048_576);
        assert_eq!(kernel.dot(&[-128; 64], &[127; 64]), -1_040_384);
    }

    #[test]
    fn scaled_tile_matches_scaled_dot_bit_for_bit() {
        let kernel = Kernel::detect();
        // Multiples of 64 take the native tile where available; others fall back.
        for width in [64usize, 128, 192, 2048, 100, 1] {
            let groups = width.div_ceil(64);
            let bytes = |seed: usize| -> Vec<u8> {
                (0..width)
                    .map(|i| ((i * 73 + seed * 151) % 256) as u8)
                    .collect()
            };
            let scales = |seed: usize| -> Vec<f32> {
                (0..groups)
                    .map(|g| match (g + seed) % 5 {
                        0 => 1e-30,
                        1 => 3.5e6,
                        k => (k as f32 + seed as f32 / 7.0) / 113.0,
                    })
                    .collect()
            };
            let xs: Vec<Vec<i8>> = (0..4)
                .map(|r| bytes(r).into_iter().map(|b| b as i8).collect())
                .collect();
            let ws: Vec<Vec<u8>> = (4..8).map(bytes).collect();
            let x_scales: Vec<Vec<f32>> = (0..4).map(scales).collect();
            let w_scales: Vec<Vec<f32>> = (4..8).map(scales).collect();
            let tile = kernel.scaled_tile(
                std::array::from_fn(|r| &xs[r][..]),
                std::array::from_fn(|r| &x_scales[r][..]),
                std::array::from_fn(|c| &ws[c][..]),
                std::array::from_fn(|c| &w_scales[c][..]),
            );
            for r in 0..4 {
                for c in 0..4 {
                    let expected = kernel.scaled_dot(&xs[r], &x_scales[r], &ws[c], &w_scales[c]);
                    assert_eq!(
                        tile[r][c].to_bits(),
                        expected.to_bits(),
                        "width {width} ({r}, {c})"
                    );
                }
            }
        }
        // Extremes: every product is (-128)(-128) or 127(-128).
        let x = [
            vec![-128i8; 128],
            vec![127; 128],
            vec![-128; 128],
            vec![0; 128],
        ];
        let w = [
            vec![128u8; 128],
            vec![128; 128],
            vec![127; 128],
            vec![1; 128],
        ];
        let s = [1.0f32, 0.5];
        let tile = kernel.scaled_tile(
            std::array::from_fn(|r| &x[r][..]),
            [&s[..]; 4],
            std::array::from_fn(|c| &w[c][..]),
            [&s[..]; 4],
        );
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(tile[r][c], kernel.scaled_dot(&x[r], &s, &w[c], &s));
            }
        }
        assert_eq!(tile[0][0], 64.0 * 16384.0 * 1.0 + 64.0 * 16384.0 * 0.25);
    }

    #[test]
    #[should_panic]
    fn mismatched_lengths_are_rejected() {
        Kernel::detect().dot(&[1; 16], &[1; 15]);
    }

    #[test]
    #[should_panic]
    fn oversized_blocks_are_rejected() {
        Kernel::detect().dot(&[1; 65], &[1; 65]);
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn arm_kernel_matches_forced_scalar_fallback() {
        // Our ARM CI explicitly requires acceleration. Other ARM users can run
        // the default test suite on CPUs that only support the scalar fallback.
        let kernel = Kernel::detect();
        if std::env::var("DECIDER_REQUIRE_DOTPROD").as_deref() == Ok("1") {
            assert!(
                kernel.dotprod,
                "ARM CI must exercise the accelerated kernel"
            );
        }
        let fallback = Kernel { dotprod: false };
        for len in 0..=64 {
            let a: Vec<_> = (0..len).map(|i| (i * 43) as i8).collect();
            let b: Vec<_> = (0..len).map(|i| (i * 91) as u8).collect();
            assert_eq!(kernel.dot(&a, &b), fallback.dot(&a, &b));
        }
    }
}
