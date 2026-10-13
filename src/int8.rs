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

fn scalar_dot(a: &[i8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| i32::from(*a) * i32::from(*b as i8))
        .sum()
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "dotprod")]
unsafe fn arm_dot(a: &[i8], b: &[u8]) -> i32 {
    use std::arch::aarch64::{vaddvq_s32, vdupq_n_s32, vld1q_s8};
    let (a_chunks, a_tail) = a.as_chunks::<16>();
    let (b_chunks, b_tail) = b.as_chunks::<16>();
    let mut acc = vdupq_n_s32(0);
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        // SAFETY: each array contains exactly 16 initialized bytes. The loads
        // permit unaligned pointers and only read within these immutable arrays.
        let (a, b) = unsafe { (vld1q_s8(a.as_ptr()), vld1q_s8(b.as_ptr().cast::<i8>())) };
        // Rust 1.88 does not expose a stable SDOT intrinsic. This register-only
        // instruction keeps the minimum compiler version without nightly Rust.
        // SAFETY: dotprod is enabled by the caller; all operands are initialized
        // vector registers. SDOT has no memory, stack, or condition flag effects.
        unsafe {
            std::arch::asm!(
                "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                acc = inout(vreg) acc,
                a = in(vreg) a,
                b = in(vreg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
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
