//! Throwaway bit-exact register-tiling benchmark for issue #6; not for merge.
#![allow(clippy::all)]
use std::hint::black_box;
use std::time::Instant;

/// The engine's current per-output dot product.
fn dot_scores(a: &[f32], b: &[f32]) -> f32 {
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

/// R x C outputs at once; each keeps dot_scores' exact lane and summation order.
#[inline(always)]
fn tile<const R: usize, const C: usize>(xs: [&[f32]; R], ws: [&[f32]; C]) -> [[f32; C]; R] {
    let len = xs[0].len();
    let chunks = len / 8;
    let mut lanes = [[[0.0f32; 8]; C]; R];
    for k in 0..chunks {
        let xa: [&[f32; 8]; R] = std::array::from_fn(|r| xs[r][k * 8..k * 8 + 8].try_into().unwrap());
        let wb: [&[f32; 8]; C] = std::array::from_fn(|c| ws[c][k * 8..k * 8 + 8].try_into().unwrap());
        for r in 0..R {
            for c in 0..C {
                for i in 0..8 {
                    if cfg!(target_arch = "aarch64") {
                        lanes[r][c][i] = xa[r][i].mul_add(wb[c][i], lanes[r][c][i]);
                    } else {
                        lanes[r][c][i] += xa[r][i] * wb[c][i];
                    }
                }
            }
        }
    }
    let mut out = [[0.0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            let mut sum = lanes[r][c].iter().sum::<f32>();
            for i in chunks * 8..len {
                sum += xs[r][i] * ws[c][i];
            }
            out[r][c] = sum;
        }
    }
    out
}

fn current(x: &[f32], w: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * output];
    for c in 0..output {
        let wc = &w[c * input..(c + 1) * input];
        for r in 0..rows {
            out[c * rows + r] = dot_scores(&x[r * input..(r + 1) * input], wc);
        }
    }
    out
}

fn tiled<const R: usize, const C: usize>(x: &[f32], w: &[f32], rows: usize, input: usize, output: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * output];
    let mut c = 0;
    while c + C <= output {
        let ws: [&[f32]; C] = std::array::from_fn(|j| &w[(c + j) * input..(c + j + 1) * input]);
        let mut r = 0;
        while r + R <= rows {
            let xs: [&[f32]; R] = std::array::from_fn(|i| &x[(r + i) * input..(r + i + 1) * input]);
            let t = tile::<R, C>(xs, ws);
            for i in 0..R { for j in 0..C { out[(c + j) * rows + r + i] = t[i][j]; } }
            r += R;
        }
        for r in r..rows { for j in 0..C { out[(c + j) * rows + r] = dot_scores(&x[r * input..(r + 1) * input], ws[j]); } }
        c += C;
    }
    for c in c..output { for r in 0..rows { out[c * rows + r] = dot_scores(&x[r * input..(r + 1) * input], &w[c * input..(c + 1) * input]); } }
    out
}


macro_rules! tile_fn {
    ($name:ident, [$($r:literal),+], [$($c:literal),+]) => {
        #[inline(never)]
        fn $name(xs: &[&[f32]], ws: &[&[f32]], out: &mut [f32], stride: usize) {
            const R: usize = [$($r),+].len();
            const C: usize = [$($c),+].len();
            let len = xs[0].len();
            let n = len / 8;
            let mut acc = [[0.0f32; 8]; R * C];
            let xr: [&[[f32; 8]]; R] = std::array::from_fn(|i| xs[i].as_chunks::<8>().0);
            let wc: [&[[f32; 8]]; C] = std::array::from_fn(|j| ws[j].as_chunks::<8>().0);
            for k in 0..n {
                let a: [[f32; 8]; R] = std::array::from_fn(|i| xr[i][k]);
                let b: [[f32; 8]; C] = std::array::from_fn(|j| wc[j][k]);
                for i in 0..R {
                    for j in 0..C {
                        let l = &mut acc[i * C + j];
                        for q in 0..8 {
                            if cfg!(target_arch = "aarch64") { l[q] = a[i][q].mul_add(b[j][q], l[q]); } else { l[q] += a[i][q] * b[j][q]; }
                        }
                    }
                }
            }
            for i in 0..R {
                for j in 0..C {
                    let mut sum = acc[i * C + j].iter().sum::<f32>();
                    for t in n * 8..len { sum += xs[i][t] * ws[j][t]; }
                    out[j * stride + i] = sum;
                }
            }
        }
    };
}
tile_fn!(t2x2, [0, 1], [0, 1]);
tile_fn!(t2x4, [0, 1], [0, 1, 2, 3]);
tile_fn!(t4x2, [0, 1, 2, 3], [0, 1]);
tile_fn!(t3x3, [0, 1, 2], [0, 1, 2]);
tile_fn!(t4x4, [0, 1, 2, 3], [0, 1, 2, 3]);
tile_fn!(t1x4, [0], [0, 1, 2, 3]);

fn tiled2(x: &[f32], w: &[f32], rows: usize, input: usize, output: usize, r_n: usize, c_n: usize,
          f: fn(&[&[f32]], &[&[f32]], &mut [f32], usize)) -> Vec<f32> {
    let mut out = vec![0.0; rows * output];
    let mut c = 0;
    while c + c_n <= output {
        let ws: Vec<&[f32]> = (0..c_n).map(|j| &w[(c + j) * input..(c + j + 1) * input]).collect();
        let mut r = 0;
        while r + r_n <= rows {
            let xs: Vec<&[f32]> = (0..r_n).map(|i| &x[(r + i) * input..(r + i + 1) * input]).collect();
            f(&xs, &ws, &mut out[c * rows + r..], rows);
            r += r_n;
        }
        for r in r..rows { for j in 0..c_n { out[(c + j) * rows + r] = dot_scores(&x[r * input..(r + 1) * input], ws[j]); } }
        c += c_n;
    }
    for c in c..output { for r in 0..rows { out[c * rows + r] = dot_scores(&x[r * input..(r + 1) * input], &w[c * input..(c + 1) * input]); } }
    out
}

fn bench(name: &str, f: impl Fn() -> Vec<f32>, flops: f64) -> Vec<f32> {
    let out = f();
    let start = Instant::now();
    for _ in 0..3 { black_box(f()); }
    let s = start.elapsed().as_secs_f64() / 3.0;
    println!("{name:>10}: {:8.2} ms {:6.1} GFLOP/s", s * 1000.0, flops / s / 1e9);
    out
}

fn main() {
    println!("arch={} workers=1", std::env::consts::ARCH);
    let (input, output) = (2048usize, 1536usize);
    let w: Vec<f32> = (0..input * output).map(|i| ((i * 17 % 89) as f32 - 44.0) / 64.0).collect();
    for rows in [1usize, 8, 32, 128] {
        let x: Vec<f32> = (0..rows * input).map(|i| ((i * 43 % 113) as f32 - 56.0) / 127.0).collect();
        let flops = 2.0 * (rows * input * output) as f64;
        println!("rows={rows} (1 thread, {input}x{output})");
        let base = bench("current", || current(&x, &w, rows, input, output), flops);
        for (name, r_n, c_n, f) in [("e2x2", 2, 2, t2x2 as fn(&[&[f32]], &[&[f32]], &mut [f32], usize)), ("e2x4", 2, 4, t2x4), ("e4x2", 4, 2, t4x2), ("e3x3", 3, 3, t3x3), ("e4x4", 4, 4, t4x4), ("e1x4", 1, 4, t1x4)] {
            let o = bench(name, || tiled2(&x, &w, rows, input, output, r_n, c_n, f), flops);
            assert!(o.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()), "not bit-exact");
        }
    }
}
