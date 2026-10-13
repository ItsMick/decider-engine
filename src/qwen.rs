//! Text-only Qwen3.5: full causal GQA and recurrent Gated DeltaNet.
//! FP32 arithmetic matches the upstream CPU serving path. No generation or KV cache.
use crate::tensor::{dot_scores, softmax, Epilogue};
use crate::weights::{Data, Linear, Weights};
use rayon::prelude::*;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub rope_parameters: Rope,
    pub attention_bias: bool,
    pub attn_output_gate: bool,
    pub hidden_act: String,
}
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Rope {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub rope_type: String,
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        let dimensions = [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.vocab_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        ];
        if dimensions.iter().any(|d| *d == 0 || *d > 1_000_000) {
            return Err("invalid Qwen dimensions".into());
        }
        if self.layer_types.len() != self.num_hidden_layers
            || self
                .layer_types
                .iter()
                .any(|s| s != "linear_attention" && s != "full_attention")
        {
            return Err("invalid Qwen layer_types".into());
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            return Err("query/value head counts must divide key head counts".into());
        }
        let r = self.rotary_dim();
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || self.attention_bias
            || !self.attn_output_gate
            || self.hidden_act != "silu"
            || self.rope_parameters.rope_type != "default"
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
            || !self.rope_parameters.partial_rotary_factor.is_finite()
            || r == 0
            || r > self.head_dim
            || !r.is_multiple_of(2)
        {
            return Err(
                "unsupported Qwen normalization, activation, attention, or RoPE config".into(),
            );
        }
        Ok(())
    }
    fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope_parameters.partial_rotary_factor) as usize
    }
}

pub(crate) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

pub(crate) fn rms(x: &mut [f32], w: &[f32], eps: f32, centered: bool) {
    for row in x.chunks_exact_mut(w.len()) {
        let inv = (row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32 + eps)
            .sqrt()
            .recip();
        for (v, g) in row.iter_mut().zip(w) {
            *v *= inv * (if centered { 1.0 + g } else { *g });
        }
    }
}
fn residual_norm(x: &mut [f32], residual: &[f32], normalized: &mut [f32], w: &[f32], eps: f32) {
    assert_eq!(x.len(), residual.len());
    assert_eq!(x.len(), normalized.len());
    for ((x, residual), dst) in x
        .chunks_exact_mut(w.len())
        .zip(residual.chunks_exact(w.len()))
        .zip(normalized.chunks_exact_mut(w.len()))
    {
        let mut sum = -0.0;
        for ((x, residual), dst) in x.iter_mut().zip(residual).zip(dst.iter_mut()) {
            *x += residual;
            *dst = *x;
            sum += *x * *x;
        }
        let inv = (sum / w.len() as f32 + eps).sqrt().recip();
        for (v, g) in dst.iter_mut().zip(w) {
            *v *= inv * (1.0 + g);
        }
    }
}
fn rotary(x: &mut [f32], angles: &[(f32, f32)]) {
    let half = angles.len();
    for (i, &(sin, cos)) in angles.iter().enumerate() {
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = b * cos + a * sin;
    }
}

struct Full {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}
struct Delta {
    qkv: Linear,
    z: Linear,
    a: Linear,
    b: Linear,
    out: Linear,
    conv: Vec<f32>,
    dt: Vec<f32>,
    decay_rate: Vec<f32>,
    norm: Vec<f32>,
}
enum Mixer {
    Full(Box<Full>),
    Delta(Box<Delta>),
}
struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub(crate) struct Model {
    config: Config,
    embedding: Data,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    rotary_divisors: Vec<f32>,
}

impl Model {
    pub fn load(
        mut w: Weights,
        adapter: &mut Weights,
        c: Config,
        rank: usize,
        scale: f32,
    ) -> Result<Self, String> {
        c.validate()?;
        let h = c.hidden_size;
        let mut layers = Vec::new();
        for i in 0..c.num_hidden_layers {
            let p = format!("layers.{i}");
            let mut linear = |name: &str, input, output| -> Result<Linear, String> {
                let key = format!("{p}.{name}");
                let mut l = Linear::load(&mut w, &key, input, output, false)?;
                l.adapter(adapter, &key, rank, scale)?;
                Ok(l)
            };
            let mixer = if c.layer_types[i] == "full_attention" {
                Mixer::Full(Box::new(Full {
                    q: linear(
                        "self_attn.q_proj",
                        h,
                        c.num_attention_heads * c.head_dim * 2,
                    )?,
                    k: linear("self_attn.k_proj", h, c.num_key_value_heads * c.head_dim)?,
                    v: linear("self_attn.v_proj", h, c.num_key_value_heads * c.head_dim)?,
                    o: linear("self_attn.o_proj", c.num_attention_heads * c.head_dim, h)?,
                    q_norm: w.take(&format!("{p}.self_attn.q_norm.weight"), &[c.head_dim])?,
                    k_norm: w.take(&format!("{p}.self_attn.k_norm.weight"), &[c.head_dim])?,
                }))
            } else {
                let kd = c.linear_key_head_dim * c.linear_num_key_heads;
                let vd = c.linear_value_head_dim * c.linear_num_value_heads;
                Mixer::Delta(Box::new(Delta {
                    qkv: linear("linear_attn.in_proj_qkv", h, 2 * kd + vd)?,
                    z: linear("linear_attn.in_proj_z", h, vd)?,
                    a: linear("linear_attn.in_proj_a", h, c.linear_num_value_heads)?,
                    b: linear("linear_attn.in_proj_b", h, c.linear_num_value_heads)?,
                    out: linear("linear_attn.out_proj", vd, h)?,
                    conv: w.take(
                        &format!("{p}.linear_attn.conv1d.weight"),
                        &[2 * kd + vd, 1, c.linear_conv_kernel_dim],
                    )?,
                    dt: w.take(
                        &format!("{p}.linear_attn.dt_bias"),
                        &[c.linear_num_value_heads],
                    )?,
                    decay_rate: w
                        .take(
                            &format!("{p}.linear_attn.A_log"),
                            &[c.linear_num_value_heads],
                        )?
                        .into_iter()
                        .map(|v| -v.exp())
                        .collect(),
                    norm: w.take(
                        &format!("{p}.linear_attn.norm.weight"),
                        &[c.linear_value_head_dim],
                    )?,
                }))
            };
            // Finish loading projections before borrowing the weight map for norms.
            let mut load_mlp = |name: &str, input, output| -> Result<Linear, String> {
                let key = format!("{p}.mlp.{name}");
                let mut l = Linear::load(&mut w, &key, input, output, false)?;
                l.adapter(adapter, &key, rank, scale)?;
                Ok(l)
            };
            let gate = load_mlp("gate_proj", h, c.intermediate_size)?;
            let up = load_mlp("up_proj", h, c.intermediate_size)?;
            let down = load_mlp("down_proj", c.intermediate_size, h)?;
            layers.push(Layer {
                mixer,
                gate,
                up,
                down,
                input_norm: w.take(&format!("{p}.input_layernorm.weight"), &[h])?,
                post_norm: w.take(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
            });
        }
        let embedding = w.take_data("embed_tokens.weight", &[c.vocab_size, h])?;
        let norm = w.take("norm.weight", &[h])?;
        if adapter.remaining() != 0 {
            return Err(format!(
                "{} unconsumed LoRA tensors: unsupported adapter targets",
                adapter.remaining()
            ));
        }
        if w.remaining() != 0 {
            return Err(format!(
                "{} unconsumed text tensors: unsupported base architecture",
                w.remaining()
            ));
        }
        let r = c.rotary_dim();
        let rotary_divisors = if c.layer_types.iter().any(|s| s == "full_attention") {
            (0..r / 2)
                .map(|i| c.rope_parameters.rope_theta.powf((2 * i) as f32 / r as f32))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            config: c,
            embedding,
            layers,
            norm,
            rotary_divisors,
        })
    }
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("empty token sequence".into());
        }
        let c = &self.config;
        let mut x = Vec::with_capacity(ids.len() * c.hidden_size);
        let mut scratch = Vec::new();
        for id in ids {
            if *id as usize >= c.vocab_size {
                return Err(format!("token {id} outside vocabulary"));
            }
            let offset = *id as usize * c.hidden_size;
            x.extend_from_slice(self.embedding.row(offset, c.hidden_size, &mut scratch));
        }
        // Position-dependent values are shared by all heads and full-attention
        // layers in this forward pass; no request state is cached on the model.
        let angles: Vec<_> = (0..ids.len())
            .flat_map(|pos| {
                self.rotary_divisors
                    .iter()
                    .map(move |divisor| (pos as f32 / divisor).sin_cos())
            })
            .collect();
        let mut n = vec![0.0; x.len()];
        for layer in &self.layers {
            n.copy_from_slice(&x);
            rms(&mut n, &layer.input_norm, c.rms_norm_eps, true);
            let mixed = match &layer.mixer {
                Mixer::Full(m) => m.forward(&n, c, &angles),
                Mixer::Delta(m) => m.forward(&n, c),
            };
            residual_norm(&mut x, &mixed, &mut n, &layer.post_norm, c.rms_norm_eps);
            let mut gate = layer.gate.apply(&n);
            layer.up.apply_into(&n, &mut gate, Epilogue::SwiGlu);
            layer.down.apply_into(&gate, &mut x, Epilogue::Add);
        }
        rms(&mut x, &self.norm, c.rms_norm_eps, true);
        if x.iter().any(|v| !v.is_finite()) {
            return Err("non-finite Qwen hidden states".into());
        }
        Ok(x)
    }
}

impl Full {
    fn forward(&self, x: &[f32], c: &Config, angles: &[(f32, f32)]) -> Vec<f32> {
        let n = x.len() / c.hidden_size;
        let d = c.head_dim;
        let qw = c.num_attention_heads * d;
        let kw = c.num_key_value_heads * d;
        let raw_q = self.q.apply(x);
        let mut q = vec![0.0; n * qw];
        let mut gate = q.clone();
        for t in 0..n {
            for h in 0..c.num_attention_heads {
                let src = t * qw * 2 + h * d * 2;
                let dst = t * qw + h * d;
                q[dst..dst + d].copy_from_slice(&raw_q[src..src + d]);
                gate[dst..dst + d].copy_from_slice(&raw_q[src + d..src + 2 * d]);
            }
        }
        let mut k = self.k.apply(x);
        let v = self.v.apply(x);
        rms(&mut q, &self.q_norm, c.rms_norm_eps, true);
        rms(&mut k, &self.k_norm, c.rms_norm_eps, true);
        let half = c.rotary_dim() / 2;
        for t in 0..n {
            let angles = &angles[t * half..(t + 1) * half];
            for h in 0..c.num_attention_heads {
                rotary(&mut q[t * qw + h * d..t * qw + (h + 1) * d], angles);
            }
            for h in 0..c.num_key_value_heads {
                rotary(&mut k[t * kw + h * d..t * kw + (h + 1) * d], angles);
            }
        }
        let mut out = causal_attention(&q, &k, &v, c.num_attention_heads, c.num_key_value_heads, d);
        for (o, g) in out.iter_mut().zip(gate) {
            *o *= sigmoid(g);
        }
        self.o.apply(&out)
    }
}

fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    key_heads: usize,
    d: usize,
) -> Vec<f32> {
    let qw = heads * d;
    let kw = key_heads * d;
    let n = q.len() / qw;
    assert_eq!(q.len(), n * qw);
    assert_eq!(k.len(), n * kw);
    assert_eq!(v.len(), k.len());
    let mut out = vec![0.0; n * qw];
    let scale = (d as f32).sqrt().recip();
    // Each token's causal attention reads immutable Q/K/V and writes a
    // disjoint output row. Score buffers belong to the sequential loop or
    // a Rayon job rather than being allocated once per token and head.
    let compute = |t: usize, token_out: &mut [f32], scores: &mut Vec<f32>| {
        for h in 0..heads {
            let kh = h / (heads / key_heads);
            let query = &q[t * qw + h * d..t * qw + (h + 1) * d];
            scores.clear();
            scores
                .extend((0..=t).map(|s| {
                    dot_scores(query, &k[s * kw + kh * d..s * kw + (kh + 1) * d]) * scale
                }));
            softmax(scores);
            let dst = &mut token_out[h * d..(h + 1) * d];
            for (s, p) in scores.iter().enumerate() {
                for (o, a) in dst
                    .iter_mut()
                    .zip(&v[s * kw + kh * d..s * kw + (kh + 1) * d])
                {
                    *o += p * a;
                }
            }
        }
    };
    if n >= 32 && qw >= 128 && rayon::current_num_threads() > 1 {
        out.par_chunks_mut(qw).enumerate().for_each_init(
            || Vec::with_capacity(n),
            |scores, (t, row)| compute(t, row, scores),
        );
    } else {
        let mut scores = Vec::with_capacity(n);
        for (t, row) in out.chunks_exact_mut(qw).enumerate() {
            compute(t, row, &mut scores);
        }
    }
    out
}

impl Delta {
    fn forward(&self, x: &[f32], c: &Config) -> Vec<f32> {
        let n = x.len() / c.hidden_size;
        let dk = c.linear_key_head_dim;
        let dv = c.linear_value_head_dim;
        let kd = c.linear_num_key_heads * dk;
        let vd = c.linear_num_value_heads * dv;
        let width = 2 * kd + vd;
        let raw = self.qkv.apply(x);
        let z = self.z.apply(x);
        let a = self.a.apply(x);
        let b = self.b.apply(x);
        let mut conv = vec![0.0; raw.len()];
        let kernel = c.linear_conv_kernel_dim;
        for t in 0..n {
            for channel in 0..width {
                let mut sum = 0.0;
                for j in 0..kernel {
                    if t + j + 1 >= kernel {
                        sum += raw[(t + j + 1 - kernel) * width + channel]
                            * self.conv[channel * kernel + j];
                    }
                }
                conv[t * width + channel] = silu(sum);
            }
        }
        let mut out = vec![0.0; n * vd];
        let mut state = vec![0.0; c.linear_num_value_heads * dk * dv];
        let mut q = vec![0.0; dk];
        let mut k = vec![0.0; dk];
        let mut delta = vec![0.0; dv];
        for t in 0..n {
            for head in 0..c.linear_num_value_heads {
                let kh = head / (c.linear_num_value_heads / c.linear_num_key_heads);
                q.copy_from_slice(&conv[t * width + kh * dk..t * width + (kh + 1) * dk]);
                k.copy_from_slice(&conv[t * width + kd + kh * dk..t * width + kd + (kh + 1) * dk]);
                let qnorm = (q.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip()
                    / (dk as f32).sqrt();
                let knorm = (k.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip();
                for v in &mut q {
                    *v *= qnorm;
                }
                for v in &mut k {
                    *v *= knorm;
                }
                let decay = (self.decay_rate[head]
                    * softplus(a[t * c.linear_num_value_heads + head] + self.dt[head]))
                .exp();
                let beta = sigmoid(b[t * c.linear_num_value_heads + head]);
                let st = &mut state[head * dk * dv..(head + 1) * dk * dv];
                for v in st.iter_mut() {
                    *v *= decay;
                }
                delta.fill(0.0);
                for (row, ki) in st.chunks_exact(dv).zip(&k) {
                    for (m, s) in delta.iter_mut().zip(row) {
                        *m += ki * s;
                    }
                }
                for (m, v) in delta.iter_mut().zip(
                    &conv[t * width + 2 * kd + head * dv..t * width + 2 * kd + (head + 1) * dv],
                ) {
                    *m = (v - *m) * beta;
                }
                let dst = &mut out[t * vd + head * dv..t * vd + (head + 1) * dv];
                for ((row, ki), qi) in st.chunks_exact_mut(dv).zip(&k).zip(&q) {
                    for ((s, de), o) in row.iter_mut().zip(&delta).zip(dst.iter_mut()) {
                        *s += ki * de;
                        *o += qi * (*s);
                    }
                }
                rms(dst, &self.norm, c.rms_norm_eps, false);
                for (o, g) in dst
                    .iter_mut()
                    .zip(&z[t * vd + head * dv..t * vd + (head + 1) * dv])
                {
                    *o *= silu(*g);
                }
            }
        }
        self.out.apply(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_norm_matches_separate_residual_copy_and_normalization() {
        for width in [1, 8, 131] {
            let x: Vec<f32> = (0..width * 3)
                .map(|i| ((i * 23 % 79) as f32 - 39.0) / 17.0)
                .collect();
            let residual: Vec<f32> = (0..x.len())
                .map(|i| ((i * 41 % 113) as f32 - 56.0) / 29.0)
                .collect();
            let weights: Vec<f32> = (0..width).map(|i| i as f32 / 19.0 - 0.5).collect();
            let expected_x: Vec<f32> = x.iter().zip(&residual).map(|(a, b)| a + b).collect();
            let mut expected_norm = expected_x.clone();
            rms(&mut expected_norm, &weights, 1e-6, true);
            let mut fused_x = x;
            let mut fused_norm = vec![0.0; fused_x.len()];
            residual_norm(&mut fused_x, &residual, &mut fused_norm, &weights, 1e-6);
            assert_eq!(fused_x, expected_x);
            assert_eq!(fused_norm, expected_norm);
        }
    }

    #[test]
    #[ignore = "synthetic attention timing; run with --release"]
    fn benchmark_causal_attention() {
        use std::{hint::black_box, time::Instant};
        let (heads, key_heads, d) = (8, 2, 256);
        let pools: Vec<_> = [1, 4]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        for n in [32, 128, 512] {
            let q: Vec<f32> = (0..n * heads * d)
                .map(|i| ((i * 71 % 113) as f32 - 56.0) / 47.0)
                .collect();
            let k: Vec<f32> = (0..n * key_heads * d)
                .map(|i| ((i * 43 % 97) as f32 - 48.0) / 39.0)
                .collect();
            let v: Vec<f32> = (0..k.len())
                .map(|i| ((i * 13 % 89) as f32 - 44.0) / 33.0)
                .collect();
            for pool in &pools {
                pool.install(|| {
                    black_box(causal_attention(&q, &k, &v, heads, key_heads, d));
                });
                let start = Instant::now();
                for _ in 0..3 {
                    pool.install(|| {
                        black_box(causal_attention(black_box(&q), &k, &v, heads, key_heads, d));
                    });
                }
                println!(
                    "attention tokens={n} workers={} mean_ms={:.3}",
                    pool.current_num_threads(),
                    start.elapsed().as_secs_f64() * 1000.0 / 3.0
                );
            }
        }
    }

    #[test]
    fn parallel_attention_matches_serial_and_preserves_causal_isolation() {
        let (heads, key_heads, d) = (2, 1, 64);
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        // Include the dispatch boundary and a partial final group of tokens.
        for n in [31, 32, 35] {
            let q: Vec<f32> = (0..n * heads * d)
                .map(|i| ((i * 71 % 113) as f32 - 56.0) / 47.0)
                .collect();
            let mut k: Vec<f32> = (0..n * key_heads * d)
                .map(|i| ((i * 43 % 97) as f32 - 48.0) / 39.0)
                .collect();
            let mut v: Vec<f32> = (0..k.len())
                .map(|i| ((i * 13 % 89) as f32 - 44.0) / 33.0)
                .collect();
            let expected = serial.install(|| causal_attention(&q, &k, &v, heads, key_heads, d));
            let actual = parallel.install(|| causal_attention(&q, &k, &v, heads, key_heads, d));
            assert_eq!(actual, expected);
            k[(n - 3) * key_heads * d..].fill(7.0);
            v[(n - 3) * key_heads * d..].fill(-11.0);
            let changed = parallel.install(|| causal_attention(&q, &k, &v, heads, key_heads, d));
            assert_eq!(
                &actual[..(n - 3) * heads * d],
                &changed[..(n - 3) * heads * d]
            );
            assert_ne!(
                &actual[(n - 3) * heads * d..],
                &changed[(n - 3) * heads * d..]
            );
        }
    }
}
