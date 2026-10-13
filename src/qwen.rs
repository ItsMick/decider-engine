//! Text-only Qwen3.5: full causal GQA and recurrent Gated DeltaNet.
//! FP32 arithmetic matches the upstream CPU serving path. No generation or KV cache.
use crate::tensor::{dot_scores, softmax};
use crate::weights::{Linear, Weights};
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
fn normalized(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut y = x.to_vec();
    rms(&mut y, w, eps, true);
    y
}
fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
fn rotary(x: &mut [f32], pos: usize, c: &Config) {
    let r = c.rotary_dim();
    for i in 0..r / 2 {
        let angle = pos as f32 / c.rope_parameters.rope_theta.powf((2 * i) as f32 / r as f32);
        let (sin, cos) = angle.sin_cos();
        let a = x[i];
        let b = x[i + r / 2];
        x[i] = a * cos - b * sin;
        x[i + r / 2] = b * cos + a * sin;
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
    a_log: Vec<f32>,
    norm: Vec<f32>,
}
enum Mixer {
    Full(Full),
    Delta(Delta),
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
    embedding: Vec<f32>,
    layers: Vec<Layer>,
    norm: Vec<f32>,
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
                Mixer::Full(Full {
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
                })
            } else {
                let kd = c.linear_key_head_dim * c.linear_num_key_heads;
                let vd = c.linear_value_head_dim * c.linear_num_value_heads;
                Mixer::Delta(Delta {
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
                    a_log: w.take(
                        &format!("{p}.linear_attn.A_log"),
                        &[c.linear_num_value_heads],
                    )?,
                    norm: w.take(
                        &format!("{p}.linear_attn.norm.weight"),
                        &[c.linear_value_head_dim],
                    )?,
                })
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
        let embedding = w.take("embed_tokens.weight", &[c.vocab_size, h])?;
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
        Ok(Self {
            config: c,
            embedding,
            layers,
            norm,
        })
    }
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("empty token sequence".into());
        }
        let c = &self.config;
        let mut x = Vec::with_capacity(ids.len() * c.hidden_size);
        for id in ids {
            if *id as usize >= c.vocab_size {
                return Err(format!("token {id} outside vocabulary"));
            }
            let offset = *id as usize * c.hidden_size;
            x.extend_from_slice(&self.embedding[offset..offset + c.hidden_size]);
        }
        for layer in &self.layers {
            let n = normalized(&x, &layer.input_norm, c.rms_norm_eps);
            let mixed = match &layer.mixer {
                Mixer::Full(m) => m.forward(&n, c),
                Mixer::Delta(m) => m.forward(&n, c),
            };
            add(&mut x, &mixed);
            let n = normalized(&x, &layer.post_norm, c.rms_norm_eps);
            let mut gate = layer.gate.apply(&n);
            let up = layer.up.apply(&n);
            for (g, u) in gate.iter_mut().zip(up) {
                *g = silu(*g) * u;
            }
            add(&mut x, &layer.down.apply(&gate));
        }
        rms(&mut x, &self.norm, c.rms_norm_eps, true);
        if x.iter().any(|v| !v.is_finite()) {
            return Err("non-finite Qwen hidden states".into());
        }
        Ok(x)
    }
}

impl Full {
    fn forward(&self, x: &[f32], c: &Config) -> Vec<f32> {
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
        for t in 0..n {
            for h in 0..c.num_attention_heads {
                rotary(&mut q[t * qw + h * d..t * qw + (h + 1) * d], t, c);
            }
            for h in 0..c.num_key_value_heads {
                rotary(&mut k[t * kw + h * d..t * kw + (h + 1) * d], t, c);
            }
        }
        let mut out = vec![0.0; n * qw];
        let scale = (d as f32).sqrt().recip();
        for t in 0..n {
            for h in 0..c.num_attention_heads {
                let kh = h / (c.num_attention_heads / c.num_key_value_heads);
                let query = &q[t * qw + h * d..t * qw + (h + 1) * d];
                let mut scores: Vec<f32> = (0..=t)
                    .map(|s| dot_scores(query, &k[s * kw + kh * d..s * kw + (kh + 1) * d]) * scale)
                    .collect();
                softmax(&mut scores);
                let dst = &mut out[t * qw + h * d..t * qw + (h + 1) * d];
                for (s, p) in scores.iter().enumerate() {
                    for (o, a) in dst
                        .iter_mut()
                        .zip(&v[s * kw + kh * d..s * kw + (kh + 1) * d])
                    {
                        *o += p * a;
                    }
                }
            }
        }
        for (o, g) in out.iter_mut().zip(gate) {
            *o *= sigmoid(g);
        }
        self.o.apply(&out)
    }
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
                let decay = (-self.a_log[head].exp()
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
