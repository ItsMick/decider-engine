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
    /// Prove products used by all model/kernel allocations before reading any
    /// payload. Independent projection widths are legal (not necessarily h).
    pub fn validate_layout(
        &self,
        tokens: usize,
        rank: usize,
        pointer: usize,
        limits: &crate::LoadLimits,
    ) -> Result<(), String> {
        use crate::loading::{product as mul, sum, within};
        self.validate()?;
        let h = self.hidden_size;
        let qw = mul("query width", &[self.num_attention_heads, self.head_dim])?;
        let kw = mul(
            "key/value width",
            &[self.num_key_value_heads, self.head_dim],
        )?;
        let kd = mul(
            "delta key width",
            &[self.linear_num_key_heads, self.linear_key_head_dim],
        )?;
        let vd = mul(
            "delta value width",
            &[self.linear_num_value_heads, self.linear_value_head_dim],
        )?;
        let qkv = sum("delta QKV width", &[mul("delta keys", &[2, kd])?, vd])?;
        let recurrent = mul(
            "recurrent state",
            &[
                self.linear_num_value_heads,
                self.linear_key_head_dim,
                self.linear_value_head_dim,
            ],
        )?;
        let mut weights = crate::loading::WeightBudget::new(*limits);
        weights.charge("embedding", &[self.vocab_size, h])?;
        weights.charge("final norm", &[h])?;
        weights.charge("pointer head", &[2, pointer, h])?;
        weights.charge(
            "pointer norms/biases",
            &[2, sum("pointer norms", &[h, pointer])?],
        )?;
        let mut max_width = h.max(self.intermediate_size).max(rank).max(pointer);
        for kind in &self.layer_types {
            let mut projections = vec![
                (h, self.intermediate_size),
                (h, self.intermediate_size),
                (self.intermediate_size, h),
            ];
            weights.charge("layer norms", &[2, h])?;
            if kind == "full_attention" {
                projections.extend([
                    (h, mul("gated query width", &[2, qw])?),
                    (h, kw),
                    (h, kw),
                    (qw, h),
                ]);
                weights.charge("attention norms", &[2, self.head_dim])?;
            } else {
                projections.extend([
                    (h, qkv),
                    (h, vd),
                    (h, self.linear_num_value_heads),
                    (h, self.linear_num_value_heads),
                    (vd, h),
                ]);
                weights.charge("convolution", &[qkv, self.linear_conv_kernel_dim])?;
                weights.charge(
                    "delta constants",
                    &[sum(
                        "delta constants",
                        &[
                            mul("delta heads", &[2, self.linear_num_value_heads])?,
                            self.linear_value_head_dim,
                        ],
                    )?],
                )?;
            }
            for (input, output) in projections {
                weights.charge("projection", &[output, input])?;
                weights.charge("LoRA A", &[rank, input])?;
                weights.charge("LoRA B", &[output, rank])?;
                max_width = max_width.max(input).max(output);
            }
        }
        // A deliberately conservative bound on live activations, projection
        // columns, LoRA intermediates, quantization buffers and worker scratch.
        // Attention scores are row-local, never an n*n*heads tensor.
        let activations = mul("activation workspace", &[tokens, max_width, 16])?;
        let angles = mul("rotary workspace", &[tokens, self.rotary_dim(), 2])?;
        let elements = sum("forward workspace", &[activations, angles, recurrent])?;
        within(
            "forward workspace",
            mul("forward workspace", &[elements, 4])?,
            limits.max_workspace_bytes,
        )
    }
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
            return Err("key head counts must divide query/value head counts".into());
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
            || !(0.0..=1.0).contains(&self.rope_parameters.partial_rotary_factor)
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
    /// Depthwise convolution weights, tap-major: `conv[tap * width + channel]`.
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
    #[cfg(test)]
    pub forward_calls: std::sync::atomic::AtomicUsize,
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
                    conv: tap_major(
                        &w.take(
                            &format!("{p}.linear_attn.conv1d.weight"),
                            &[2 * kd + vd, 1, c.linear_conv_kernel_dim],
                        )?,
                        c.linear_conv_kernel_dim,
                    ),
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
            #[cfg(test)]
            forward_calls: std::sync::atomic::AtomicUsize::new(0),
            config: c,
            embedding,
            layers,
            norm,
            rotary_divisors,
        })
    }
    pub fn forward(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        Ok(self.run(ids, None, false)?.0)
    }

    /// Run a token prefix and keep what later tokens need from it. Hidden states
    /// of the prefix are checked exactly as in `forward`, then discarded.
    pub fn prefill(&self, ids: &[u32]) -> Result<Prefix, String> {
        Ok(self
            .run(ids, None, true)?
            .1
            .expect("prefill captures layer state"))
    }

    /// Hidden states of `ids` following `prefix`; bit-identical to the matching
    /// rows of `forward` over the concatenated tokens.
    pub fn forward_after(&self, prefix: &Prefix, ids: &[u32]) -> Result<Vec<f32>, String> {
        Ok(self.run(ids, Some(prefix), false)?.0)
    }

    fn run(
        &self,
        ids: &[u32],
        past: Option<&Prefix>,
        capture: bool,
    ) -> Result<(Vec<f32>, Option<Prefix>), String> {
        #[cfg(test)]
        self.forward_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let start = past.map_or(0, |p| p.len);
        if ids.is_empty() || start + ids.len() > 4096 {
            return Err("token sequence must contain 1–4096 tokens".into());
        }
        let c = &self.config;
        let size = crate::loading::product("hidden activations", &[ids.len(), c.hidden_size])?;
        let mut x = Vec::with_capacity(size);
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
        let angles: Vec<_> = (start..start + ids.len())
            .flat_map(|pos| {
                self.rotary_divisors
                    .iter()
                    .map(move |divisor| (pos as f32 / divisor).sin_cos())
            })
            .collect();
        let mut captured = Vec::new();
        let mut n = vec![0.0; x.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            n.copy_from_slice(&x);
            rms(&mut n, &layer.input_norm, c.rms_norm_eps, true);
            let (mixed, cache) = match (&layer.mixer, past.map(|p| &p.layers[i])) {
                (Mixer::Full(m), None) => {
                    let (out, k, v) = m.forward(&n, c, &angles, None);
                    (out, LayerCache::Full { k, v })
                }
                (Mixer::Full(m), Some(LayerCache::Full { k, v })) => {
                    let (out, k, v) = m.forward(&n, c, &angles, Some((k, v)));
                    (out, LayerCache::Full { k, v })
                }
                (Mixer::Delta(m), None) => {
                    let (out, tail, state) = m.forward(&n, c, start, None);
                    (out, LayerCache::Delta { tail, state })
                }
                (Mixer::Delta(m), Some(LayerCache::Delta { tail, state })) => {
                    let (out, tail, state) = m.forward(&n, c, start, Some((tail, state)));
                    (out, LayerCache::Delta { tail, state })
                }
                _ => return Err("prefix cache does not match the layer types".into()),
            };
            if capture {
                captured.push(cache);
            }
            residual_norm(&mut x, &mixed, &mut n, &layer.post_norm, c.rms_norm_eps);
            let mut gate = layer.gate.apply(&n);
            layer.up.apply_into(&n, &mut gate, Epilogue::SwiGlu);
            layer.down.apply_into(&gate, &mut x, Epilogue::Add);
        }
        rms(&mut x, &self.norm, c.rms_norm_eps, true);
        if x.iter().any(|v| !v.is_finite()) {
            return Err("non-finite Qwen hidden states".into());
        }
        let prefix = capture.then(|| Prefix {
            len: start + ids.len(),
            layers: captured,
        });
        Ok((x, prefix))
    }
}

/// What tokens after a prefix need from it: the prefix length and, per layer,
/// full-attention keys and values or DeltaNet convolution inputs and state.
pub(crate) struct Prefix {
    len: usize,
    layers: Vec<LayerCache>,
}

impl Prefix {
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.len
    }
}

enum LayerCache {
    /// Normalized, rotated keys and values of every prefix token.
    Full { k: Vec<f32>, v: Vec<f32> },
    /// The last `kernel - 1` convolution inputs and each head's recurrent state.
    Delta { tail: Vec<f32>, state: Vec<f32> },
}

impl Full {
    /// Returns the layer output and these tokens' normalized, rotated keys and values.
    fn forward(
        &self,
        x: &[f32],
        c: &Config,
        angles: &[(f32, f32)],
        past: Option<(&[f32], &[f32])>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
        let joined = past.map(|(pk, pv)| ([pk, &k].concat(), [pv, &v].concat()));
        let (keys, values) = joined
            .as_ref()
            .map_or((&k[..], &v[..]), |(pk, pv)| (&pk[..], &pv[..]));
        let offset = past.map_or(0, |(pk, _)| pk.len() / kw);
        let mut out = causal_attention(
            &q,
            keys,
            values,
            c.num_attention_heads,
            c.num_key_value_heads,
            d,
            offset,
        );
        for (o, g) in out.iter_mut().zip(gate) {
            *o *= sigmoid(g);
        }
        (self.o.apply(&out), k, v)
    }
}

/// Causal attention for `n` query tokens at positions `offset..offset + n`, over
/// keys and values for positions `0..offset + n`.
fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    key_heads: usize,
    d: usize,
    offset: usize,
) -> Vec<f32> {
    let qw = heads * d;
    let kw = key_heads * d;
    let n = q.len() / qw;
    assert_eq!(q.len(), n * qw);
    assert_eq!(k.len(), (offset + n) * kw);
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
                .extend((0..=offset + t).map(|s| {
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
            || Vec::with_capacity(offset + n),
            |scores, (t, row)| compute(t, row, scores),
        );
    } else {
        let mut scores = Vec::with_capacity(offset + n);
        for (t, row) in out.chunks_exact_mut(qw).enumerate() {
            compute(t, row, &mut scores);
        }
    }
    out
}

impl Delta {
    /// Continue from `past` (convolution inputs of the preceding tokens, at most
    /// `kernel - 1` rows, and the recurrent state), with these tokens starting at
    /// position `start`. Returns the layer output and the state after them.
    fn forward(
        &self,
        x: &[f32],
        c: &Config,
        start: usize,
        past: Option<(&[f32], &[f32])>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
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
        let kernel = c.linear_conv_kernel_dim;
        let tail = past.map_or(&[][..], |(tail, _)| tail);
        let tail_rows = tail.len() / width;
        assert_eq!(tail_rows, start.min(kernel - 1));
        let conv = convolve(&raw, tail, start, width, kernel, &self.conv);
        let mut state = past.map_or_else(
            || vec![0.0; c.linear_num_value_heads * dk * dv],
            |(_, state)| state.to_vec(),
        );
        let out = Recurrence {
            key_heads: c.linear_num_key_heads,
            value_heads: c.linear_num_value_heads,
            dk,
            dv,
            eps: c.rms_norm_eps,
            decay_rate: &self.decay_rate,
            dt: &self.dt,
            norm: &self.norm,
        }
        .run(&conv, &a, &b, &z, &mut state);
        // The last `kernel - 1` convolution inputs across the tail and these tokens.
        let keep = (start + n).min(kernel - 1);
        let mut next_tail = Vec::with_capacity(keep * width);
        let from_tail = keep.saturating_sub(n);
        next_tail.extend_from_slice(&tail[(tail_rows - from_tail) * width..]);
        next_tail.extend_from_slice(&raw[(n - (keep - from_tail)) * width..]);
        (self.out.apply(&out), next_tail, state)
    }
}

/// Checkpoint `[channel][1][tap]` convolution weights as `[tap][channel]`.
fn tap_major(weights: &[f32], kernel: usize) -> Vec<f32> {
    let width = weights.len() / kernel;
    (0..kernel * width)
        .map(|i| weights[(i % width) * kernel + i / width])
        .collect()
}

/// Causal depthwise convolution with SiLU over `raw` (`n` rows of `width`
/// channels) for tokens at positions `start..start + n`; inputs before `start`
/// come from `tail`, the preceding tokens' last rows. `taps` is tap-major.
///
/// Each channel's sum starts at zero and adds `input * weight` tap by tap in
/// order, skipping taps before the sequence start, as a scalar loop would. Rows
/// are independent, so they run in parallel, and channels vectorize.
fn convolve(
    raw: &[f32],
    tail: &[f32],
    start: usize,
    width: usize,
    kernel: usize,
    taps: &[f32],
) -> Vec<f32> {
    let n = raw.len() / width;
    let tail_rows = tail.len() / width;
    assert_eq!(taps.len(), kernel * width);
    let row = |t: usize, out: &mut [f32]| {
        let position = start + t;
        for (j, weights) in taps.chunks_exact(width).enumerate() {
            if position + j + 1 < kernel {
                continue;
            }
            // Inputs before `start` come from the preceding tokens' tail.
            let source = position + j + 1 - kernel;
            let input = if source >= start {
                &raw[(source - start) * width..(source - start + 1) * width]
            } else {
                let row = source + tail_rows - start;
                &tail[row * width..(row + 1) * width]
            };
            for ((sum, x), w) in out.iter_mut().zip(input).zip(weights) {
                *sum += x * w;
            }
        }
        for sum in out.iter_mut() {
            *sum = silu(*sum);
        }
    };
    let mut conv = vec![0.0; raw.len()];
    if n * width >= 1 << 16 && rayon::current_num_threads() > 1 {
        conv.par_chunks_mut(width)
            .enumerate()
            .for_each(|(t, out)| row(t, out));
    } else {
        conv.chunks_mut(width)
            .enumerate()
            .for_each(|(t, out)| row(t, out));
    }
    conv
}

/// Shapes and per-head constants of the Gated DeltaNet recurrence.
struct Recurrence<'a> {
    key_heads: usize,
    value_heads: usize,
    dk: usize,
    dv: usize,
    eps: f32,
    decay_rate: &'a [f32],
    dt: &'a [f32],
    norm: &'a [f32],
}

impl Recurrence<'_> {
    /// Normalized, gated output `[n, value_heads * dv]` from the convolved q/k/v
    /// channels and the `a`, `b` and `z` projections of `n` tokens.
    /// `state` holds each head's `dk x dv` recurrent state; it is read as the
    /// state before these tokens and left as the state after them.
    fn run(&self, conv: &[f32], a: &[f32], b: &[f32], z: &[f32], state: &mut [f32]) -> Vec<f32> {
        let (dk, dv, heads) = (self.dk, self.dv, self.value_heads);
        let vd = heads * dv;
        let n = conv.len() / (2 * self.key_heads * dk + vd);
        assert_eq!(state.len(), heads * dk * dv);
        if n == 0 {
            return Vec::new();
        }
        let mut by_head = vec![0.0; heads * n * dv];
        let run = |(head, (dst, state)): (usize, (&mut [f32], &mut [f32]))| {
            self.head(head, conv, a, b, z, dst, state)
        };
        // Heads own disjoint state and output rows, so running them in parallel
        // changes no arithmetic. Short sequences stay on the calling thread.
        if heads > 1 && n * dk * dv >= 1 << 16 && rayon::current_num_threads() > 1 {
            by_head
                .par_chunks_mut(n * dv)
                .zip(state.par_chunks_mut(dk * dv))
                .enumerate()
                .for_each(run);
        } else {
            by_head
                .chunks_mut(n * dv)
                .zip(state.chunks_mut(dk * dv))
                .enumerate()
                .for_each(run);
        }
        let mut out = vec![0.0; n * vd];
        for (head, rows) in by_head.chunks_exact(n * dv).enumerate() {
            for (t, row) in rows.chunks_exact(dv).enumerate() {
                out[t * vd + head * dv..t * vd + (head + 1) * dv].copy_from_slice(row);
            }
        }
        out
    }

    /// One value head over every token, writing `[n, dv]` rows to `dst`.
    #[allow(clippy::too_many_arguments)]
    fn head(
        &self,
        head: usize,
        conv: &[f32],
        a: &[f32],
        b: &[f32],
        z: &[f32],
        dst: &mut [f32],
        state: &mut [f32],
    ) {
        let (dk, dv, heads) = (self.dk, self.dv, self.value_heads);
        let kd = self.key_heads * dk;
        let vd = heads * dv;
        let width = 2 * kd + vd;
        let kh = head / (heads / self.key_heads);
        let mut q = vec![0.0; dk];
        let mut k = vec![0.0; dk];
        let mut delta = vec![0.0; dv];
        for (t, dst) in dst.chunks_exact_mut(dv).enumerate() {
            q.copy_from_slice(&conv[t * width + kh * dk..t * width + (kh + 1) * dk]);
            k.copy_from_slice(&conv[t * width + kd + kh * dk..t * width + kd + (kh + 1) * dk]);
            let qnorm =
                (q.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip() / (dk as f32).sqrt();
            let knorm = (k.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt().recip();
            for v in &mut q {
                *v *= qnorm;
            }
            for v in &mut k {
                *v *= knorm;
            }
            let decay =
                (self.decay_rate[head] * softplus(a[t * heads + head] + self.dt[head])).exp();
            let beta = sigmoid(b[t * heads + head]);
            for v in state.iter_mut() {
                *v *= decay;
            }
            delta.fill(0.0);
            for (row, ki) in state.chunks_exact(dv).zip(&k) {
                for (m, s) in delta.iter_mut().zip(row) {
                    *m += ki * s;
                }
            }
            for (m, v) in delta
                .iter_mut()
                .zip(&conv[t * width + 2 * kd + head * dv..t * width + 2 * kd + (head + 1) * dv])
            {
                *m = (v - *m) * beta;
            }
            for ((row, ki), qi) in state.chunks_exact_mut(dv).zip(&k).zip(&q) {
                for ((s, de), o) in row.iter_mut().zip(&delta).zip(dst.iter_mut()) {
                    *s += ki * de;
                    *o += qi * (*s);
                }
            }
            rms(dst, self.norm, self.eps, false);
            for (o, g) in dst
                .iter_mut()
                .zip(&z[t * vd + head * dv..t * vd + (head + 1) * dv])
            {
                *o *= silu(*g);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/tiny/base/config.json")).unwrap();
        serde_json::from_value(v["text_config"].clone()).unwrap()
    }

    #[test]
    fn invalid_derived_shapes_fail_before_allocating() {
        let limits = crate::LoadLimits::default();
        let mut c = config();
        c.hidden_size = 1_000_000;
        c.intermediate_size = 1_000_000;
        assert!(c
            .validate_layout(4096, 16, 4, &limits)
            .unwrap_err()
            .contains("budget"));
        let mut c = config();
        c.linear_num_value_heads = 1_000_000;
        c.linear_key_head_dim = 1_000_000;
        c.linear_value_head_dim = 1_000_000;
        assert!(c.validate_layout(4096, 16, 4, &limits).is_err());
        let mut c = config();
        c.num_attention_heads = 3;
        c.num_key_value_heads = 2;
        assert!(c.validate().unwrap_err().contains("head counts"));
        let mut c = config();
        c.linear_num_value_heads = 3;
        assert!(c.validate().unwrap_err().contains("head counts"));
        let mut c = config();
        c.num_hidden_layers = 1_000_000;
        c.layer_types = vec!["full_attention".into(); c.num_hidden_layers];
        assert!(c
            .validate_layout(4096, 4096, 4, &limits)
            .unwrap_err()
            .contains("budget"));
    }

    /// Per-operator timing of one DeltaNet layer, one full-attention layer and
    /// their MLPs at the Qwen3.5-2B dimensions, with synthetic BF16 weights and
    /// rank-16 F32 adapters. Prints one line per operator.
    #[test]
    #[ignore = "real-size operator profile; run with --release and a fixed RAYON_NUM_THREADS"]
    fn profile_layer_operators() {
        use crate::Precision;
        use std::{hint::black_box, time::Instant};
        let (h, inter, rank) = (2048, 6144, 16);
        let (heads, kv_heads, head_dim) = (8, 2, 256);
        let (lk, lv, dk, dv, kernel) = (16, 16, 128, 128, 4);
        let (kd, vd) = (lk * dk, lv * dv);
        let tokens: Vec<usize> = std::env::var("PROFILE_TOKENS")
            .map(|v| v.split(',').map(|n| n.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![64, 512, 2048]);
        let modes = [Precision::Checkpoint, Precision::Int8Compute];
        println!(
            "profile arch={} workers={}",
            std::env::consts::ARCH,
            rayon::current_num_threads()
        );
        let time = |label: &str, precision: Precision, n: usize, f: &mut dyn FnMut()| {
            f();
            let start = Instant::now();
            for _ in 0..2 {
                f();
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / 2.0;
            println!("profile precision={precision:?} tokens={n} op={label} ms={ms:.2}");
        };
        for precision in modes {
            let lin = |i, o, seed| Linear::synthetic(i, o, rank, seed, precision);
            let delta = Delta {
                qkv: lin(h, 2 * kd + vd, 1),
                z: lin(h, vd, 2),
                a: lin(h, lv, 3),
                b: lin(h, lv, 4),
                out: lin(vd, h, 5),
                conv: (0..(2 * kd + vd) * kernel)
                    .map(|i| (i % 7) as f32 / 9.0 - 0.3)
                    .collect(),
                dt: vec![0.1; lv],
                decay_rate: vec![-0.5; lv],
                norm: vec![1.0; dv],
            };
            let full = Full {
                q: lin(h, heads * head_dim * 2, 6),
                k: lin(h, kv_heads * head_dim, 7),
                v: lin(h, kv_heads * head_dim, 8),
                o: lin(heads * head_dim, h, 9),
                q_norm: vec![0.1; head_dim],
                k_norm: vec![0.1; head_dim],
            };
            let (gate, up, down) = (lin(h, inter, 10), lin(h, inter, 11), lin(inter, h, 12));
            for &n in &tokens {
                let x: Vec<f32> = (0..n * h)
                    .map(|i| ((i * 43 % 113) as f32 - 56.0) / 64.0)
                    .collect();
                let raw = delta.qkv.apply(&x);
                let (a, b, z) = (delta.a.apply(&x), delta.b.apply(&x), delta.z.apply(&x));
                let conv = convolve(&raw, &[], 0, 2 * kd + vd, kernel, &delta.conv);
                let recurrence = Recurrence {
                    key_heads: lk,
                    value_heads: lv,
                    dk,
                    dv,
                    eps: 1e-6,
                    decay_rate: &delta.decay_rate,
                    dt: &delta.dt,
                    norm: &delta.norm,
                };
                let mixed: Vec<f32> = (0..n * vd).map(|i| (i % 5) as f32 / 5.0).collect();
                time("delta.in_proj_qkv", precision, n, &mut || {
                    black_box(delta.qkv.apply(&x));
                });
                time("delta.in_proj_z", precision, n, &mut || {
                    black_box(delta.z.apply(&x));
                });
                time("delta.in_proj_a+b", precision, n, &mut || {
                    black_box((delta.a.apply(&x), delta.b.apply(&x)));
                });
                time("delta.conv1d", precision, n, &mut || {
                    black_box(convolve(&raw, &[], 0, 2 * kd + vd, kernel, &delta.conv));
                });
                time("delta.recurrence", precision, n, &mut || {
                    let mut state = vec![0.0; lv * dk * dv];
                    black_box(recurrence.run(&conv, &a, &b, &z, &mut state));
                });
                time("delta.out_proj", precision, n, &mut || {
                    black_box(delta.out.apply(&mixed));
                });
                let c = Config {
                    hidden_size: h,
                    intermediate_size: inter,
                    num_hidden_layers: 1,
                    vocab_size: 1,
                    layer_types: vec!["full_attention".into()],
                    num_attention_heads: heads,
                    num_key_value_heads: kv_heads,
                    head_dim,
                    rms_norm_eps: 1e-6,
                    linear_conv_kernel_dim: kernel,
                    linear_key_head_dim: dk,
                    linear_value_head_dim: dv,
                    linear_num_key_heads: lk,
                    linear_num_value_heads: lv,
                    rope_parameters: Rope {
                        rope_theta: 1e7,
                        partial_rotary_factor: 0.25,
                        rope_type: "default".into(),
                    },
                    attention_bias: false,
                    attn_output_gate: true,
                    hidden_act: "silu".into(),
                };
                time("delta.layer_mixer_total", precision, n, &mut || {
                    black_box(delta.forward(&x, &c, 0, None));
                });
                let angles: Vec<(f32, f32)> =
                    (0..n * 32).map(|i| (i as f32 / 1000.0).sin_cos()).collect();
                let (q, k, v) = (
                    (0..n * heads * head_dim)
                        .map(|i| (i % 11) as f32 / 11.0)
                        .collect::<Vec<f32>>(),
                    (0..n * kv_heads * head_dim)
                        .map(|i| (i % 13) as f32 / 13.0)
                        .collect::<Vec<f32>>(),
                    (0..n * kv_heads * head_dim)
                        .map(|i| (i % 7) as f32 / 7.0)
                        .collect::<Vec<f32>>(),
                );
                let attended: Vec<f32> = (0..n * heads * head_dim)
                    .map(|i| (i % 3) as f32 / 3.0)
                    .collect();
                time("full.q_proj", precision, n, &mut || {
                    black_box(full.q.apply(&x));
                });
                time("full.k+v_proj", precision, n, &mut || {
                    black_box((full.k.apply(&x), full.v.apply(&x)));
                });
                time("full.attention_core", precision, n, &mut || {
                    black_box(causal_attention(&q, &k, &v, heads, kv_heads, head_dim, 0));
                });
                time("full.o_proj", precision, n, &mut || {
                    black_box(full.o.apply(&attended));
                });
                time("full.layer_mixer_total", precision, n, &mut || {
                    black_box(full.forward(&x, &c, &angles, None));
                });
                time("mlp.gate+up+down", precision, n, &mut || {
                    let mut g = gate.apply(&x);
                    up.apply_into(&x, &mut g, Epilogue::SwiGlu);
                    let mut residual = x.clone();
                    down.apply_into(&g, &mut residual, Epilogue::Add);
                    black_box(residual);
                });
                time("norms(2x rms)", precision, n, &mut || {
                    let mut t = x.clone();
                    rms(&mut t, &vec![0.1; h], 1e-6, true);
                    rms(&mut t, &vec![0.1; h], 1e-6, true);
                    black_box(t);
                });
            }
        }
    }

    /// The original channel-by-channel convolution over checkpoint-layout weights.
    fn reference_convolve(
        raw: &[f32],
        tail: &[f32],
        start: usize,
        width: usize,
        kernel: usize,
        weights: &[f32],
    ) -> Vec<f32> {
        let n = raw.len() / width;
        let tail_rows = tail.len() / width;
        let mut conv = vec![0.0; raw.len()];
        for t in 0..n {
            let position = start + t;
            for channel in 0..width {
                let mut sum = 0.0;
                for j in 0..kernel {
                    if position + j + 1 >= kernel {
                        let source = position + j + 1 - kernel;
                        let value = if source >= start {
                            raw[(source - start) * width + channel]
                        } else {
                            tail[(source + tail_rows - start) * width + channel]
                        };
                        sum += value * weights[channel * kernel + j];
                    }
                }
                conv[t * width + channel] = silu(sum);
            }
        }
        conv
    }

    #[test]
    fn vectorized_convolution_matches_channel_loop_exactly() {
        let pools: Vec<_> = [1, 4]
            .into_iter()
            .map(|t| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
            })
            .collect();
        let value = |i: usize, m: usize| match i % 13 {
            0 => -0.0,
            1 => 1e-30,
            k => ((i * m % 997) as f32 - 498.0) / (k as f32 * 9.0),
        };
        // Widths on both sides of the parallel threshold; sequences shorter than
        // the kernel; continuations after 0, 1, 2 and 3 tail rows.
        for (width, kernel) in [(5, 4), (64, 4), (6144, 4), (7, 1), (9, 3)] {
            let weights: Vec<f32> = (0..width * kernel).map(|i| value(i, 31)).collect();
            let taps = tap_major(&weights, kernel);
            for (start, n) in [
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 17),
                (1, 4),
                (2, 1),
                (3, 5),
                (9, 12),
            ] {
                let tail_rows = start.min(kernel - 1);
                let tail: Vec<f32> = (0..tail_rows * width).map(|i| value(i, 17)).collect();
                let raw: Vec<f32> = (0..n * width).map(|i| value(i, 43)).collect();
                let expected = reference_convolve(&raw, &tail, start, width, kernel, &weights);
                for pool in &pools {
                    let actual =
                        pool.install(|| convolve(&raw, &tail, start, width, kernel, &taps));
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "width {width}, kernel {kernel}, start {start}, n {n}"
                    );
                }
            }
        }
    }

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
                    black_box(causal_attention(&q, &k, &v, heads, key_heads, d, 0));
                });
                let start = Instant::now();
                for _ in 0..3 {
                    pool.install(|| {
                        black_box(causal_attention(
                            black_box(&q),
                            &k,
                            &v,
                            heads,
                            key_heads,
                            d,
                            0,
                        ));
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

    /// The original token-major recurrence, kept as the bit-exact reference.
    fn reference_recurrence(
        r: &Recurrence,
        conv: &[f32],
        a: &[f32],
        b: &[f32],
        z: &[f32],
    ) -> Vec<f32> {
        let (dk, dv, heads) = (r.dk, r.dv, r.value_heads);
        let kd = r.key_heads * dk;
        let vd = heads * dv;
        let width = 2 * kd + vd;
        let n = conv.len() / width;
        let mut out = vec![0.0; n * vd];
        let mut state = vec![0.0; heads * dk * dv];
        let mut q = vec![0.0; dk];
        let mut k = vec![0.0; dk];
        let mut delta = vec![0.0; dv];
        for t in 0..n {
            for head in 0..heads {
                let kh = head / (heads / r.key_heads);
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
                let decay = (r.decay_rate[head] * softplus(a[t * heads + head] + r.dt[head])).exp();
                let beta = sigmoid(b[t * heads + head]);
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
                rms(dst, r.norm, r.eps, false);
                for (o, g) in dst
                    .iter_mut()
                    .zip(&z[t * vd + head * dv..t * vd + (head + 1) * dv])
                {
                    *o *= silu(*g);
                }
            }
        }
        out
    }

    #[test]
    #[ignore = "synthetic recurrence timing; run with --release"]
    fn benchmark_deltanet_recurrence() {
        use std::{hint::black_box, time::Instant};
        // Qwen3.5-2B linear-attention shape: 16 key and value heads of 128.
        let (key_heads, value_heads, dk, dv) = (16, 16, 128, 128);
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let pools: Vec<_> = [1, threads]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        let decay_rate = vec![-0.5; value_heads];
        let dt = vec![0.1; value_heads];
        let norm = vec![1.0; dv];
        let recurrence = Recurrence {
            key_heads,
            value_heads,
            dk,
            dv,
            eps: 1e-6,
            decay_rate: &decay_rate,
            dt: &dt,
            norm: &norm,
        };
        for n in [32, 128, 512] {
            let width = 2 * key_heads * dk + value_heads * dv;
            let conv: Vec<f32> = (0..n * width)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 23.0)
                .collect();
            let gates: Vec<f32> = (0..n * value_heads).map(|i| (i % 7) as f32 / 7.0).collect();
            let z: Vec<f32> = (0..n * value_heads * dv)
                .map(|i| (i % 5) as f32 / 5.0)
                .collect();
            for pool in &pools {
                let mut state = vec![0.0; value_heads * dk * dv];
                pool.install(|| black_box(recurrence.run(&conv, &gates, &gates, &z, &mut state)));
                let start = Instant::now();
                for _ in 0..3 {
                    state.fill(0.0);
                    pool.install(|| {
                        black_box(recurrence.run(black_box(&conv), &gates, &gates, &z, &mut state))
                    });
                }
                println!(
                    "deltanet tokens={n} workers={} mean_ms={:.3}",
                    pool.current_num_threads(),
                    start.elapsed().as_secs_f64() * 1000.0 / 3.0
                );
            }
        }
    }

    #[test]
    fn parallel_deltanet_heads_match_the_token_major_recurrence_exactly() {
        let pools: Vec<_> = [1, 4]
            .into_iter()
            .map(|threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
            })
            .collect();
        // Small shapes take the serial path; 32x32 heads over 80 tokens cross the
        // parallel threshold. Includes grouped value heads per key head.
        for (key_heads, value_heads, dk, dv, n) in [
            (1, 1, 4, 3, 5),
            (2, 4, 8, 6, 17),
            (2, 4, 32, 32, 80),
            (4, 4, 32, 32, 64),
        ] {
            let width = 2 * key_heads * dk + value_heads * dv;
            let wave = |len: usize, a: usize, b: usize, scale: f32| -> Vec<f32> {
                (0..len)
                    .map(|i| ((i * a % b) as f32 - (b / 2) as f32) / scale)
                    .collect()
            };
            let conv = wave(n * width, 37, 101, 23.0);
            let a = wave(n * value_heads, 11, 17, 5.0);
            let b = wave(n * value_heads, 13, 19, 4.0);
            let z = wave(n * value_heads * dv, 7, 29, 9.0);
            let decay_rate: Vec<f32> = (0..value_heads).map(|h| -0.5 - h as f32 / 7.0).collect();
            let dt: Vec<f32> = (0..value_heads).map(|h| h as f32 / 11.0 - 0.2).collect();
            let norm: Vec<f32> = (0..dv).map(|i| 0.75 + i as f32 / 64.0).collect();
            let recurrence = Recurrence {
                key_heads,
                value_heads,
                dk,
                dv,
                eps: 1e-6,
                decay_rate: &decay_rate,
                dt: &dt,
                norm: &norm,
            };
            let expected = reference_recurrence(&recurrence, &conv, &a, &b, &z);
            for pool in &pools {
                let actual = pool.install(|| {
                    let mut state = vec![0.0; value_heads * dk * dv];
                    recurrence.run(&conv, &a, &b, &z, &mut state)
                });
                assert_eq!(
                    actual, expected,
                    "{key_heads}/{value_heads} heads, {dk}x{dv}, {n} tokens"
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
            let expected = serial.install(|| causal_attention(&q, &k, &v, heads, key_heads, d, 0));
            let actual = parallel.install(|| causal_attention(&q, &k, &v, heads, key_heads, d, 0));
            assert_eq!(actual, expected);
            k[(n - 3) * key_heads * d..].fill(7.0);
            v[(n - 3) * key_heads * d..].fill(-11.0);
            let changed = parallel.install(|| causal_attention(&q, &k, &v, heads, key_heads, d, 0));
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
