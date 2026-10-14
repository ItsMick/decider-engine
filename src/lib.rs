//! A native, text-only Strands Decider runtime. Rust owns the Qwen3.5 forward
//! pass, LoRA projections, pointer head, prompt binding and calibration.
//! CPU kernels and model inference are implemented in this crate.
//! Loading never downloads or executes Python. Requests over the trained window
//! are rejected in full rather than silently losing state or options.
#![deny(unsafe_code)]
#![doc = include_str!("../README.md")]

mod error;
mod int8;
mod limits;
mod loading;
mod prompt;
mod qwen;
pub mod schema;
mod tensor;
mod weights;

use crate::tensor::{dot_scores, softmax};
pub use error::DecideError;
pub use limits::InferenceLimits;
pub use loading::LoadLimits;
pub use schema::{Answer, Question, Request, Response};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};
use weights::{Linear, Weights};

pub const ENGINE_ID: &str = "strands-decider-2b-native";
/// Lower precision modes quantize base linear weights only and need quality evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Checkpoint,
    Int8,
    /// Experimental INT8 weights and dynamically quantized projection inputs.
    Int8Compute,
    Int4,
}
impl Precision {
    fn engine_id(self) -> &'static str {
        match self {
            Self::Checkpoint => ENGINE_ID,
            Self::Int8 => "strands-decider-2b-native-int8",
            Self::Int8Compute => "strands-decider-2b-native-int8-compute",
            Self::Int4 => "strands-decider-2b-native-int4",
        }
    }
}
pub trait DecisionEngine {
    fn model(&self) -> &str;
    fn decide(&self, request: &Request) -> Result<Response, String>;
}

#[derive(Deserialize)]
struct DeciderConfig {
    base_model: String,
    head_type: String,
    pointer_dim: usize,
    max_length: usize,
    temperature: f32,
    #[serde(default)]
    temperature_by_kind: BTreeMap<String, f32>,
    #[serde(default)]
    ordinal_smoothing: f32,
    use_lora: bool,
    #[serde(default)]
    force_bos: bool,
}
struct Head {
    norm: Vec<f32>,
    bias: Vec<f32>,
    q: Linear,
    k: Linear,
}
impl Head {
    fn load(mut w: Weights, hidden: usize, dim: usize) -> Result<Self, String> {
        let head = Self {
            norm: w.take("norm.weight", &[hidden])?,
            bias: w.take("norm.bias", &[hidden])?,
            q: Linear::load(&mut w, "q", hidden, dim, true)?,
            k: Linear::load(&mut w, "k", hidden, dim, true)?,
        };
        if w.remaining() != 0 {
            return Err("unsupported pointer head tensors".into());
        }
        Ok(head)
    }
    fn normalize(&self, x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let inv = (variance + 1e-5).sqrt().recip();
        x.iter()
            .zip(&self.norm)
            .zip(&self.bias)
            .map(|((v, w), b)| (v - mean) * inv * w + b)
            .collect()
    }
    fn logits(&self, hidden: &[f32], positions: &[usize]) -> Vec<f32> {
        let size = self.norm.len();
        let query = self
            .q
            .apply(&self.normalize(&hidden[hidden.len() - size..]));
        let scale = (self.q.output as f32).sqrt().recip();
        positions
            .iter()
            .map(|i| {
                let key = self
                    .k
                    .apply(&self.normalize(&hidden[i * size..(i + 1) * size]));
                dot_scores(&query, &key) * scale
            })
            .collect()
    }
}

pub struct NativeEngine {
    model: qwen::Model,
    tokenizer: tokenizers::Tokenizer,
    head: Head,
    config: DeciderConfig,
    precision: Precision,
    checkpoint_revision: Option<CheckpointRevision>,
}
/// Revisions declared by the checkpoint's optional revisions.json. This metadata
/// is not a weight-integrity attestation; verify files with the pinned downloader.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct CheckpointRevision {
    pub decider: String,
    pub base: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub option_positions: Vec<usize>,
    pub answer_position: usize,
}

fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit {
        return Err(format!(
            "{}: metadata exceeds {limit} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}
fn json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&read_bounded(path, 1024 * 1024)?)
        .map_err(|e| format!("{}: {e}", path.display()))
}
impl NativeEngine {
    /// Apply an explicit per-call serving policy. Every question is preflighted
    /// before any model execution; errors never return partial answers.
    pub fn decide_with_limits(
        &self,
        request: &Request,
        limits: &InferenceLimits,
    ) -> Result<Response, String> {
        self.decide_with_limits_checked(request, limits)
            .map_err(|e| e.to_string())
    }
    /// Typed serving API for hosts that distinguish client rejection from faults.
    pub fn decide_checked(&self, request: &Request) -> Result<Response, DecideError> {
        self.decide_with_limits_checked(request, &InferenceLimits::default())
    }

    pub fn decide_with_limits_checked(
        &self,
        request: &Request,
        limits: &InferenceLimits,
    ) -> Result<Response, DecideError> {
        self.decide_bounded(request, limits)
    }

    pub fn checkpoint_revision(&self) -> Option<&CheckpointRevision> {
        self.checkpoint_revision.as_ref()
    }

    /// `dir` holds a Strands export and a `base/` directory with Qwen weights.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_with_precision(dir, Precision::Checkpoint)
    }
    /// Groupwise symmetric modes, not GGUF/AWQ formats. `Int8Compute` also
    /// quantizes projection inputs and uses ARM dot products when available.
    pub fn load_with_precision(dir: &Path, precision: Precision) -> Result<Self, String> {
        Self::load_with_limits(dir, precision, &LoadLimits::default())
    }
    /// Load with explicit allocation budgets. This does not sandbox checkpoints.
    pub fn load_with_limits(
        dir: &Path,
        precision: Precision,
        limits: &LoadLimits,
    ) -> Result<Self, String> {
        let revision_path = dir.join("revisions.json");
        let checkpoint_revision = if revision_path.try_exists().map_err(|e| e.to_string())? {
            let revision: CheckpointRevision = serde_json::from_value(json(&revision_path)?)
                .map_err(|e| format!("invalid checkpoint revisions: {e}"))?;
            if [&revision.decider, &revision.base]
                .iter()
                .any(|r| r.len() != 40 || !r.bytes().all(|b| b.is_ascii_hexdigit()))
            {
                return Err("checkpoint revisions must be 40-character Git SHA values".into());
            }
            Some(revision)
        } else {
            None
        };
        let config_path = if dir.join("strands_decider_config.json").exists() {
            dir.join("strands_decider_config.json")
        } else {
            dir.join("hobson_config.json")
        };
        let config: DeciderConfig =
            serde_json::from_value(json(&config_path)?).map_err(|e| e.to_string())?;
        if config.base_model != "Qwen/Qwen3.5-2B-Base"
            || config.head_type != "pointer"
            || !config.use_lora
            || config.force_bos
        {
            return Err("native engine requires Qwen3.5-2B-Base, pointer head, LoRA, and default BOS behavior".into());
        }
        if config.max_length == 0
            || config.max_length > 4096
            || config.pointer_dim == 0
            || config.pointer_dim > 4096
            || !config.temperature.is_finite()
            || config.temperature <= 0.0
            || config
                .temperature_by_kind
                .values()
                .any(|t| !t.is_finite() || *t <= 0.0)
            || !config.ordinal_smoothing.is_finite()
            || !(0.0..=1.0).contains(&config.ordinal_smoothing)
        {
            return Err("invalid decision window, pointer dimension, or calibration".into());
        }
        let base = json(&dir.join("base/config.json"))?;
        if base["model_type"] != "qwen3_5" {
            return Err("base model must be qwen3_5".into());
        }
        let cfg: qwen::Config =
            serde_json::from_value(base["text_config"].clone()).map_err(|e| e.to_string())?;
        cfg.validate()?;
        let a = json(&dir.join("lora/adapter_config.json"))?;
        let rank = a["r"]
            .as_u64()
            .filter(|r| *r > 0 && *r <= 4096)
            .ok_or("invalid LoRA rank")? as usize;
        let alpha = a["lora_alpha"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or("invalid LoRA alpha")? as f32;
        if !alpha.is_finite() || alpha / rank as f32 == 0.0 {
            return Err("LoRA alpha is not representable as positive finite FP32".into());
        }
        cfg.validate_layout(config.max_length, rank, config.pointer_dim, limits)?;
        if a["peft_type"] != "LORA"
            || a["bias"] != "none"
            || a["use_dora"].as_bool().unwrap_or(false)
            || a["use_rslora"].as_bool().unwrap_or(false)
            || a["fan_in_fan_out"].as_bool().unwrap_or(false)
            || a["rank_pattern"].as_object().is_some_and(|v| !v.is_empty())
            || a["alpha_pattern"]
                .as_object()
                .is_some_and(|v| !v.is_empty())
            || a["modules_to_save"]
                .as_array()
                .is_some_and(|v| !v.is_empty())
        {
            return Err("unsupported LoRA adapter configuration".into());
        }
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(read_bounded(
            &dir.join("tokenizer.json"),
            64 * 1024 * 1024,
        )?)
        .map_err(|e| format!("tokenizer: {e}"))?;
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        let mut budget = loading::WeightBudget::new(*limits);
        let mut adapter = Weights::file_bounded(
            &dir.join("lora/adapter_model.safetensors"),
            None,
            &mut budget,
        )?;
        if adapter.remaining() == 0 {
            return Err("empty LoRA adapter".into());
        }
        let head = Head::load(
            Weights::file_bounded(&dir.join("head.safetensors"), None, &mut budget)?,
            cfg.hidden_size,
            config.pointer_dim,
        )?;
        let model = qwen::Model::load(
            Weights::base(&dir.join("base"), precision, &mut budget)?,
            &mut adapter,
            cfg,
            rank,
            alpha / rank as f32,
        )?;
        Ok(Self {
            model,
            tokenizer,
            head,
            config,
            precision,
            checkpoint_revision,
        })
    }
    pub fn max_length(&self) -> usize {
        self.config.max_length
    }
    /// Useful for reference parity checks; offsets are computed from the exact BPE encoding.
    pub fn encode(&self, state: &Value, question: &Question) -> Result<Encoding, String> {
        schema::content(state, false)?;
        question.validate()?;
        let rendered = prompt::render(question);
        encode(&self.tokenizer, state, &rendered, self.config.max_length).map_err(|e| e.to_string())
    }
    /// Run an already bound prompt through the native torso and pointer head.
    pub fn raw_logits(&self, encoded: &Encoding) -> Result<Vec<f32>, String> {
        if encoded.input_ids.is_empty()
            || encoded.input_ids.len() > self.max_length()
            || encoded.answer_position != encoded.input_ids.len() - 1
            || encoded.option_positions.is_empty()
            || encoded
                .option_positions
                .iter()
                .any(|p| *p >= encoded.answer_position)
        {
            return Err("invalid encoding positions or window".into());
        }
        let hidden = self.model.forward(&encoded.input_ids)?;
        Ok(self.head.logits(&hidden, &encoded.option_positions))
    }
}

fn encode(
    tokenizer: &tokenizers::Tokenizer,
    state: &Value,
    rendered: &prompt::Rendered,
    max_length: usize,
) -> Result<Encoding, DecideError> {
    // Strands tokenizes the prefix and question independently, then concatenates.
    let prefix = format!("<state>\n{}\n</state>\n", prompt::render_content(state));
    let s = tokenizer
        .encode(prefix, true)
        .map_err(|e| DecideError::Engine(e.to_string()))?;
    let q = tokenizer
        .encode(rendered.text.as_str(), false)
        .map_err(|e| DecideError::Engine(e.to_string()))?;
    let total = s.len() + q.len();
    if total > max_length {
        return Err(DecideError::ContextWindow {
            tokens: total,
            limit: max_length,
        });
    }
    let mut positions = Vec::new();
    for (start, end) in &rendered.spans {
        let index = q
            .get_offsets()
            .iter()
            .enumerate()
            .filter(|(_, (lo, hi))| hi > lo && lo >= start && hi <= end)
            .map(|(i, _)| i)
            .next_back()
            .ok_or_else(|| {
                DecideError::InvalidRequest("option has no tokens in its span".into())
            })?;
        positions.push(s.len() + index);
    }
    let ids = s.get_ids().iter().chain(q.get_ids()).copied().collect();
    Ok(Encoding {
        input_ids: ids,
        option_positions: positions,
        answer_position: total - 1,
    })
}
impl DecisionEngine for NativeEngine {
    fn model(&self) -> &str {
        self.precision.engine_id()
    }
    fn decide(&self, request: &Request) -> Result<Response, String> {
        self.decide_checked(request).map_err(|e| e.to_string())
    }
}

impl NativeEngine {
    fn decide_bounded(
        &self,
        request: &Request,
        limits: &InferenceLimits,
    ) -> Result<Response, DecideError> {
        limits.validate_checked(request)?;
        let pinned_v21 = self.checkpoint_revision.as_ref().is_some_and(|r| {
            r.decider == "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"
                && r.base == "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
        });
        if request.model.as_deref().is_some_and(|m| {
            m != self.model()
                && m != "strands-decider-latest"
                && !(m == "strands-decider-2B-hobson-v21" && pinned_v21)
        }) {
            return Err(DecideError::InvalidRequest(
                "requested model or revision is not this native engine".into(),
            ));
        }
        // Validate and bind every question before spending CPU on any of them.
        let mut total_tokens = 0;
        let encoded: Vec<_> = request
            .questions
            .iter()
            .map(|(name, q)| {
                let rendered = prompt::render(q);
                let enc = encode(
                    &self.tokenizer,
                    &request.state,
                    &rendered,
                    self.max_length(),
                )?;
                total_tokens = limits
                    .add_tokens(total_tokens, enc.input_ids.len())
                    .map_err(DecideError::Limit)?;
                Ok((name, q, rendered, enc))
            })
            .collect::<Result<_, DecideError>>()?;
        let mut input_tokens = 0;
        let mut answers = BTreeMap::new();
        for (name, q, rendered, enc) in encoded {
            let mut logits = self.raw_logits(&enc).map_err(DecideError::Engine)?;
            let temperature = self
                .config
                .temperature_by_kind
                .get(q.kind())
                .copied()
                .unwrap_or(self.config.temperature);
            for v in &mut logits {
                *v /= temperature;
            }
            softmax(&mut logits);
            if logits.iter().any(|p| !p.is_finite()) {
                return Err(DecideError::Engine(
                    "non-finite decision probabilities".into(),
                ));
            }
            answers.insert(
                name.clone(),
                rendered.answer(q, &logits, self.config.ordinal_smoothing),
            );
            input_tokens += enc.input_ids.len();
        }
        Ok(Response {
            model: self.model().into(),
            answers,
            usage: schema::Usage {
                input_tokens,
                output_tokens: 0,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn late_or_aggregate_rejection_never_starts_a_forward_pass() {
        let e = NativeEngine::load(&fixture()).unwrap();
        let mut request: Request =
            serde_json::from_str(include_str!("../tests/request.json")).unwrap();
        let total = e
            .encode(&request.state, request.questions.values().next().unwrap())
            .unwrap()
            .input_ids
            .len();
        let limits = InferenceLimits {
            max_total_tokens: total,
            ..InferenceLimits::default()
        };
        assert!(e
            .decide_with_limits(&request, &limits)
            .unwrap_err()
            .contains("total tokens"));
        request.questions.insert(
            "zz_last".into(),
            Question::Noul {
                instructions: Value::String("too long ".repeat(300)),
                criteria: None,
            },
        );
        assert!(e.decide(&request).unwrap_err().contains("context window"));
        assert_eq!(
            e.model
                .forward_calls
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }
    fn fixture() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny")
    }
    fn close(a: &Value, b: &Value, tolerance: f64) {
        match (a, b) {
            (Value::Number(x), Value::Number(y)) => assert!(
                (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < tolerance,
                "{x} != {y}"
            ),
            (Value::Array(x), Value::Array(y)) => {
                assert_eq!(x.len(), y.len());
                for (u, v) in x.iter().zip(y) {
                    close(u, v, tolerance);
                }
            }
            (Value::Object(x), Value::Object(y)) => {
                assert_eq!(x.len(), y.len());
                for (k, v) in x {
                    close(v, &y[k], tolerance);
                }
            }
            _ => assert_eq!(a, b),
        }
    }
    #[test]
    fn native_torso_and_pointer_match_independent_pytorch_fixture() {
        let e = NativeEngine::load(&fixture()).unwrap();
        let cases = json(&fixture().join("raw-reference.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let enc: Encoding = serde_json::from_value(case["encoding"].clone()).unwrap();
            let logits = e.raw_logits(&enc).unwrap();
            close(
                &serde_json::to_value(logits).unwrap(),
                &case["logits"],
                1e-5,
            );
            let hidden = e.model.forward(&enc.input_ids).unwrap();
            let expected: Vec<f32> = case["hidden"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap() as f32)
                })
                .collect();
            for (a, b) in hidden.iter().zip(&expected) {
                assert!((a - b).abs() < 2e-5, "hidden: {a} != {b}");
            }
        }
    }
    #[test]
    fn all_three_typed_decisions_and_positions_match_upstream() {
        let e = NativeEngine::load(&fixture()).unwrap();
        let request: Request = serde_json::from_str(include_str!("../tests/request.json")).unwrap();
        let reference = json(&fixture().join("reference.json")).unwrap();
        for (name, q) in &request.questions {
            let enc = e.encode(&request.state, q).unwrap();
            assert_eq!(
                serde_json::to_value(&enc).unwrap(),
                reference["probes"][name]["encoding"]
            );
        }
        let response = serde_json::to_value(e.decide(&request).unwrap()).unwrap();
        close(
            &response["answers"],
            &reference["response"]["answers"],
            1e-5,
        );
        assert_eq!(response["usage"], reference["response"]["usage"]);
    }
    #[test]
    fn future_tokens_cannot_change_past_hidden_states() {
        let e = NativeEngine::load(&fixture()).unwrap();
        let a = e.model.forward(&[1, 2, 5, 3, 7, 11]).unwrap();
        let b = e.model.forward(&[1, 2, 5, 3, 9, 10]).unwrap();
        assert_eq!(&a[..4 * 8], &b[..4 * 8]);
    }
    #[test]
    fn rejects_missing_weights_invalid_positions_and_overlong_prompts() {
        assert!(NativeEngine::load(Path::new("/nonexistent-decider-model")).is_err());
        let e = NativeEngine::load(&fixture()).unwrap();
        let q: Question = serde_json::from_value(serde_json::json!({"type":"choice","instructions":"pick","criteria":{"a":null,"b":null}})).unwrap();
        assert!(e
            .encode(&Value::String("state ".repeat(300)), &q)
            .unwrap_err()
            .contains("exceeds"));
        assert!(e
            .raw_logits(&Encoding {
                input_ids: vec![1, 2],
                option_positions: vec![20],
                answer_position: 1
            })
            .is_err());
        assert!(e.model.forward(&[100]).is_err());
    }
    #[test]
    fn malformed_decisions_and_images_are_rejected() {
        for raw in [
            serde_json::json!({"state":"x","questions":{}}),
            serde_json::json!({"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":{"one":null}}}}),
            serde_json::json!({"state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":["only"]}}}),
            serde_json::json!({"state":"x","questions":{"q":{"type":"noul","instructions":"x","criteria":{"maybe":null}}}}),
            serde_json::json!({"state":10,"questions":{"q":{"type":"noul","instructions":"x"}}}),
            serde_json::json!({"state":"x","images":["image"],"questions":{"q":{"type":"noul","instructions":"x"}}}),
        ] {
            assert!(serde_json::from_value::<Request>(raw)
                .unwrap()
                .validate()
                .is_err());
        }
    }
    #[test]
    fn confidence_measures_distinguish_ordinal_precision_and_choice_certainty() {
        assert_eq!(schema::choice_confidence(&[0.5, 0.5]), 0.0);
        assert_eq!(schema::choice_confidence(&[0.0, 1.0]), 1.0);
        assert!(
            schema::score_confidence(&[0.0, 0.5, 0.5, 0.0], 0.0)
                > schema::score_confidence(&[0.5, 0.0, 0.0, 0.5], 0.0)
        );
    }

    #[test]
    fn lower_precision_modes_identify_themselves_and_preserve_causality() {
        let request: Request = serde_json::from_str(include_str!("../tests/request.json")).unwrap();
        for precision in [Precision::Int8, Precision::Int8Compute, Precision::Int4] {
            let engine = NativeEngine::load_with_precision(&fixture(), precision).unwrap();
            let a = engine.model.forward(&[1, 2, 5, 3, 7, 11]).unwrap();
            let b = engine.model.forward(&[1, 2, 5, 3, 9, 10]).unwrap();
            assert_eq!(&a[..4 * 8], &b[..4 * 8]);
            let response = engine.decide(&request).unwrap();
            assert_eq!(response.model, precision.engine_id());
            assert_eq!(response.answers.len(), 3);
            assert_eq!(response.usage.output_tokens, 0);
        }
    }
}
