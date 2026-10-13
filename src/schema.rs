//! Typed decisions; confidence follows the upstream Strands formulas.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: Value,
    pub questions: BTreeMap<String, Question>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Noul {
        instructions: Value,
        #[serde(default)]
        criteria: Option<Map<String, Value>>,
    },
    Choice {
        instructions: Value,
        criteria: Map<String, Value>,
    },
    Score {
        instructions: Value,
        criteria: Vec<String>,
    },
}

impl Question {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let instructions = match self {
            Self::Noul {
                instructions,
                criteria,
            } => {
                if criteria
                    .as_ref()
                    .is_some_and(|c| c.keys().any(|k| k != "true" && k != "false"))
                {
                    return Err("noul criteria keys must be true or false".into());
                }
                if let Some(c) = criteria {
                    for v in c.values() {
                        content(v, true)?;
                    }
                }
                instructions
            }
            Self::Choice {
                instructions,
                criteria,
            } => {
                if !(2..=255).contains(&criteria.len()) {
                    return Err("choice requires 2–255 options".into());
                }
                for (k, v) in criteria {
                    if k.is_empty() || k.chars().any(char::is_control) {
                        return Err("option labels must be nonempty single lines".into());
                    }
                    content(v, true)?;
                }
                instructions
            }
            Self::Score {
                instructions,
                criteria,
            } => {
                if !(2..=10).contains(&criteria.len()) {
                    return Err("score requires 2–10 ordered levels".into());
                }
                instructions
            }
        };
        content(instructions, false)
    }
}

pub(crate) fn content(v: &Value, nullable: bool) -> Result<(), String> {
    if v.is_string() || v.is_object() || v.is_array() || (nullable && v.is_null()) {
        Ok(())
    } else {
        Err("content must be a string, object, or array".into())
    }
}

impl Request {
    pub fn validate(&self) -> Result<(), String> {
        content(&self.state, false)?;
        if self.questions.is_empty() {
            return Err("at least one question is required".into());
        }
        if !self.images.is_empty() {
            return Err("the native engine supports text only".into());
        }
        for q in self.questions.values() {
            q.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}

pub fn choice_confidence(p: &[f32]) -> f32 {
    if p.len() <= 1 {
        return 1.0;
    }
    let n = p.len() as f32;
    ((n * p.iter().copied().fold(0.0, f32::max) - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}
pub fn score_confidence(p: &[f32], smoothing: f32) -> f32 {
    if p.len() <= 1 {
        return 1.0;
    }
    let total: f32 = p.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mean: f32 = p
        .iter()
        .enumerate()
        .map(|(i, v)| i as f32 * v / total)
        .sum();
    let sigma = p
        .iter()
        .enumerate()
        .map(|(i, v)| v / total * (i as f32 - mean).powi(2))
        .sum::<f32>()
        .sqrt();
    let max = (p.len() - 1) as f32 / 2.0;
    let floor = smoothing.max(0.0).sqrt();
    if max <= floor {
        return if sigma <= floor { 1.0 } else { 0.0 };
    }
    ((max - sigma) / (max - floor)).clamp(0.0, 1.0)
}
