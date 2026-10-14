//! Serving limits are independent of the checkpoint's trained context window.
use crate::{DecideError, Question, Request};
use serde::Serialize;
use std::io::{self, Write};

/// Per-call resource policy. Byte budgets count canonical pretty-printed JSON,
/// including keys and escaping, without allocating a serialized copy. Network
/// hosts must also bound the wire body before deserialization and concurrency.
#[derive(Clone, Copy, Debug)]
pub struct InferenceLimits {
    pub max_request_bytes: usize,
    pub max_state_bytes: usize,
    pub max_questions: usize,
    pub max_options: usize,
    pub max_question_tokens: usize,
    pub max_total_tokens: usize,
}

impl Default for InferenceLimits {
    fn default() -> Self {
        Self {
            max_request_bytes: 2 * 1024 * 1024,
            max_state_bytes: 256 * 1024,
            max_questions: 16,
            max_options: 255,
            max_question_tokens: 4096,
            max_total_tokens: 8192,
        }
    }
}

impl InferenceLimits {
    /// Conservative starting policy, not a latency or memory guarantee.
    pub fn lambda() -> Self {
        Self {
            max_request_bytes: 256 * 1024,
            max_state_bytes: 64 * 1024,
            max_questions: 3,
            max_options: 32,
            max_question_tokens: 2048,
            max_total_tokens: 4096,
        }
    }

    /// Validate byte/collection budgets before rendering or tokenization.
    pub fn validate(&self, request: &Request) -> Result<(), String> {
        self.validate_checked(request).map_err(|e| e.to_string())
    }

    /// Typed variant of `validate`, distinguishing budgets from schema errors.
    pub fn validate_checked(&self, request: &Request) -> Result<(), DecideError> {
        self.validate_budgets(request).map_err(DecideError::Limit)?;
        request.validate().map_err(DecideError::InvalidRequest)
    }

    fn validate_budgets(&self, request: &Request) -> Result<(), String> {
        check(
            "question count",
            request.questions.len(),
            self.max_questions,
        )?;
        for q in request.questions.values() {
            let options = match q {
                Question::Noul { .. } => 2,
                Question::Choice { criteria, .. } => criteria.len(),
                Question::Score { criteria, .. } => criteria.len(),
            };
            check("option count", options, self.max_options)?;
        }
        // Bound recursion before serde or the prompt renderer visits a Value
        // supplied directly by a library caller (without serde's parser guard).
        check_depth(&request.state, 0)?;
        for q in request.questions.values() {
            match q {
                Question::Noul {
                    instructions,
                    criteria,
                } => {
                    check_depth(instructions, 0)?;
                    if let Some(c) = criteria {
                        for v in c.values() {
                            check_depth(v, 0)?;
                        }
                    }
                }
                Question::Choice {
                    instructions,
                    criteria,
                } => {
                    check_depth(instructions, 0)?;
                    for v in criteria.values() {
                        check_depth(v, 0)?;
                    }
                }
                Question::Score { instructions, .. } => check_depth(instructions, 0)?,
            }
        }
        json_budget(&request.state, self.max_state_bytes, "state bytes")?;
        json_budget(request, self.max_request_bytes, "request bytes")?;
        Ok(())
    }

    pub(crate) fn add_tokens(&self, total: usize, tokens: usize) -> Result<usize, String> {
        check("per-question tokens", tokens, self.max_question_tokens)?;
        let total = total
            .checked_add(tokens)
            .ok_or("resource budget: total tokens overflow")?;
        check("total tokens", total, self.max_total_tokens)?;
        Ok(total)
    }
}

fn check_depth(value: &serde_json::Value, depth: usize) -> Result<(), String> {
    if depth > 64 {
        return Err("resource budget: JSON content nesting exceeds 64".into());
    }
    match value {
        serde_json::Value::Array(values) => {
            for v in values {
                check_depth(v, depth + 1)?;
            }
        }
        serde_json::Value::Object(values) => {
            for v in values.values() {
                check_depth(v, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn check(name: &str, actual: usize, limit: usize) -> Result<(), String> {
    if actual > limit {
        Err(format!(
            "resource budget: {name} {actual} exceeds limit {limit}"
        ))
    } else {
        Ok(())
    }
}

fn json_budget(value: &impl Serialize, limit: usize, name: &str) -> Result<(), String> {
    struct Counter(usize);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 = self
                .0
                .checked_sub(bytes.len())
                .ok_or_else(|| io::Error::other("byte limit exceeded"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer_pretty(Counter(limit), value)
        .map_err(|e| format!("resource budget: {name} exceeds limit {limit}: {e}"))
}
