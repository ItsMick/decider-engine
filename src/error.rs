//! Typed decision errors, so hosts can separate client mistakes from engine faults.
use serde::Serialize;
use std::fmt;

/// Why a request was not answered. Every kind except `Engine` is caused by the
/// request itself and can be reported back to the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorKind {
    /// The request violates the schema or its documented constraints.
    InvalidRequest,
    /// A bound prompt is longer than the checkpoint's context window.
    ContextWindow,
    /// The request exceeds an `InferenceLimits` budget or the CLI input bound.
    ResourceLimit,
    /// The request pins a model ID this engine does not serve.
    ModelMismatch,
    /// Tokenization, inference or numerical failure inside the engine.
    Engine,
}

/// An error from `NativeEngine::evaluate`. `Display` prints the message alone,
/// matching the strings returned by `decide`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DecideError {
    kind: ErrorKind,
    message: String,
}

impl DecideError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    pub fn message(&self) -> &str {
        &self.message
    }
    /// True when the request caused the error rather than the engine.
    pub fn is_client_error(&self) -> bool {
        self.kind != ErrorKind::Engine
    }
}

impl fmt::Display for DecideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DecideError {}

impl From<DecideError> for String {
    fn from(error: DecideError) -> Self {
        error.message
    }
}
