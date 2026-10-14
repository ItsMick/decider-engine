use std::fmt;

/// Typed serving failures. Request errors may be returned to a client; engine
/// faults should stop the resident worker and be investigated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecideError {
    InvalidRequest(String),
    Limit(String),
    ContextWindow { tokens: usize, limit: usize },
    Engine(String),
}
impl DecideError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::Limit(_) => "limit",
            Self::ContextWindow { .. } => "context_window",
            Self::Engine(_) => "engine",
        }
    }
    pub fn is_request_error(&self) -> bool {
        !matches!(self, Self::Engine(_))
    }
}
impl fmt::Display for DecideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::Limit(message) | Self::Engine(message) => f.write_str(message),
            Self::ContextWindow { tokens, limit } => write!(f, "prompt of {tokens} tokens exceeds context window of {limit}; shorten the state or question"),
        }
    }
}
impl std::error::Error for DecideError {}
