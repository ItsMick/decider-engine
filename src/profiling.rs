//! Opt-in real-model operator timings; compiled out of normal builds.
use std::time::Instant;

pub(crate) struct Span {
    stage: &'static str,
    shape: [usize; 4],
    start: Instant,
}

impl Span {
    pub(crate) fn new(stage: &'static str, shape: [usize; 4]) -> Self {
        Self {
            stage,
            shape,
            start: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        eprintln!(
            "DECIDER_PROFILE {}",
            serde_json::json!({
                "stage": self.stage, "shape": self.shape,
                "elapsed_ms": self.start.elapsed().as_secs_f64() * 1000.0,
            })
        );
    }
}
