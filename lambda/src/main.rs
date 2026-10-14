//! Private, bounded Lambda benchmark host. One resident engine per environment.
#![deny(unsafe_code)]
use decider_engine::{InferenceLimits, NativeEngine, Precision, Request};
use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde_json::{json, value::RawValue, Value};
use std::{path::Path, sync::Arc, time::Instant};
use tokio::sync::Semaphore;

fn parse_request(raw: &str) -> Result<Request, String> {
    if raw == r#""__decider_event_too_large__""#
        || raw.len() > InferenceLimits::lambda().max_request_bytes
    {
        return Err("event exceeds raw request byte limit".into());
    }
    let request: Request = serde_json::from_str(raw).map_err(|e| e.to_string())?;
    InferenceLimits::lambda().validate(&request)?;
    Ok(request)
}

fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Error> {
    let workers: usize = std::env::var("RAYON_NUM_THREADS")?.parse()?;
    if !(1..=6).contains(&workers) {
        return Err("RAYON_NUM_THREADS must be explicitly set to 1..=6".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build_global()?;
    let mode = std::env::var("DECIDER_PRECISION")?;
    let precision = match mode.as_str() {
        "checkpoint" => Precision::Checkpoint,
        "int8" => Precision::Int8,
        "int8-compute" => Precision::Int8Compute,
        _ => return Err("DECIDER_PRECISION must be checkpoint, int8, or int8-compute".into()),
    };
    let dir = std::env::var("DECIDER_MODEL_DIR").unwrap_or_else(|_| "/opt/model".into());
    let revisions: Value =
        serde_json::from_slice(&std::fs::read(Path::new(&dir).join("revisions.json"))?)?;
    let start = Instant::now();
    let engine = Arc::new(NativeEngine::load_with_precision(
        Path::new(&dir),
        precision,
    )?);
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let kernel = decider_engine::integer_kernel();
    if std::env::var("DECIDER_REQUIRE_SDOT").as_deref() == Ok("1") && kernel != "aarch64-sdot" {
        return Err("deployment requires SDOT but runtime selected scalar integer kernel".into());
    }
    let serial = Arc::new(Semaphore::new(1));
    let count = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let environment =
        std::env::var("AWS_LAMBDA_LOG_STREAM_NAME").unwrap_or_else(|_| "local".into());
    lambda_runtime::Runtime::new(service_fn(|event: LambdaEvent<Box<RawValue>>| {
        let engine = Arc::clone(&engine);
        let serial = Arc::clone(&serial);
        let count = Arc::clone(&count);
        let mode = mode.clone();
        let revisions = revisions.clone();
        let environment = environment.clone();
        async move {
            // Parse and cheap admission before waiting for a forward-pass slot.
            let request = parse_request(event.payload.get())?;
            let permit = serial.acquire_owned().await?;
            let result = tokio::task::spawn_blocking(move || -> Result<Value, String> {
                // The permit follows the blocking work even if the async waiter is cancelled.
                let _permit = permit;
                let sequence = count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let start = Instant::now();
                let response = engine.decide_with_limits(&request, &InferenceLimits::lambda())?;
                Ok(json!({"response":response, "metrics": {
                    "request_id":event.context.request_id, "environment":environment,
                    "sequence":sequence, "cold":sequence == 0, "load_ms":load_ms,
                    "inference_ms":start.elapsed().as_secs_f64()*1000.0,
                    "peak_rss_kib":peak_rss_kib(), "precision":mode,
                    "rayon_workers":workers, "architecture":std::env::consts::ARCH,
                    "integer_kernel":kernel, "model_revisions":revisions
                }}))
            })
            .await??;
            Ok::<_, Error>(result)
        }
    }))
    // Runs before SDK JSON deserialization, including whitespace in the byte count.
    // The SDK has already received the bounded Lambda transport body in memory.
    .layer(tower::util::MapRequestLayer::new(
        |mut event: lambda_runtime::LambdaInvocation| {
            if event.body.len() > InferenceLimits::lambda().max_request_bytes {
                event.body = (&b"\"__decider_event_too_large__\""[..]).into();
            }
            event
        },
    ))
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_raw_oversize_before_json_and_collection_limits_before_forward() {
        assert!(
            parse_request(&" ".repeat(InferenceLimits::lambda().max_request_bytes + 1))
                .unwrap_err()
                .contains("raw request")
        );
        let question = json!({"type":"noul", "instructions":"test"});
        let request =
            json!({"state":"", "questions":{"a":question,"b":question,"c":question,"d":question}});
        assert!(parse_request(&request.to_string()).is_err());
        let request = include_str!("../../tests/request.json");
        assert_eq!(parse_request(request).unwrap().questions.len(), 3);
    }
}
