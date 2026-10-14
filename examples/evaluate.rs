//! Bounded, opt-in semantic/precision smoke evaluation; never downloads weights.
use decider_engine::{DecisionEngine, NativeEngine, Precision, Request};
use serde_json::{json, Value};
use std::{io::Read, path::Path, time::Instant};

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "usage: evaluate MODEL_DIR SUITE_JSON checkpoint|int8|int8-compute|int4".into(),
        );
    }
    let precision = match args[2].as_str() {
        "checkpoint" => Precision::Checkpoint,
        "int8" => Precision::Int8,
        "int8-compute" => Precision::Int8Compute,
        "int4" => Precision::Int4,
        _ => return Err("unknown precision".into()),
    };
    let mut bytes = Vec::new();
    std::fs::File::open(&args[1])
        .map_err(|e| e.to_string())?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err("evaluation suite exceeds 8 MiB".into());
    }
    let suite: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let cases = suite["cases"].as_array().ok_or("suite needs cases")?;
    if cases.is_empty() || cases.len() > 256 {
        return Err("suite requires 1–256 cases".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut requests = Vec::new();
    for case in cases {
        let id = case["id"].as_str().ok_or("case needs string id")?;
        if !ids.insert(id) {
            return Err(format!("duplicate case {id}"));
        }
        let request: Request =
            serde_json::from_value(case["request"].clone()).map_err(|e| e.to_string())?;
        decider_engine::InferenceLimits::default().validate(&request)?;
        requests.push(request);
    }
    let start = Instant::now();
    let engine = NativeEngine::load_with_precision(Path::new(&args[0]), precision)?;
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut results = Vec::new();
    for (case, request) in cases.iter().zip(requests) {
        let mut bindings = serde_json::Map::new();
        for (name, q) in &request.questions {
            let encoding = engine.encode(&request.state, q)?;
            bindings.insert(name.clone(), json!({"tokens":encoding.input_ids.len(), "option_positions":encoding.option_positions, "answer_position":encoding.answer_position}));
        }
        let start = Instant::now();
        let response = engine.decide(&request)?;
        results.push(json!({"id":case["id"], "bindings":bindings, "elapsed_ms":start.elapsed().as_secs_f64()*1000.0, "response":response}));
    }
    println!("{}", serde_json::to_string_pretty(&json!({
        "suite_version":suite["version"], "precision":args[2], "architecture":std::env::consts::ARCH,
        "rayon_workers":rayon::current_num_threads(), "load_ms":load_ms, "cases":results,
        "warning":"Smoke evaluation only; timings are single samples, not latency quantiles or Lambda measurements."
    })).map_err(|e| e.to_string())?);
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
