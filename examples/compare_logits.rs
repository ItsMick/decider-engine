//! Compare already-bound prompts without tokenization or HTTP overhead.
use decider_engine::{Encoding, NativeEngine, Precision};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: compare_logits MODEL CASES_JSON REPEATS THREADS".into());
    }
    let repeats: usize = a[3].parse()?;
    let threads: usize = a[4].parse()?;
    if repeats == 0 || threads == 0 {
        return Err("positive repeats/threads required".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let start = Instant::now();
    let engine = NativeEngine::load_with_precision(Path::new(&a[1]), Precision::Checkpoint)?;
    println!(
        "{}",
        json!({"load_ms":start.elapsed().as_secs_f64()*1000.0,"threads":threads})
    );
    for case in cases {
        let encoding: Encoding = serde_json::from_value(case["encoding"].clone())?;
        for sample in 0..=repeats {
            let start = Instant::now();
            let logits = engine.raw_logits(&encoding)?;
            println!(
                "{}",
                json!({"id":case["id"],"sample":sample,"warmup":sample==0,"tokens":encoding.input_ids.len(),"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"logits":logits})
            );
        }
    }
    Ok(())
}
