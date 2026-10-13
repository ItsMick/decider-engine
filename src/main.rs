#![forbid(unsafe_code)]
use decider_engine::{DecisionEngine, NativeEngine, Precision, Request};
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

const LIMIT: usize = 2 * 1024 * 1024;
const HELP: &str = "Usage: decider-engine [--precision checkpoint|int8|int8-compute|int4] MODEL_DIR [REQUEST_JSON]\n\nWith a file, evaluate one JSON request. Otherwise read JSONL from stdin,\nkeeping the model resident, and write one JSON response per line.\nQuantized modes are experimental; confidence needs evaluation.\nINT8-compute also quantizes projection inputs and enables ARM dot products.\nMaximum request size: 2 MiB. No network access during inference.\n";

fn answer(engine: &NativeEngine, bytes: &[u8], out: &mut impl Write) -> Result<(), String> {
    if bytes.len() > LIMIT {
        return Err("request exceeds 2 MiB".into());
    }
    let request: Request = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let response = engine.decide(&request)?;
    serde_json::to_writer(&mut *out, &response).map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    let precision = if args.first().is_some_and(|a| a == "--precision") {
        if args.len() < 3 {
            return Err(HELP.into());
        }
        let precision = match args[1].to_str() {
            Some("checkpoint") => Precision::Checkpoint,
            Some("int8") => Precision::Int8,
            Some("int8-compute") => Precision::Int8Compute,
            Some("int4") => Precision::Int4,
            _ => return Err("precision must be checkpoint, int8, int8-compute, or int4".into()),
        };
        args.drain(..2);
        precision
    } else {
        Precision::Checkpoint
    };
    if !(1..=2).contains(&args.len()) {
        return Err(HELP.into());
    }
    let engine = NativeEngine::load_with_precision(Path::new(&args[0]), precision)?;
    let mut out = io::stdout().lock();
    if args.len() == 2 {
        let file = std::fs::File::open(&args[1]).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take((LIMIT + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        return answer(&engine, &bytes, &mut out);
    }
    let mut input = io::stdin().lock();
    loop {
        let mut line = Vec::new();
        let count = (&mut input)
            .take((LIMIT + 1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(|e| e.to_string())?;
        if count == 0 {
            return Ok(());
        }
        if line.len() > LIMIT {
            return Err("request exceeds 2 MiB".into());
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        answer(&engine, &line, &mut out)?;
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
