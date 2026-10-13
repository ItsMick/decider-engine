#![forbid(unsafe_code)]
use decider_engine::{DecisionEngine, NativeEngine, Request};
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

const LIMIT: usize = 2 * 1024 * 1024;
const HELP: &str = "Usage: decider-engine MODEL_DIR [REQUEST_JSON]\n\nWith a file, evaluate one JSON request. Otherwise read JSONL from stdin,\nkeeping the model resident, and write one JSON response per line.\nMaximum request size: 2 MiB. No network access during inference.\n";

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
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        print!("{HELP}");
        return Ok(());
    }
    if !(1..=2).contains(&args.len()) {
        return Err(HELP.into());
    }
    let engine = NativeEngine::load(Path::new(&args[0]))?;
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
