#![forbid(unsafe_code)]
use decider_engine::{DecideError, NativeEngine, Precision, Request, Response};
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

const LIMIT: usize = 2 * 1024 * 1024;
const HELP: &str = "Usage: decider-engine [--precision checkpoint|int8|int8-compute|int4] MODEL_DIR [REQUEST_JSON]\n\nWith a file, evaluate one JSON request. Otherwise read JSONL from stdin,\nkeeping the model resident, and write one JSON response or request error per line.\n--precision may appear before or after positional arguments; -- ends options.\nQuantized modes are experimental; confidence needs evaluation.\nINT8-compute also quantizes projection inputs and enables ARM dot products.\nMaximum request size: 2 MiB, excluding the JSONL newline. No network access during inference.\n";

fn answer(engine: &NativeEngine, bytes: &[u8]) -> Result<Response, DecideError> {
    if bytes.len() > LIMIT {
        return Err(DecideError::Limit("request exceeds 2 MiB".into()));
    }
    let request: Request =
        serde_json::from_slice(bytes).map_err(|e| DecideError::InvalidRequest(e.to_string()))?;
    engine.decide_checked(&request)
}

fn write_line(out: &mut impl Write, value: &impl serde::Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *out, value).map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())
}

fn drain_line(input: &mut impl BufRead) -> io::Result<()> {
    loop {
        let bytes = input.fill_buf()?;
        if bytes.is_empty() {
            return Ok(());
        }
        let end = bytes.iter().position(|b| *b == b'\n');
        let consumed = end.map_or(bytes.len(), |i| i + 1);
        input.consume(consumed);
        if end.is_some() {
            return Ok(());
        }
    }
}

fn run() -> Result<(), String> {
    let mut positional = Vec::new();
    let mut precision = Precision::Checkpoint;
    let mut seen_precision = false;
    let mut options = true;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        if options && (arg == "--help" || arg == "-h") {
            print!("{HELP}");
            return Ok(());
        } else if options && arg == "--" {
            options = false;
        } else if options && arg == "--precision" {
            if seen_precision {
                return Err("--precision may only be specified once".into());
            }
            seen_precision = true;
            precision = match args.next().as_deref().and_then(|a| a.to_str()) {
                Some("checkpoint") => Precision::Checkpoint,
                Some("int8") => Precision::Int8,
                Some("int8-compute") => Precision::Int8Compute,
                Some("int4") => Precision::Int4,
                _ => return Err("precision must be checkpoint, int8, int8-compute, or int4".into()),
            };
        } else if options && arg.to_string_lossy().starts_with('-') {
            return Err(format!(
                "unknown option {}; use -- before paths beginning with '-'",
                arg.to_string_lossy()
            ));
        } else {
            positional.push(arg);
        }
    }
    if !(1..=2).contains(&positional.len()) {
        return Err(HELP.into());
    }
    let engine = NativeEngine::load_with_precision(Path::new(&positional[0]), precision)?;
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    let accelerated = std::arch::is_aarch64_feature_detected!("dotprod");
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    let accelerated = false;
    if precision == Precision::Int8Compute && !accelerated {
        eprintln!("INT8-compute uses the scalar integer fallback on this CPU; benchmark it before deployment.");
    }
    let mut out = io::stdout().lock();
    if positional.len() == 2 {
        let file = std::fs::File::open(&positional[1]).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take((LIMIT + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        return write_line(
            &mut out,
            &answer(&engine, &bytes).map_err(|e| e.to_string())?,
        );
    }
    let mut input = io::stdin().lock();
    loop {
        let mut line = Vec::new();
        // Two extra bytes admit an exact-limit payload plus CRLF without making
        // a too-long line's suffix look like a second request.
        let count = (&mut input)
            .take((LIMIT + 2) as u64)
            .read_until(b'\n', &mut line)
            .map_err(|e| e.to_string())?;
        if count == 0 {
            return Ok(());
        }
        let newline = line.last() == Some(&b'\n');
        if newline {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
        }
        let response = if line.len() > LIMIT {
            if !newline {
                drain_line(&mut input).map_err(|e| e.to_string())?;
            }
            Err(DecideError::Limit("request exceeds 2 MiB".into()))
        } else {
            answer(&engine, &line)
        };
        match response {
            Ok(response) => write_line(&mut out, &response)?,
            Err(error) if error.is_request_error() => {
                write_line(
                    &mut out,
                    &serde_json::json!({"error":{"kind":error.kind(), "message":error.to_string()}}),
                )?;
            }
            Err(error) => return Err(error.to_string()),
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
