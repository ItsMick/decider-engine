#![forbid(unsafe_code)]
use decider_engine::{DecideError, ErrorKind, InferenceLimits, NativeEngine, Precision, Request};
use std::io::{self, BufRead, Read, Write};
use std::path::Path;

const LIMIT: usize = 2 * 1024 * 1024;
const HELP: &str = "Usage: decider-engine [--precision checkpoint|int8|int8-compute|int4] MODEL_DIR [REQUEST_JSON]\n\nWith a file, evaluate one JSON request. Otherwise read JSONL from stdin,\nkeeping the model resident, and write one JSON line per non-blank input line:\na response, or {\"error\":{\"kind\":...,\"message\":...}} for a request that\ncannot be answered. Only I/O failures stop the stdin loop.\nQuantized modes are experimental; confidence needs evaluation.\nINT8-compute also quantizes projection inputs and enables ARM dot products.\nMaximum request size: 2 MiB. No network access during inference.\n";

fn parse_args() -> Result<Option<(Precision, Vec<std::ffi::OsString>)>, String> {
    let mut precision = None;
    let mut positional = Vec::new();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let value = match arg.to_str() {
            Some("-h" | "--help") => return Ok(None),
            Some("--precision") => Some(args.next().ok_or(HELP)?),
            Some(s) if s.starts_with("--precision=") => Some(s["--precision=".len()..].into()),
            Some(s) if s.starts_with('-') && s != "-" => {
                return Err(format!("unknown option {s}\n\n{HELP}"))
            }
            _ => {
                positional.push(arg);
                None
            }
        };
        if let Some(value) = value {
            if precision.is_some() {
                return Err("--precision given more than once".into());
            }
            precision = Some(match value.to_str() {
                Some("checkpoint") => Precision::Checkpoint,
                Some("int8") => Precision::Int8,
                Some("int8-compute") => Precision::Int8Compute,
                Some("int4") => Precision::Int4,
                _ => return Err("precision must be checkpoint, int8, int8-compute, or int4".into()),
            });
        }
    }
    if !(1..=2).contains(&positional.len()) {
        return Err(HELP.into());
    }
    Ok(Some((precision.unwrap_or_default(), positional)))
}

fn answer(engine: &NativeEngine, bytes: &[u8]) -> Result<String, DecideError> {
    let request: Request = serde_json::from_slice(bytes)
        .map_err(|e| DecideError::new(ErrorKind::InvalidRequest, e.to_string()))?;
    let response = engine.evaluate(&request, &InferenceLimits::default())?;
    Ok(serde_json::to_string(&response).expect("responses serialize"))
}

/// Discard the rest of an oversized line in bounded reads, so it is never parsed
/// as a request and never held in memory.
fn skip_line(input: &mut impl BufRead) -> io::Result<()> {
    loop {
        let buffer = input.fill_buf()?;
        if buffer.is_empty() {
            return Ok(());
        }
        if let Some(end) = buffer.iter().position(|b| *b == b'\n') {
            input.consume(end + 1);
            return Ok(());
        }
        let length = buffer.len();
        input.consume(length);
    }
}

fn error_line(error: &DecideError) -> String {
    serde_json::to_string(&serde_json::json!({ "error": error })).expect("errors serialize")
}

fn run() -> Result<(), String> {
    let Some((precision, args)) = parse_args()? else {
        print!("{HELP}");
        return Ok(());
    };
    if precision == Precision::Int8Compute && !decider_engine::int8_compute_accelerated() {
        eprintln!("warning: this CPU has no native INT8 dot products; int8-compute uses a portable fallback that is usually slower than checkpoint precision");
    }
    let engine = NativeEngine::load_with_precision(Path::new(&args[0]), precision)?;
    let mut out = io::stdout().lock();
    let io_error = |e: io::Error| e.to_string();
    if args.len() == 2 {
        let file = std::fs::File::open(&args[1]).map_err(io_error)?;
        let mut bytes = Vec::new();
        file.take((LIMIT + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(io_error)?;
        if bytes.len() > LIMIT {
            return Err("request exceeds 2 MiB".into());
        }
        let line = answer(&engine, &bytes)?;
        writeln!(out, "{line}").map_err(io_error)?;
        return out.flush().map_err(io_error);
    }
    let mut input = io::stdin().lock();
    loop {
        // Read at most the limit plus the line's terminating newline.
        let mut line = Vec::new();
        let count = (&mut input)
            .take((LIMIT + 1) as u64)
            .read_until(b'\n', &mut line)
            .map_err(io_error)?;
        if count == 0 {
            return Ok(());
        }
        let complete = line.last() == Some(&b'\n');
        if complete {
            line.pop();
        }
        let reply = if !complete && line.len() > LIMIT {
            skip_line(&mut input).map_err(io_error)?;
            error_line(&DecideError::new(
                ErrorKind::ResourceLimit,
                "request exceeds 2 MiB",
            ))
        } else if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        } else {
            answer(&engine, &line).unwrap_or_else(|e| error_line(&e))
        };
        writeln!(out, "{reply}").map_err(io_error)?;
        out.flush().map_err(io_error)?;
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
