use std::io::Write;
use std::process::{Command, Stdio};

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_decider-engine"));
    command.arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny"));
    command
}

#[test]
fn resident_jsonl_matches_file_mode() {
    let output = command()
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/request.json"))
        .output()
        .unwrap();
    assert!(output.status.success());
    let expected: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let request: serde_json::Value = serde_json::from_str(include_str!("request.json")).unwrap();
    let line = serde_json::to_string(&request).unwrap();
    let mut child = command()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{line}\n\n{line}").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let responses: Vec<serde_json::Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0], expected);
    assert_eq!(responses[1]["error"]["kind"], "invalid_request");
    assert_eq!(responses[2], expected);
}

#[test]
fn oversized_input_is_rejected_before_json_parsing() {
    let mut child = command()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&vec![b'x'; 2 * 1024 * 1024 + 1])
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let error: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(error["error"]["kind"], "limit");
    assert!(error["error"]["message"]
        .as_str()
        .unwrap()
        .contains("exceeds 2 MiB"));
}

#[test]
fn precision_flag_selects_the_requested_engine() {
    for precision in ["int8", "int8-compute", "int4"] {
        let output = Command::new(env!("CARGO_BIN_EXE_decider-engine"))
            .args(["--precision", precision])
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny"))
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/request.json"))
            .output()
            .unwrap();
        assert!(output.status.success());
        #[cfg(not(target_arch = "aarch64"))]
        if precision == "int8-compute" {
            assert!(String::from_utf8_lossy(&output.stderr).contains("scalar integer fallback"));
        }
        let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            response["model"],
            format!("strands-decider-2b-native-{precision}")
        );
    }
}

#[test]
fn mixed_stream_recovers_after_schema_context_and_oversized_requests() {
    let good = r#"{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#;
    let long = serde_json::json!({"state":"state ".repeat(300), "questions":{"q":{"type":"noul","instructions":"x"}}}).to_string();
    let mut child = command()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{good}\n{{bad json}}\n{long}").unwrap();
    input.write_all(&vec![b'x'; 2 * 1024 * 1024 + 100]).unwrap();
    writeln!(input, "\n{good}").unwrap();
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let rows: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0], rows[4]);
    assert_eq!(rows[1]["error"]["kind"], "invalid_request");
    assert_eq!(rows[2]["error"]["kind"], "context_window");
    assert_eq!(rows[3]["error"]["kind"], "limit");
}

#[test]
fn exact_byte_limit_excludes_lf_and_crlf_and_precision_can_follow_paths() {
    let good = r#"{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#;
    for newline in ["\n", "\r\n"] {
        let mut child = command()
            .args(["--precision", "int4"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        input.write_all(good.as_bytes()).unwrap();
        input
            .write_all(&vec![b' '; 2 * 1024 * 1024 - good.len()])
            .unwrap();
        input.write_all(newline.as_bytes()).unwrap();
        drop(input);
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(response["model"], "strands-decider-2b-native-int4");
        assert!(response.get("error").is_none());
    }
}
