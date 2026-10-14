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
    assert_eq!(responses, vec![expected.clone(), expected]);
}

fn run_stdin(input: &[u8]) -> (std::process::Output, Vec<serde_json::Value>) {
    let mut child = command()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let input = input.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&input).unwrap());
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    let lines = String::from_utf8(output.stdout.clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    (output, lines)
}

#[test]
fn oversized_input_is_rejected_before_json_parsing() {
    let mut input = vec![b'x'; 2 * 1024 * 1024 + 1];
    input.extend_from_slice(b"x\n");
    input.extend_from_slice(include_str!("request.json").replace('\n', " ").as_bytes());
    let (output, lines) = run_stdin(&input);
    assert!(output.status.success());
    assert_eq!(lines.len(), 2, "the oversized line's remainder is skipped");
    assert_eq!(lines[0]["error"]["kind"], "resource_limit");
    assert!(lines[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("exceeds 2 MiB"));
    assert_eq!(lines[1]["answers"].as_object().unwrap().len(), 3);
}

#[test]
fn a_request_of_exactly_two_mebibytes_is_accepted() {
    let request = include_str!("request.json").replace('\n', " ");
    let mut line = request.into_bytes();
    line.resize(2 * 1024 * 1024, b' ');
    line.push(b'\n');
    let (output, lines) = run_stdin(&line);
    assert!(output.status.success());
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["answers"].as_object().unwrap().len(), 3);
}

#[test]
fn per_request_errors_are_reported_in_order_without_stopping() {
    let ok = r#"{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#;
    let long = serde_json::json!({
        "state": "state ".repeat(300),
        "questions": {"q": {"type": "noul", "instructions": "x"}}
    })
    .to_string();
    let wrong_model =
        r#"{"state":"x","model":"other","questions":{"q":{"type":"noul","instructions":"x"}}}"#;
    let input = format!(
        "{ok}\n{long}\nnot json\n\n{wrong_model}\n{{\"state\":\"x\",\"questions\":{{}}}}\n{ok}"
    );
    let (output, lines) = run_stdin(input.as_bytes());
    assert!(output.status.success());
    let kinds: Vec<_> = lines
        .iter()
        .map(|line| line["error"]["kind"].as_str().unwrap_or("answer"))
        .collect();
    assert_eq!(
        kinds,
        [
            "answer",
            "context_window",
            "invalid_request",
            "model_mismatch",
            "invalid_request",
            "answer"
        ]
    );
}

#[test]
fn precision_flag_may_follow_positional_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_decider-engine"))
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny"))
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/request.json"))
        .arg("--precision=int8")
        .output()
        .unwrap();
    assert!(output.status.success());
    let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["model"], "strands-decider-2b-native-int8");
    let output = command()
        .args(["--precision", "int8", "--precision", "int4"])
        .output()
        .unwrap();
    assert!(!output.status.success());
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
        let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            response["model"],
            format!("strands-decider-2b-native-{precision}")
        );
    }
}
