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
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr)
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
        let response: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            response["model"],
            format!("strands-decider-2b-native-{precision}")
        );
    }
}
