//! Prompt binding with the real v21 tokenizer, without model weights.
use decider_engine::{Encoder, ErrorKind, NativeEngine, Request};
use serde_json::Value;
use std::path::Path;

#[test]
fn encoder_binds_exactly_like_the_engine() {
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny"));
    let encoder = Encoder::load(dir).unwrap();
    let engine = NativeEngine::load(dir).unwrap();
    assert_eq!(encoder.max_length(), engine.max_length());
    let request: Request = serde_json::from_str(include_str!("request.json")).unwrap();
    for q in request.questions.values() {
        assert_eq!(
            serde_json::to_value(encoder.encode(&request.state, q).unwrap()).unwrap(),
            serde_json::to_value(engine.encode(&request.state, q).unwrap()).unwrap()
        );
    }
}

#[test]
#[ignore = "set DECIDER_TEST_ENCODER_DIR to a directory with the pinned v21 tokenizer.json and strands_decider_config.json"]
fn pinned_v21_prompt_binding_matches_upstream() {
    let dir = std::env::var_os("DECIDER_TEST_ENCODER_DIR").expect("set DECIDER_TEST_ENCODER_DIR");
    let encoder = Encoder::load(Path::new(&dir)).unwrap();
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/v21-encoding-reference.json")).unwrap();
    assert_eq!(reference["version"], 1);
    let (mut probes, mut rejected) = (0, 0);
    for case in reference["cases"].as_array().unwrap() {
        let id = &case["id"];
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        request.validate().unwrap();
        let results: Vec<_> = request
            .questions
            .iter()
            .map(|(name, q)| (name, encoder.encode(&request.state, q)))
            .collect();
        if case.get("rejected").is_some() {
            // Upstream's strict window rejects the request; at least one prompt
            // must exceed the window here too, and nothing else may fail.
            assert!(results.iter().any(|(_, r)| r
                .as_ref()
                .is_err_and(|e| e.kind() == ErrorKind::ContextWindow)));
            assert!(results.iter().all(|(_, r)| r
                .as_ref()
                .map_or_else(|e| e.kind() == ErrorKind::ContextWindow, |_| true)));
            rejected += 1;
            continue;
        }
        for (name, result) in results {
            let encoding = result.unwrap_or_else(|e| panic!("{id} {name}: {e}"));
            assert_eq!(
                serde_json::to_value(encoding).unwrap(),
                case["probes"][name]["encoding"],
                "{id} {name}"
            );
            probes += 1;
        }
    }
    eprintln!("{probes} encodings and {rejected} window rejections match upstream");
    assert!(probes >= 10 && rejected >= 1);
}
