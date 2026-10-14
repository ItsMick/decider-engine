//! Explicit real-weight parity check. Default tests exercise the tiny checkpoint.
use decider_engine::{DecisionEngine, NativeEngine, Request};
use serde_json::Value;

#[test]
#[ignore = "set DECIDER_TEST_MODEL_DIR to the pinned v21 checkpoint; no downloads"]
fn pinned_v21_matches_pytorch_tokens_logits_and_decisions() {
    let dir = std::env::var_os("DECIDER_TEST_MODEL_DIR").expect("set DECIDER_TEST_MODEL_DIR");
    let engine = NativeEngine::load(std::path::Path::new(&dir)).unwrap();
    let req: Request = serde_json::from_str(include_str!("request.json")).unwrap();
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/v21-reference.json")).unwrap();
    for (name, q) in &req.questions {
        let encoded = engine.encode(&req.state, q).unwrap();
        assert_eq!(
            serde_json::to_value(&encoded).unwrap(),
            reference["probes"][name]["encoding"]
        );
        let logits = engine.raw_logits(&encoded).unwrap();
        let expected = reference["probes"][name]["logits"].as_array().unwrap();
        assert_eq!(logits.len(), expected.len(), "{name}: logit count");
        for (a, b) in logits.iter().zip(expected) {
            assert!(
                (*a as f64 - b.as_f64().unwrap()).abs() < 1e-4,
                "{name}: logit parity"
            );
        }
    }
    let response = serde_json::to_value(engine.decide(&req).unwrap()).unwrap();
    assert_eq!(response["usage"], reference["response"]["usage"]);
    assert_eq!(
        response["answers"].as_object().unwrap().len(),
        req.questions.len()
    );
    for (name, q) in &req.questions {
        let a = &response["answers"][name];
        let b = &reference["response"]["answers"][name];
        if q.kind() == "noul" {
            assert!((a["noul"].as_f64().unwrap() - b["noul"].as_f64().unwrap()).abs() < 1e-5);
        } else {
            assert_eq!(
                a["probabilities"].as_object().unwrap().len(),
                b["probabilities"].as_object().unwrap().len()
            );
            assert!(
                (a["confidence"].as_f64().unwrap() - b["confidence"].as_f64().unwrap()).abs()
                    < 1e-5
            );
            for (label, p) in a["probabilities"].as_object().unwrap() {
                assert!(
                    (p.as_f64().unwrap() - b["probabilities"][label].as_f64().unwrap()).abs()
                        < 1e-5,
                    "{name}/{label}: probability parity"
                );
            }
            if q.kind() == "choice" {
                assert_eq!(a["choice"], b["choice"]);
            } else {
                assert!((a["score"].as_f64().unwrap() - b["score"].as_f64().unwrap()).abs() < 1e-5);
            }
        }
    }
}
