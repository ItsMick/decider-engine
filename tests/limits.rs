use decider_engine::{
    DecisionEngine, InferenceLimits, LoadLimits, NativeEngine, Precision, Request,
};
use serde_json::{json, Value};
use std::path::Path;

fn fixture() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny"))
}
fn request() -> Request {
    serde_json::from_str(include_str!("request.json")).unwrap()
}

#[test]
fn byte_collection_and_token_budgets_include_exact_boundaries() {
    let engine = NativeEngine::load(fixture()).unwrap();
    let req = request();
    let tokens: Vec<_> = req
        .questions
        .values()
        .map(|q| engine.encode(&req.state, q).unwrap().input_ids.len())
        .collect();
    let limits = InferenceLimits {
        max_request_bytes: serde_json::to_vec_pretty(&req).unwrap().len(),
        max_state_bytes: serde_json::to_vec_pretty(&req.state).unwrap().len(),
        max_questions: req.questions.len(),
        max_options: 3,
        max_question_tokens: *tokens.iter().max().unwrap(),
        max_total_tokens: tokens.iter().sum(),
    };
    let response = engine.decide_with_limits(&req, &limits).unwrap();
    assert_eq!(response.usage.input_tokens, limits.max_total_tokens);
    for (name, policy) in [
        (
            "request bytes",
            InferenceLimits {
                max_request_bytes: limits.max_request_bytes - 1,
                ..limits
            },
        ),
        (
            "state bytes",
            InferenceLimits {
                max_state_bytes: limits.max_state_bytes - 1,
                ..limits
            },
        ),
        (
            "question count",
            InferenceLimits {
                max_questions: limits.max_questions - 1,
                ..limits
            },
        ),
        (
            "option count",
            InferenceLimits {
                max_options: 2,
                ..limits
            },
        ),
        (
            "per-question tokens",
            InferenceLimits {
                max_question_tokens: limits.max_question_tokens - 1,
                ..limits
            },
        ),
        (
            "total tokens",
            InferenceLimits {
                max_total_tokens: limits.max_total_tokens - 1,
                ..limits
            },
        ),
    ] {
        assert!(
            engine
                .decide_with_limits(&req, &policy)
                .unwrap_err()
                .contains(name),
            "{name}"
        );
    }
}

#[test]
fn default_library_limits_reject_large_states_and_question_collections() {
    let engine = NativeEngine::load(fixture()).unwrap();
    let mut req = request();
    req.state = Value::String("x".repeat(InferenceLimits::default().max_state_bytes));
    assert!(engine.decide(&req).unwrap_err().contains("state bytes"));
    let mut req = request();
    for i in 0..17 {
        req.questions.insert(
            format!("q{i}"),
            serde_json::from_value(json!({"type":"noul","instructions":"x"})).unwrap(),
        );
    }
    assert!(engine.decide(&req).unwrap_err().contains("question count"));
}

#[test]
fn explicit_policy_can_raise_serving_limits_but_not_model_window() {
    let engine = NativeEngine::load(fixture()).unwrap();
    let mut req = request();
    let q = serde_json::to_value(req.questions.values().next().unwrap()).unwrap();
    req.questions.clear();
    for i in 0..17 {
        req.questions
            .insert(format!("q{i}"), serde_json::from_value(q.clone()).unwrap());
    }
    let limits = InferenceLimits {
        max_questions: 17,
        ..InferenceLimits::default()
    };
    assert_eq!(
        engine
            .decide_with_limits(&req, &limits)
            .unwrap()
            .answers
            .len(),
        17
    );
    req.state = json!("word ".repeat(300));
    assert!(engine
        .decide_with_limits(
            &req,
            &InferenceLimits {
                max_question_tokens: usize::MAX,
                ..limits
            }
        )
        .unwrap_err()
        .contains("context window"));
}

#[test]
fn nested_library_values_fail_before_recursive_serialization() {
    let mut req = request();
    for _ in 0..65 {
        req.state = Value::Array(vec![req.state]);
    }
    assert!(InferenceLimits::default()
        .validate(&req)
        .unwrap_err()
        .contains("nesting"));
}

#[test]
fn load_budgets_cover_weights_and_forward_workspace() {
    for limits in [
        LoadLimits {
            max_tensor_bytes: 1,
            ..LoadLimits::default()
        },
        LoadLimits {
            max_weight_bytes: 1,
            ..LoadLimits::default()
        },
        LoadLimits {
            max_workspace_bytes: 1,
            ..LoadLimits::default()
        },
    ] {
        let err = NativeEngine::load_with_limits(fixture(), Precision::Checkpoint, &limits)
            .err()
            .unwrap();
        assert!(err.contains("budget"), "{err}");
    }
}
