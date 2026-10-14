use decider_engine::{
    DecideError, DecisionEngine, InferenceLimits, NativeEngine, Precision, Request,
};
use std::path::Path;

#[test]
fn model_pin_requires_exact_precision_and_declared_revision() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    for precision in [
        Precision::Checkpoint,
        Precision::Int8,
        Precision::Int8Compute,
        Precision::Int4,
    ] {
        let engine = NativeEngine::load_with_precision(&root, precision).unwrap();
        assert!(engine.checkpoint_revision().is_none());
        let mut req: Request = serde_json::from_str(include_str!("request.json")).unwrap();
        for id in [
            "strands-decider-2b-native",
            "strands-decider-2b-native-int8",
            "strands-decider-2b-native-int8-compute",
            "strands-decider-2b-native-int4",
            "strands-decider-latest",
            "strands-decider-2B-hobson-v21",
            "unknown",
        ] {
            req.model = Some(id.into());
            let result = engine.decide_checked(&req);
            if id == engine.model() || id == "strands-decider-latest" {
                assert!(result.is_ok(), "{id}");
            } else {
                assert!(
                    matches!(result, Err(DecideError::InvalidRequest(_))),
                    "{id}"
                );
            }
        }
    }
}

#[test]
fn typed_admission_distinguishes_schema_policy_and_trained_window() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let engine = NativeEngine::load(&root).unwrap();
    let mut req: Request = serde_json::from_str(include_str!("request.json")).unwrap();
    let limits = InferenceLimits {
        max_question_tokens: 1,
        ..InferenceLimits::default()
    };
    assert!(matches!(
        engine.decide_with_limits_checked(&req, &limits),
        Err(DecideError::Limit(_))
    ));
    req.state = serde_json::json!("state ".repeat(300));
    assert!(matches!(
        engine.decide_checked(&req),
        Err(DecideError::ContextWindow { limit: 256, .. })
    ));
    req.questions.clear();
    assert!(matches!(
        engine.decide_checked(&req),
        Err(DecideError::InvalidRequest(_))
    ));
    assert!(!DecideError::Engine("fault".into()).is_request_error());
}

#[test]
fn revision_metadata_controls_v21_alias_without_claiming_integrity() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny");
    let root = std::env::temp_dir().join(format!("decider-revisions-{}", std::process::id()));
    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &to.join(entry.file_name()));
            } else {
                std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
            }
        }
    }
    copy_dir(&source, &root);
    let mut request: Request = serde_json::from_str(include_str!("request.json")).unwrap();
    request.model = Some("strands-decider-2B-hobson-v21".into());
    for revision in [
        "0000000000000000000000000000000000000000",
        "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5",
    ] {
        std::fs::write(root.join("revisions.json"), serde_json::json!({"decider":revision, "base":"b1485b2fa6dfa1287294f269f5fb618e03d52d7c"}).to_string()).unwrap();
        for precision in [
            Precision::Checkpoint,
            Precision::Int8,
            Precision::Int8Compute,
            Precision::Int4,
        ] {
            let engine = NativeEngine::load_with_precision(&root, precision).unwrap();
            assert_eq!(engine.checkpoint_revision().unwrap().decider, revision);
            assert_eq!(
                engine.decide_checked(&request).is_ok(),
                revision.starts_with("2b52")
            );
        }
    }
    std::fs::write(root.join("revisions.json"), "{}").unwrap();
    assert!(NativeEngine::load(&root).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
