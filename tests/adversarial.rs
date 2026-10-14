use decider_engine::{
    Answer, DecisionEngine, InferenceLimits, NativeEngine, Precision, Question, Request,
};
use serde_json::Value;
use std::path::Path;

#[test]
fn hostile_text_preserves_deterministic_binding_and_typed_answer_domains() {
    let suite: Value = serde_json::from_str(include_str!("fixtures/adversarial.json")).unwrap();
    for precision in [
        Precision::Checkpoint,
        Precision::Int8,
        Precision::Int8Compute,
        Precision::Int4,
    ] {
        let engine = NativeEngine::load_with_precision(
            Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tiny")),
            precision,
        )
        .unwrap();
        for case in suite["cases"].as_array().unwrap() {
            let req: Request = serde_json::from_value(case["request"].clone()).unwrap();
            let q = &req.questions["decision"];
            let encoded = engine.encode(&req.state, q).unwrap();
            assert_eq!(
                serde_json::to_value(&encoded).unwrap(),
                serde_json::to_value(engine.encode(&req.state, q).unwrap()).unwrap(),
                "{}",
                case["id"]
            );
            assert_eq!(encoded.answer_position, encoded.input_ids.len() - 1);
            assert!(encoded.option_positions.windows(2).all(|p| p[0] < p[1]));
            assert!(encoded
                .option_positions
                .iter()
                .all(|p| *p < encoded.answer_position));
            let expected: Vec<_> = match q {
                Question::Noul { .. } => vec!["false".into(), "true".into()],
                Question::Choice { criteria, .. } => criteria.keys().cloned().collect(),
                Question::Score { criteria, .. } => {
                    (0..criteria.len()).map(|i| i.to_string()).collect()
                }
            };
            assert_eq!(encoded.option_positions.len(), expected.len());
            let response = engine.decide(&req).unwrap();
            assert_eq!(response.answers.len(), 1);
            match &response.answers["decision"] {
                Answer::Noul { noul } => assert!((0.0..=1.0).contains(noul)),
                Answer::Choice {
                    choice,
                    probabilities,
                    confidence,
                } => {
                    assert!(expected.contains(choice));
                    assert_eq!(probabilities.len(), expected.len());
                    assert!(expected.iter().all(|k| probabilities.contains_key(k)));
                    assert!(probabilities.values().all(|p| (0.0..=1.0).contains(p)));
                    assert!((probabilities.values().sum::<f32>() - 1.0).abs() < 1e-5);
                    assert!((0.0..=1.0).contains(confidence));
                }
                Answer::Score {
                    score,
                    probabilities,
                    legend,
                    confidence,
                } => {
                    assert!((0.0..=(expected.len() - 1) as f32).contains(score));
                    assert_eq!(probabilities.len(), expected.len());
                    assert!(expected
                        .iter()
                        .all(|k| probabilities.contains_key(k) && legend.contains_key(k)));
                    assert!((probabilities.values().sum::<f32>() - 1.0).abs() < 1e-5);
                    assert!((0.0..=1.0).contains(confidence));
                }
            }
            // Delimiters do not exempt any actual tokens from admission checks.
            let limits = InferenceLimits {
                max_question_tokens: encoded.input_ids.len() - 1,
                ..InferenceLimits::default()
            };
            assert!(engine.decide_with_limits(&req, &limits).is_err());
        }
    }
}
