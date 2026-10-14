use crate::schema::{choice_confidence, score_confidence, Answer, Question};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) struct Rendered {
    pub text: String,
    pub labels: Vec<String>,
    /// Byte offsets, matching the Rust tokenizer (Python uses character offsets).
    pub spans: Vec<(usize, usize)>,
}

pub fn render_content(v: &Value) -> String {
    match v {
        Value::String(s) => s.trim().to_string(),
        Value::Null => String::new(),
        _ => serde_json::to_string_pretty(v).expect("JSON values serialize"),
    }
}
pub(crate) fn render(q: &Question) -> Rendered {
    let (instructions, header, pairs): (_, _, Vec<(String, String)>) = match q {
        Question::Noul {
            instructions,
            criteria,
        } => {
            let defaults = [
                ("false", "the statement does not hold for this state"),
                ("true", "the statement holds for this state"),
            ];
            (
                instructions,
                "Decide whether the statement is true of the state.",
                defaults
                    .iter()
                    .map(|(k, d)| {
                        (
                            (*k).into(),
                            criteria
                                .as_ref()
                                .and_then(|c| c.get(*k))
                                .map(render_content)
                                .unwrap_or_else(|| (*d).into()),
                        )
                    })
                    .collect(),
            )
        }
        Question::Choice {
            instructions,
            criteria,
        } => (
            instructions,
            "Select exactly one option.",
            criteria
                .iter()
                .map(|(k, v)| (k.clone(), render_content(v)))
                .collect(),
        ),
        Question::Score {
            instructions,
            criteria,
        } => (
            instructions,
            "Rate the state against the ordered levels below (lowest first).",
            criteria
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v.clone()))
                .collect(),
        ),
    };
    let mut text = format!(
        "<question type=\"{}\">\n{header}\n{}\n<options>\n",
        q.kind(),
        render_content(instructions)
    );
    let mut labels = Vec::new();
    let mut spans = Vec::new();
    for (i, (name, desc)) in pairs.into_iter().enumerate() {
        let desc = desc.split_whitespace().collect::<Vec<_>>().join(" ");
        let start = text.len();
        text.push_str(&format!("{}. {name}", i + 1));
        if !desc.is_empty() {
            text.push_str(&format!(" — {desc}"));
        }
        spans.push((start, text.len()));
        text.push('\n');
        labels.push(name);
    }
    text.push_str("</options>\n</question>\n<answer>");
    Rendered {
        text,
        labels,
        spans,
    }
}
impl Rendered {
    pub fn answer(&self, q: &Question, p: &[f32], smoothing: f32) -> Answer {
        let probabilities = self.labels.iter().cloned().zip(p.iter().copied()).collect();
        match q {
            Question::Noul { .. } => Answer::Noul { noul: p[1] },
            Question::Choice { .. } => {
                // First option wins exact ties, as torch.argmax does.
                let mut best = 0;
                for i in 1..p.len() {
                    if p[i] > p[best] {
                        best = i;
                    }
                }
                Answer::Choice {
                    choice: self.labels[best].clone(),
                    probabilities,
                    confidence: choice_confidence(p),
                }
            }
            Question::Score { criteria, .. } => Answer::Score {
                score: p.iter().enumerate().map(|(i, v)| i as f32 * v).sum(),
                legend: criteria
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.trim().to_string()))
                    .collect::<BTreeMap<_, _>>(),
                probabilities,
                confidence: score_confidence(p, smoothing),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delimiter_looking_content_cannot_redefine_option_spans_or_order() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/adversarial.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let request: crate::Request = serde_json::from_value(case["request"].clone()).unwrap();
            let q = &request.questions["decision"];
            let rendered = render(q);
            let labels: Vec<String> = match q {
                Question::Noul { .. } => vec!["false".into(), "true".into()],
                Question::Choice { criteria, .. } => criteria.keys().cloned().collect(),
                Question::Score { criteria, .. } => {
                    (0..criteria.len()).map(|i| i.to_string()).collect()
                }
            };
            assert_eq!(rendered.labels, labels);
            assert_eq!(rendered.spans.len(), labels.len());
            for (i, ((start, end), label)) in rendered.spans.iter().zip(labels).enumerate() {
                assert!(rendered.text[*start..*end].starts_with(&format!("{}. {label}", i + 1)));
            }
            assert!(rendered.spans.windows(2).all(|p| p[0].1 < p[1].0));
        }
    }
}
