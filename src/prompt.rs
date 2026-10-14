use crate::schema::{choice_confidence, score_confidence, Answer, Question};
use serde_json::Value;
use std::collections::BTreeMap;

pub(crate) struct Rendered {
    pub text: String,
    pub labels: Vec<String>,
    /// Byte offsets, matching the Rust tokenizer (Python uses character offsets).
    pub spans: Vec<(usize, usize)>,
}

/// Upstream renders content with Python, so trimming, whitespace collapsing and
/// JSON text must follow Python rather than Rust conventions to tokenize identically.
pub fn render_content(v: &Value) -> String {
    match v {
        Value::String(s) => py_strip(s).to_string(),
        Value::Null => String::new(),
        _ => {
            let mut out = String::new();
            python_json(v, 0, &mut out);
            out
        }
    }
}

/// The shared prefix, tokenized separately from each question.
pub(crate) fn render_state(state: &Value) -> String {
    format!("<state>\n{}\n</state>\n", render_content(state))
}

/// Python `str.isspace`: Unicode White_Space plus the separators U+001C–U+001F.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
/// Python `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}
/// Python `" ".join(s.split())`.
fn py_collapse(s: &str) -> String {
    s.split(py_space)
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Python `json.dumps(v, indent=2, ensure_ascii=False)`. serde_json escapes
/// strings the same way, but formats floats differently. Integers outside the
/// 64-bit range and a literal `-0` reach this function as floats, so they render
/// as Python would render the float, not the original integer.
fn python_json(v: &Value, depth: usize, out: &mut String) {
    fn newline(out: &mut String, depth: usize) {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', 2 * depth));
    }
    fn string(s: &str, out: &mut String) {
        out.push_str(&serde_json::to_string(s).expect("strings serialize"));
    }
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => out.push_str(&i.to_string()),
            (_, Some(u), _) => out.push_str(&u.to_string()),
            (_, _, Some(f)) => python_float(f, out),
            _ => unreachable!("JSON numbers are integers or finite floats"),
        },
        Value::String(s) => string(s, out),
        Value::Array(a) if a.is_empty() => out.push_str("[]"),
        Value::Object(m) if m.is_empty() => out.push_str("{}"),
        Value::Array(a) => {
            out.push('[');
            for (i, item) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, depth + 1);
                python_json(item, depth + 1, out);
            }
            newline(out, depth);
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (key, item)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, depth + 1);
                string(key, out);
                out.push_str(": ");
                python_json(item, depth + 1, out);
            }
            newline(out, depth);
            out.push('}');
        }
    }
}

/// Python `repr(float)`: the shortest round-trip digits, in positional form when
/// the decimal exponent is in [-4, 16) and otherwise as `d.ddde±XX`.
fn python_float(x: f64, out: &mut String) {
    let split = |scientific: &str| -> (String, i32) {
        let (mantissa, exponent) = scientific
            .split_once('e')
            .expect("exponential float formatting");
        let digits = mantissa.chars().filter(|c| *c != '.').collect();
        (digits, exponent.parse().expect("integer float exponent"))
    };
    // Rust's `{:e}` gives the shortest round-trip digit count, e.g. `1.5e-7`.
    // Among equally short candidates Python takes the nearest, breaking exact
    // ties toward an even last digit. Rust's fixed-precision formatting rounds
    // correctly with the same tie rule, so prefer it whenever it round-trips.
    let shortest = format!("{:e}", x.abs());
    let (mut digits, mut exponent) = split(&shortest);
    let nearest = format!("{:.*e}", digits.len() - 1, x.abs());
    if nearest.parse::<f64>() == Ok(x.abs()) {
        (digits, exponent) = split(&nearest);
    }
    if x.is_sign_negative() {
        out.push('-');
    }
    // Position of the decimal point relative to the digits: x = 0.DIGITS × 10^point.
    let point = exponent + 1;
    if !(-3..=16).contains(&point) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push_str(&format!(
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.unsigned_abs()
        ));
    } else if point <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', point.unsigned_abs() as usize));
        out.push_str(&digits);
    } else if (point as usize) < digits.len() {
        out.push_str(&digits[..point as usize]);
        out.push('.');
        out.push_str(&digits[point as usize..]);
    } else {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
        out.push_str(".0");
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
        let desc = py_collapse(&desc);
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
                    .map(|(i, v)| (i.to_string(), py_strip(v).to_string()))
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

    fn parity_fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/prompt-parity.json")).unwrap()
    }

    #[test]
    fn content_renders_exactly_as_upstream_python() {
        let fixture = parity_fixture();
        let cases = fixture["content"].as_array().unwrap();
        assert!(cases.len() >= 20);
        for case in cases {
            assert_eq!(
                render_content(&case["value"]),
                case["rendered"].as_str().unwrap(),
                "{}",
                case["value"]
            );
        }
    }

    #[test]
    fn prompts_spans_and_legends_match_upstream_render_question() {
        for case in parity_fixture()["prompts"].as_array().unwrap() {
            let request: crate::Request = serde_json::from_value(case["request"].clone()).unwrap();
            request.validate().unwrap();
            assert_eq!(
                render_state(&request.state),
                case["state"].as_str().unwrap()
            );
            for (name, q) in &request.questions {
                let expected = &case["questions"][name];
                let text = expected["text"].as_str().unwrap();
                let rendered = render(q);
                assert_eq!(rendered.text, text, "{name}");
                assert_eq!(
                    serde_json::to_value(&rendered.labels).unwrap(),
                    expected["labels"],
                    "{name}"
                );
                // Upstream spans count characters; ours count UTF-8 bytes.
                let byte = |chars: &Value| {
                    let chars = chars.as_u64().unwrap() as usize;
                    text.char_indices()
                        .nth(chars)
                        .map_or(text.len(), |(i, _)| i)
                };
                let spans: Vec<_> = expected["spans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| (byte(&s[0]), byte(&s[1])))
                    .collect();
                assert_eq!(rendered.spans, spans, "{name}");
                if let Question::Score { criteria, .. } = q {
                    let uniform = vec![1.0 / criteria.len() as f32; criteria.len()];
                    let Answer::Score { legend, .. } = rendered.answer(q, &uniform, 0.0) else {
                        unreachable!()
                    };
                    let legend: Vec<_> = legend.values().cloned().collect();
                    assert_eq!(
                        serde_json::to_value(legend).unwrap(),
                        expected["descriptions"],
                        "{name}"
                    );
                }
            }
        }
    }

    #[test]
    fn floats_use_python_repr_switch_points_and_exponents() {
        let render = |x: f64| {
            let mut out = String::new();
            python_float(x, &mut out);
            out
        };
        for (x, expected) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (-123.25, "-123.25"),
            (25.0, "25.0"),
            // Exactly halfway between two 17-digit candidates: Python picks the even one.
            (-7_365_035_090_735_513.0 / 4.0, "-1841258772683878.2"),
        ] {
            assert_eq!(render(x), expected);
        }
    }
}
