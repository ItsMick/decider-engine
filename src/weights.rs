//! Validated F32/BF16/F16 safetensors; no pickle or model code execution.
use crate::tensor::linear;
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, path::Path};

pub(crate) struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
pub(crate) struct Weights {
    tensors: HashMap<String, Tensor>,
}
impl Weights {
    pub fn file(path: &Path, prefix: Option<&str>) -> Result<Self, String> {
        let mut w = Self {
            tensors: HashMap::new(),
        };
        w.extend(path, prefix)?;
        Ok(w)
    }
    pub fn base(dir: &Path) -> Result<Self, String> {
        let mut w = Self {
            tensors: HashMap::new(),
        };
        let single = dir.join("model.safetensors");
        if single.exists() {
            w.extend(&single, Some("model.language_model."))?;
        } else {
            let index: serde_json::Value = serde_json::from_slice(
                &std::fs::read(dir.join("model.safetensors.index.json"))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let map = index["weight_map"]
                .as_object()
                .ok_or("missing weight_map in base index")?;
            let mut files = std::collections::BTreeSet::new();
            for (name, file) in map {
                if name.starts_with("model.language_model.") {
                    let f = file.as_str().ok_or("invalid base shard name")?;
                    if f.starts_with('.')
                        || f.contains('\\')
                        || !matches!(
                            Path::new(f).components().collect::<Vec<_>>().as_slice(),
                            [std::path::Component::Normal(_)]
                        )
                    {
                        return Err("invalid shard path".into());
                    }
                    files.insert(f.to_string());
                }
            }
            if files.is_empty() {
                return Err("no Qwen3.5 text weights in base index".into());
            }
            for f in files {
                w.extend(&dir.join(f), Some("model.language_model."))?;
            }
        }
        Ok(w)
    }
    fn extend(&mut self, path: &Path, prefix: Option<&str>) -> Result<(), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let st =
            SafeTensors::deserialize(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        for (name, t) in st.tensors() {
            let name = if let Some(p) = prefix {
                match name.strip_prefix(p) {
                    Some(n) => n.to_string(),
                    None => continue,
                }
            } else {
                name
            };
            let data = decode(t.data(), t.dtype())?;
            if data.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite weights in {name}"));
            }
            if self
                .tensors
                .insert(
                    name.clone(),
                    Tensor {
                        data,
                        shape: t.shape().to_vec(),
                    },
                )
                .is_some()
            {
                return Err(format!("duplicate tensor {name}"));
            }
        }
        Ok(())
    }
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if t.shape != shape {
            return Err(format!("{name}: expected {shape:?}, found {:?}", t.shape));
        }
        Ok(t.data)
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn remaining(&self) -> usize {
        self.tensors.len()
    }
}

fn decode(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()),
        Dtype::BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect()),
        Dtype::F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half(u16::from_le_bytes(*c)))
            .collect()),
        _ => Err(format!("unsupported weight dtype {dtype:?}")),
    }
}
fn half(v: u16) -> f32 {
    let sign = if v & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = (v >> 10) & 31;
    let m = v & 1023;
    match e {
        0 => sign * m as f32 * 2f32.powi(-24),
        31 => {
            if m == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
    }
}

pub(crate) struct Linear {
    w: Vec<f32>,
    bias: Option<Vec<f32>>,
    pub input: usize,
    pub output: usize,
    lora: Option<(Vec<f32>, Vec<f32>, usize, f32)>,
}
impl Linear {
    pub fn load(
        w: &mut Weights,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            w: w.take(&format!("{prefix}.weight"), &[output, input])?,
            bias: if bias {
                Some(w.take(&format!("{prefix}.bias"), &[output])?)
            } else {
                None
            },
            input,
            output,
            lora: None,
        })
    }
    pub fn adapter(
        &mut self,
        w: &mut Weights,
        prefix: &str,
        rank: usize,
        scale: f32,
    ) -> Result<(), String> {
        let a = format!("base_model.model.{prefix}.lora_A.weight");
        let b = format!("base_model.model.{prefix}.lora_B.weight");
        // The supported Strands Qwen checkpoint adapts every torso projection.
        // Missing a whole A/B pair must fail rather than silently run the base.
        if !w.contains(&a) || !w.contains(&b) {
            return Err(format!("missing required LoRA pair for {prefix}"));
        }
        self.lora = Some((
            w.take(&a, &[rank, self.input])?,
            w.take(&b, &[self.output, rank])?,
            rank,
            scale,
        ));
        Ok(())
    }
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.input;
        let mut out = vec![0.0; rows * self.output];
        linear(
            x,
            rows,
            self.input,
            &self.w,
            self.output,
            self.bias.as_deref(),
            &mut out,
        );
        if let Some((a, b, rank, scale)) = &self.lora {
            let mut mid = vec![0.0; rows * rank];
            linear(x, rows, self.input, a, *rank, None, &mut mid);
            let mut delta = vec![0.0; out.len()];
            linear(&mid, rows, *rank, b, self.output, None, &mut delta);
            for (o, d) in out.iter_mut().zip(delta) {
                *o += scale * d;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_and_bfloat_readout() {
        assert_eq!(
            decode(&[0x80, 0x3f, 0x00, 0xc0], Dtype::BF16).unwrap(),
            vec![1.0, -2.0]
        );
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(1), 2f32.powi(-24));
        assert!(decode(&[0], Dtype::I8).is_err());
    }
}
