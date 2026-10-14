//! Validated F32/BF16/F16 safetensors; no pickle or model code execution.
use crate::tensor::{linear, Adapter, Epilogue, Projection};
use crate::Precision;
use safetensors::{tensor::Metadata, Dtype};
use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

/// Preserve checkpoint precision or quantize selected base projections.
pub(crate) enum Data {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    F16(Vec<u16>),
    Quantized {
        values: Vec<u8>,
        scales: Vec<f32>,
        width: usize,
        elements: usize,
        bits: u8,
        compute_int8: bool,
    },
}
impl Data {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Bf16(v) | Self::F16(v) => v.len(),
            Self::Quantized { elements, .. } => *elements,
        }
    }
    pub fn row<'a>(&'a self, start: usize, len: usize, scratch: &'a mut Vec<f32>) -> &'a [f32] {
        match self {
            Self::F32(v) => &v[start..start + len],
            Self::Bf16(v) => {
                scratch.clear();
                scratch.extend(
                    v[start..start + len]
                        .iter()
                        .map(|v| f32::from_bits((*v as u32) << 16)),
                );
                scratch
            }
            Self::F16(v) => {
                scratch.clear();
                scratch.extend(v[start..start + len].iter().map(|v| half(*v)));
                scratch
            }
            Self::Quantized {
                values,
                scales,
                width,
                bits,
                ..
            } => {
                let groups = width.div_ceil(64);
                scratch.clear();
                scratch.reserve(len);
                let mut position = start;
                while position < start + len {
                    let row = position / width;
                    let offset = position % width;
                    let count = (start + len - position)
                        .min(64 - offset % 64)
                        .min(width - offset);
                    let scale = scales[row * groups + offset / 64];
                    scratch.extend((position..position + count).map(|i| {
                        let q = if *bits == 8 {
                            values[i] as i8 as f32
                        } else {
                            let nibble = (values[i / 2] >> (4 * (i % 2))) & 15;
                            nibble as f32 - 8.0
                        };
                        q * scale
                    }));
                    position += count;
                }
                scratch
            }
        }
    }
    fn into_f32(self) -> Vec<f32> {
        match self {
            Self::F32(v) => v,
            Self::Bf16(v) => v
                .into_iter()
                .map(|v| f32::from_bits((v as u32) << 16))
                .collect(),
            Self::F16(v) => v.into_iter().map(half).collect(),
            data @ Self::Quantized { .. } => {
                let mut result = Vec::new();
                data.row(0, data.len(), &mut result);
                result
            }
        }
    }
    pub(crate) fn quantize(self, width: usize, precision: Precision) -> Self {
        let bits = match precision {
            Precision::Checkpoint => return self,
            Precision::Int8 | Precision::Int8Compute => 8,
            Precision::Int4 => 4,
        };
        let elements = self.len();
        assert!(width > 0 && elements.is_multiple_of(width));
        let mut values = vec![
            0;
            if bits == 8 {
                elements
            } else {
                elements.div_ceil(2)
            }
        ];
        let mut scales = Vec::with_capacity((elements / width) * width.div_ceil(64));
        let mut scratch = Vec::new();
        let bound = if bits == 8 { 127.0 } else { 7.0 };
        for row in 0..elements / width {
            let weights = self.row(row * width, width, &mut scratch);
            for (group, weights) in weights.chunks(64).enumerate() {
                let max = weights.iter().map(|v| v.abs()).fold(0.0, f32::max);
                let scale = if max == 0.0 {
                    1.0
                } else {
                    (max / bound).max(f32::MIN_POSITIVE)
                };
                scales.push(scale);
                for (offset, weight) in weights.iter().enumerate() {
                    let index = row * width + group * 64 + offset;
                    let q = (weight / scale).round().clamp(-bound, bound) as i8;
                    if bits == 8 {
                        values[index] = q as u8;
                    } else {
                        values[index / 2] |= ((q + 8) as u8) << (4 * (index % 2));
                    }
                }
            }
        }
        Self::Quantized {
            values,
            scales,
            width,
            elements,
            bits,
            compute_int8: precision == Precision::Int8Compute,
        }
    }
    fn read(reader: &mut impl Read, bytes: usize, dtype: Dtype) -> Result<Self, String> {
        let stride = match dtype {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            _ => return Err(format!("unsupported weight dtype {dtype:?}")),
        };
        if !bytes.is_multiple_of(stride) {
            return Err("misaligned tensor data".into());
        }
        let mut data = match dtype {
            Dtype::F32 => Self::F32(Vec::with_capacity(bytes / stride)),
            Dtype::BF16 => Self::Bf16(Vec::with_capacity(bytes / stride)),
            _ => Self::F16(Vec::with_capacity(bytes / stride)),
        };
        // A bounded transfer buffer avoids retaining a whole checkpoint or tensor twice.
        let mut buffer = vec![0; bytes.min(1024 * 1024)];
        let mut remaining = bytes;
        while remaining > 0 {
            let count = remaining.min(buffer.len());
            reader
                .read_exact(&mut buffer[..count])
                .map_err(|e| e.to_string())?;
            match &mut data {
                Self::F32(v) => {
                    for c in buffer[..count].as_chunks::<4>().0 {
                        let value = f32::from_le_bytes(*c);
                        if !value.is_finite() {
                            return Err("non-finite weights".into());
                        }
                        v.push(value);
                    }
                }
                Self::Bf16(v) | Self::F16(v) => {
                    let mask = if dtype == Dtype::BF16 { 0x7f80 } else { 0x7c00 };
                    for c in buffer[..count].as_chunks::<2>().0 {
                        let value = u16::from_le_bytes(*c);
                        if value & mask == mask {
                            return Err("non-finite weights".into());
                        }
                        v.push(value);
                    }
                }
                Self::Quantized { .. } => unreachable!("quantization is applied after reading"),
            }
            remaining -= count;
        }
        Ok(data)
    }
}

pub(crate) struct Tensor {
    pub data: Data,
    pub shape: Vec<usize>,
}
pub(crate) struct Weights {
    tensors: HashMap<String, Tensor>,
    precision: Precision,
}
impl Weights {
    pub fn file_bounded(
        path: &Path,
        prefix: Option<&str>,
        budget: &mut crate::loading::WeightBudget,
    ) -> Result<Self, String> {
        let mut w = Self {
            tensors: HashMap::new(),
            precision: Precision::Checkpoint,
        };
        w.extend(path, prefix, budget)?;
        Ok(w)
    }
    pub fn base(
        dir: &Path,
        precision: Precision,
        budget: &mut crate::loading::WeightBudget,
    ) -> Result<Self, String> {
        let mut w = Self {
            tensors: HashMap::new(),
            precision,
        };
        let single = dir.join("model.safetensors");
        if single.exists() {
            w.extend(&single, Some("model.language_model."), budget)?;
        } else {
            let index = crate::json(&dir.join("model.safetensors.index.json"))?;
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
                w.extend(&dir.join(f), Some("model.language_model."), budget)?;
            }
        }
        Ok(w)
    }
    fn extend(
        &mut self,
        path: &Path,
        prefix: Option<&str>,
        budget: &mut crate::loading::WeightBudget,
    ) -> Result<(), String> {
        let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        for (name, tensor) in read_tensors(&mut file, length, prefix, self.precision, budget)
            .map_err(|e| format!("{}: {e}", path.display()))?
        {
            if self.tensors.insert(name.clone(), tensor).is_some() {
                return Err(format!("duplicate tensor {name}"));
            }
        }
        Ok(())
    }
    pub fn take_data(&mut self, name: &str, shape: &[usize]) -> Result<Data, String> {
        let t = self
            .tensors
            .remove(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if t.shape != shape {
            return Err(format!("{name}: expected {shape:?}, found {:?}", t.shape));
        }
        Ok(t.data)
    }
    pub fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        Ok(self.take_data(name, shape)?.into_f32())
    }
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub fn remaining(&self) -> usize {
        self.tensors.len()
    }
}

fn read_tensors(
    reader: &mut (impl Read + Seek),
    length: u64,
    prefix: Option<&str>,
    precision: Precision,
    budget: &mut crate::loading::WeightBudget,
) -> Result<HashMap<String, Tensor>, String> {
    let mut size = [0; 8];
    reader.read_exact(&mut size).map_err(|e| e.to_string())?;
    let header_size = u64::from_le_bytes(size);
    if header_size > 16 * 1024 * 1024 || header_size + 8 > length {
        return Err("invalid or oversized safetensors header".into());
    }
    let mut header = vec![0; header_size as usize];
    reader.read_exact(&mut header).map_err(|e| e.to_string())?;
    // Metadata deserialization validates shapes, ordered offsets and byte counts.
    let metadata: Metadata = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    let start = header_size + 8;
    if start.checked_add(metadata.data_len() as u64) != Some(length) {
        return Err("safetensors file length does not match metadata".into());
    }
    // Inspect every selected shape before reading any tensor in this file.
    for name in metadata.offset_keys() {
        if prefix.is_none_or(|p| name.starts_with(p)) {
            let info = metadata.info(&name).ok_or("missing tensor metadata")?;
            budget.charge(&name, &info.shape)?;
        }
    }
    let mut tensors = HashMap::new();
    for original in metadata.offset_keys() {
        let info = metadata.info(&original).ok_or("missing tensor metadata")?;
        let name = if let Some(p) = prefix {
            match original.strip_prefix(p) {
                Some(n) => n.to_string(),
                None => continue,
            }
        } else {
            original
        };
        reader
            .seek(SeekFrom::Start(start + info.data_offsets.0 as u64))
            .map_err(|e| e.to_string())?;
        let mut data = Data::read(
            reader,
            info.data_offsets.1 - info.data_offsets.0,
            info.dtype,
        )
        .map_err(|e| format!("{name}: {e}"))?;
        if name.starts_with("layers.")
            && name.contains("_proj")
            && name.ends_with(".weight")
            && info.shape.len() == 2
        {
            if info.shape[1] == 0 {
                return Err("zero-width projection".into());
            }
            data = data.quantize(info.shape[1], precision);
        }
        tensors.insert(
            name,
            Tensor {
                data,
                shape: info.shape.clone(),
            },
        );
    }
    Ok(tensors)
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
    w: Data,
    bias: Option<Vec<f32>>,
    pub input: usize,
    pub output: usize,
    lora: Option<(Data, Data, usize, f32)>,
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
            w: w.take_data(&format!("{prefix}.weight"), &[output, input])?,
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
            w.take_data(&a, &[rank, self.input])?,
            w.take_data(&b, &[self.output, rank])?,
            rank,
            scale,
        ));
        Ok(())
    }
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.input;
        let mut out = vec![0.0; rows * self.output];
        self.apply_into(x, &mut out, Epilogue::Write);
        out
    }
    pub fn apply_into(&self, x: &[f32], out: &mut [f32], epilogue: Epilogue) {
        // Keep the measured scalar INT8 fallback's separate execution plan.
        // Native ARM dot products and FP32 modes use the fused output path.
        if matches!(
            self.w,
            Data::Quantized {
                compute_int8: true,
                ..
            }
        ) && !crate::int8::Kernel::detect().accelerated()
        {
            let direct = matches!(epilogue, Epilogue::Write);
            let mut temporary = Vec::new();
            let values = if direct {
                &mut *out
            } else {
                temporary.resize(out.len(), 0.0);
                &mut temporary
            };
            let rows = x.len() / self.input;
            linear(
                x,
                rows,
                self.input,
                &self.w,
                self.output,
                self.bias.as_deref(),
                values,
            );
            if let Some((a, b, rank, scale)) = &self.lora {
                let mut mid = vec![0.0; rows * rank];
                linear(x, rows, self.input, a, *rank, None, &mut mid);
                let mut delta = vec![0.0; values.len()];
                linear(&mid, rows, *rank, b, self.output, None, &mut delta);
                for (value, delta) in values.iter_mut().zip(delta) {
                    *value += scale * delta;
                }
            }
            if !direct {
                for (dst, value) in out.iter_mut().zip(temporary) {
                    epilogue.store(dst, value);
                }
            }
            return;
        }
        let rows = x.len() / self.input;
        let mut mid = Vec::new();
        let adapter = if let Some((a, b, rank, scale)) = &self.lora {
            mid.resize(rows * rank, 0.0);
            linear(x, rows, self.input, a, *rank, None, &mut mid);
            Some(Adapter {
                inputs: &mid,
                weights: b,
                rank: *rank,
                scale: *scale,
            })
        } else {
            None
        };
        Projection {
            input: self.input,
            output: self.output,
            weights: &self.w,
            bias: self.bias.as_deref(),
            adapter,
        }
        .apply(x, out, epilogue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_tensor_metadata_is_rejected_before_payload_read() {
        let header = serde_json::to_vec(&serde_json::json!({
            "w": {"dtype":"F32", "shape":[1_000_000,1_000_000], "data_offsets":[0,4_000_000_000_000u64]}
        })).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(&header);
        let declared_len = bytes.len() as u64 + 4_000_000_000_000;
        let result = read_tensors(
            &mut std::io::Cursor::new(bytes),
            declared_len,
            None,
            Precision::Checkpoint,
            &mut crate::loading::WeightBudget::new(crate::LoadLimits::default()),
        );
        assert!(result.err().unwrap().contains("budget"));
    }

    #[test]
    #[ignore = "synthetic fusion timing; run with --release and fixed RAYON_NUM_THREADS"]
    fn benchmark_fused_projection() {
        use std::{hint::black_box, time::Instant};
        let (input, output, rank) = (2048usize, 6144usize, 16usize);
        let packed = |len| {
            Data::Bf16(
                (0..len)
                    .map(|i| ((((i * 17 % 89) as f32 - 44.0) / 64.0).to_bits() >> 16) as u16)
                    .collect(),
            )
        };
        for precision in [Precision::Checkpoint, Precision::Int8Compute] {
            let layer = Linear {
                w: packed(input * output).quantize(input, precision),
                bias: None,
                input,
                output,
                lora: Some((packed(rank * input), packed(output * rank), rank, 0.5)),
            };
            for rows in [32, 128] {
                let x: Vec<f32> = (0..rows * input)
                    .map(|i| ((i * 43 % 113) as f32 - 56.0) / 127.0)
                    .collect();
                let original_gate: Vec<f32> = (0..rows * output)
                    .map(|i| ((i * 53 % 127) as f32 - 63.0) / 32.0)
                    .collect();
                let mut gate = original_gate.clone();
                for fused in [false, true] {
                    let run = |gate: &mut [f32]| {
                        gate.copy_from_slice(&original_gate);
                        if fused {
                            layer.apply_into(black_box(&x), gate, Epilogue::SwiGlu);
                        } else {
                            let up = unfused_apply(&layer, black_box(&x));
                            for (gate, up) in gate.iter_mut().zip(up) {
                                *gate = crate::qwen::silu(*gate) * up;
                            }
                        }
                        black_box(gate);
                    };
                    run(&mut gate);
                    let start = Instant::now();
                    for _ in 0..3 {
                        run(&mut gate);
                    }
                    println!("fusion precision={precision:?} rows={rows} fused={fused} workers={} mean_ms={:.3}",
                        rayon::current_num_threads(), start.elapsed().as_secs_f64() * 1000.0 / 3.0);
                }
            }
        }
    }

    fn unfused_apply(layer: &Linear, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / layer.input;
        let mut out = vec![0.0; rows * layer.output];
        linear(
            x,
            rows,
            layer.input,
            &layer.w,
            layer.output,
            layer.bias.as_deref(),
            &mut out,
        );
        if let Some((a, b, rank, scale)) = &layer.lora {
            let mut mid = vec![0.0; rows * rank];
            linear(x, rows, layer.input, a, *rank, None, &mut mid);
            let mut delta = vec![0.0; out.len()];
            linear(&mid, rows, *rank, b, layer.output, None, &mut delta);
            for (value, delta) in out.iter_mut().zip(delta) {
                *value += scale * delta;
            }
        }
        out
    }

    #[test]
    fn fused_lora_and_epilogues_match_separate_projections_in_every_precision() {
        let (input, output, rank) = (131usize, 31, 13);
        for rows in [1, 19, 67] {
            let x: Vec<f32> = (0..rows * input)
                .map(|i| ((i * 43 % 113) as f32 - 56.0) / 32.0)
                .collect();
            for precision in [
                Precision::Checkpoint,
                Precision::Int8,
                Precision::Int8Compute,
                Precision::Int4,
            ] {
                let packed = |count: usize| {
                    Data::Bf16(
                        (0..count)
                            .map(|i| {
                                ((((i * 17 % 89) as f32 - 44.0) / 32.0).to_bits() >> 16) as u16
                            })
                            .collect(),
                    )
                };
                let layer = Linear {
                    w: packed(input * output).quantize(input, precision),
                    bias: Some((0..output).map(|i| i as f32 / 64.0 - 0.25).collect()),
                    input,
                    output,
                    lora: Some((packed(rank * input), packed(output * rank), rank, 0.375)),
                };
                let expected = unfused_apply(&layer, &x);
                assert_eq!(layer.apply(&x), expected);
                for epilogue in [Epilogue::Add, Epilogue::SwiGlu] {
                    let original: Vec<f32> =
                        (0..expected.len()).map(|i| i as f32 / 16.0 - 3.0).collect();
                    let separate: Vec<f32> = original
                        .iter()
                        .zip(&expected)
                        .map(|(old, value)| match epilogue {
                            Epilogue::Add => old + value,
                            Epilogue::SwiGlu => crate::qwen::silu(*old) * value,
                            Epilogue::Write => unreachable!(),
                        })
                        .collect();
                    let mut fused = original;
                    layer.apply_into(&x, &mut fused, epilogue);
                    assert_eq!(fused, separate);
                }
            }
        }
    }
    #[test]
    fn half_and_bfloat_readout() {
        assert_eq!(
            Data::read(&mut &[0x80, 0x3f, 0x00, 0xc0][..], 4, Dtype::BF16)
                .unwrap()
                .into_f32(),
            vec![1.0, -2.0]
        );
        assert_eq!(half(0x3c00), 1.0);
        assert_eq!(half(0xc000), -2.0);
        assert_eq!(half(1), 2f32.powi(-24));
        assert!(Data::read(&mut &[0][..], 1, Dtype::I8).is_err());
    }

    #[test]
    fn packed_projections_match_fp32_without_quantization() {
        let (rows, input, output) = (19, 13, 31);
        let x: Vec<_> = (0..rows * input).map(|i| (i % 17) as f32 / 7.0).collect();
        for dtype in [Dtype::BF16, Dtype::F16] {
            let bytes: Vec<_> = (0..input * output)
                .flat_map(|i| (0x3800 + (i % 31) as u16).to_le_bytes())
                .collect();
            let packed = Data::read(&mut bytes.as_slice(), bytes.len(), dtype).unwrap();
            let expanded = Data::F32(
                Data::read(&mut bytes.as_slice(), bytes.len(), dtype)
                    .unwrap()
                    .into_f32(),
            );
            let mut a = vec![0.0; rows * output];
            let mut b = a.clone();
            linear(&x, rows, input, &packed, output, None, &mut a);
            linear(&x, rows, input, &expanded, output, None, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn transfer_buffer_is_bounded_and_bf16_is_retained() {
        struct Reader {
            inner: std::io::Cursor<Vec<u8>>,
            max: usize,
        }
        impl Read for Reader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                self.max = self.max.max(buffer.len());
                self.inner.read(buffer)
            }
        }
        let size = 2 * 1024 * 1024 + 2;
        let mut reader = Reader {
            inner: std::io::Cursor::new(vec![0; size]),
            max: 0,
        };
        let data = Data::read(&mut reader, size, Dtype::BF16).unwrap();
        assert!(matches!(data, Data::Bf16(v) if v.len() == size / 2));
        assert!(reader.max <= 1024 * 1024);
    }

    #[test]
    fn streaming_metadata_rejects_invalid_offsets_lengths_and_values() {
        fn load(header: serde_json::Value, data: &[u8]) -> Result<HashMap<String, Tensor>, String> {
            let header = serde_json::to_vec(&header).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&header);
            bytes.extend_from_slice(data);
            let len = bytes.len() as u64;
            read_tensors(
                &mut std::io::Cursor::new(bytes),
                len,
                Some("text."),
                Precision::Checkpoint,
                &mut crate::loading::WeightBudget::new(crate::LoadLimits::default()),
            )
        }
        let valid = serde_json::json!({"text.w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"vision.w":{"dtype":"I8","shape":[1],"data_offsets":[2,3]}});
        let mut tensors = load(valid.clone(), &[0x80, 0x3f, 0]).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors.remove("w").unwrap().data.into_f32(), vec![1.0]);
        assert!(load(valid.clone(), &[0x80, 0x3f]).is_err());
        assert!(load(valid.clone(), &[0x80, 0x3f, 0, 0]).is_err());
        assert!(load(valid, &[0x80, 0x7f, 0]).is_err());
        assert!(load(
            serde_json::json!({"text.w":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}),
            &[0, 0, 0]
        )
        .is_err());
        let mut oversized = std::io::Cursor::new(u64::MAX.to_le_bytes());
        assert!(read_tensors(
            &mut oversized,
            u64::MAX,
            None,
            Precision::Checkpoint,
            &mut crate::loading::WeightBudget::new(crate::LoadLimits::default())
        )
        .is_err());
        for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16] {
            assert!(Data::read(&mut &[0][..], 2, dtype).is_err());
        }
    }

    #[test]
    fn quantization_bounds_error_and_handles_odd_rows_partial_groups_and_zeroes() {
        let width = 131;
        let weights: Vec<_> = (0..width * 3)
            .map(|i| {
                if i < width {
                    0.0
                } else {
                    ((i % 113) as f32 - 56.0) / 19.0
                }
            })
            .collect();
        for (precision, bound) in [(Precision::Int8, 127.0), (Precision::Int4, 7.0)] {
            let data = Data::F32(weights.clone()).quantize(width, precision);
            assert_eq!(data.len(), weights.len());
            let restored = data.into_f32();
            for (a, b) in weights.chunks(width).zip(restored.chunks(width)) {
                for (a, b) in a.chunks(64).zip(b.chunks(64)) {
                    let max = a.iter().map(|v| v.abs()).fold(0.0, f32::max);
                    for (a, b) in a.iter().zip(b) {
                        assert!((a - b).abs() <= max / (2.0 * bound) + 1e-6);
                    }
                }
            }
        }
    }
}
