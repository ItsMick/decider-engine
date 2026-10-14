//! Checked allocation arithmetic at the checkpoint boundary.

/// Conservative allocation budgets, independent of available system RAM.
/// Estimates use FP32 for all weights, include LoRA and loading conversion
/// scratch, and intentionally overestimate packed checkpoints. They are not an
/// RSS guarantee: tokenizer, allocator, thread stacks, and concurrent calls are
/// additional. Lower these budgets for constrained hosts.
#[derive(Clone, Copy, Debug)]
pub struct LoadLimits {
    pub max_tensor_bytes: usize,
    pub max_weight_bytes: usize,
    pub max_workspace_bytes: usize,
}

impl Default for LoadLimits {
    fn default() -> Self {
        let fit = |n: u64| n.min(isize::MAX as u64) as usize;
        Self {
            max_tensor_bytes: fit(2 << 30),
            max_weight_bytes: fit(12 << 30),
            max_workspace_bytes: fit(2 << 30),
        }
    }
}

pub(crate) fn product(name: &str, dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d)
            .ok_or_else(|| format!("{name}: dimension product overflow"))
    })
}

pub(crate) fn sum(name: &str, values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(0usize, |n, v| {
        n.checked_add(*v)
            .ok_or_else(|| format!("{name}: dimension sum overflow"))
    })
}

pub(crate) fn within(name: &str, bytes: usize, limit: usize) -> Result<(), String> {
    if bytes > limit || bytes > isize::MAX as usize {
        Err(format!(
            "{name}: allocation of {bytes} bytes exceeds budget {limit}"
        ))
    } else {
        Ok(())
    }
}

/// Shared across head, adapter, and every base shard. Charge before reading
/// payloads, including unsupported extra tensors, to bound malformed files too.
pub(crate) struct WeightBudget {
    limits: LoadLimits,
    used: usize,
}

impl WeightBudget {
    pub fn new(limits: LoadLimits) -> Self {
        Self { limits, used: 0 }
    }
    pub fn charge(&mut self, name: &str, shape: &[usize]) -> Result<(), String> {
        let mut bytes = product(name, &[product(name, shape)?, 4])?;
        // Include possible group scales even for narrow (e.g. width-one)
        // projections where quantized storage can exceed FP32 storage.
        if let [rows, width] = shape {
            bytes = sum(
                name,
                &[bytes, product(name, &[*rows, width.div_ceil(64), 4])?],
            )?;
        }
        within(name, bytes, self.limits.max_tensor_bytes)?;
        // Covers an original tensor plus quantization/conversion scratch.
        self.used = sum("checkpoint weights", &[self.used, bytes])?;
        within(
            "checkpoint weights and conversion scratch",
            sum(name, &[self.used, bytes])?,
            self.limits.max_weight_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overflow_and_aggregate_budgets_fail_without_allocation() {
        assert!(product("test", &[usize::MAX, 2])
            .unwrap_err()
            .contains("overflow"));
        assert!(sum("test", &[usize::MAX, 1])
            .unwrap_err()
            .contains("overflow"));
        let mut budget = WeightBudget::new(LoadLimits {
            max_tensor_bytes: 16,
            max_weight_bytes: 32,
            max_workspace_bytes: 32,
        });
        budget.charge("first", &[4]).unwrap();
        assert!(budget
            .charge("second", &[4])
            .unwrap_err()
            .contains("budget"));
    }
}
