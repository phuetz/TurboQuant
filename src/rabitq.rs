use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::error::{Result, TurboQuantError};
use crate::math::{dot, validate_dimension};
use crate::packed::PackedBits;
use crate::rotation::{Rotation, RotationBackend};

const DEFAULT_SCALING_SEARCH_STEPS: usize = 96;

#[derive(Debug, Clone)]
pub struct RaBitQCode {
    delta: f64,
    indices: PackedBits,
}

#[derive(Debug, Clone)]
pub struct RaBitQQuantizer {
    dimension: usize,
    bit_width: u8,
    center: Vec<f64>,
    rotation: Rotation,
    scaling_search_steps: usize,
}

impl RaBitQQuantizer {
    pub fn new(dimension: usize, bit_width: u8, seed: u64) -> Result<Self> {
        Self::with_center(vec![0.0; dimension], bit_width, seed)
    }

    pub fn with_center(center: Vec<f64>, bit_width: u8, seed: u64) -> Result<Self> {
        let dimension = center.len();
        if dimension < 2 {
            return Err(TurboQuantError::InvalidDimension(dimension));
        }
        if bit_width == 0 || bit_width > 20 {
            return Err(TurboQuantError::UnsupportedBitWidth(bit_width));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let rotation = Rotation::new(dimension, &mut rng);

        Ok(Self {
            dimension,
            bit_width,
            center,
            rotation,
            scaling_search_steps: DEFAULT_SCALING_SEARCH_STEPS,
        })
    }

    pub fn train(dataset: &[Vec<f64>], bit_width: u8, seed: u64) -> Result<Self> {
        let center = dataset_mean(dataset)?;
        Self::with_center(center, bit_width, seed)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn center(&self) -> &[f64] {
        &self.center
    }

    pub fn rotation_backend(&self) -> RotationBackend {
        self.rotation.backend()
    }

    pub fn encoded_bits_per_vector(&self) -> usize {
        self.dimension * self.bit_width as usize
    }

    pub fn quantize(&self, vector: &[f64]) -> Result<RaBitQCode> {
        validate_dimension(self.dimension, vector.len())?;
        let centered = vector
            .iter()
            .zip(&self.center)
            .map(|(value, center)| value - center)
            .collect::<Vec<_>>();
        let rotated = self.rotation.apply(&centered);
        let (delta, codes) = scalar_quantize(&rotated, self.bit_width, self.scaling_search_steps);

        Ok(RaBitQCode {
            delta,
            indices: PackedBits::pack_values(&codes, self.bit_width)?,
        })
    }

    pub fn dequantize(&self, code: &RaBitQCode) -> Result<Vec<f64>> {
        if code.indices.symbol_count() != self.dimension
            || code.indices.bits_per_symbol() != self.bit_width
        {
            return Err(TurboQuantError::PackedLayoutMismatch {
                expected_symbols: self.dimension,
                actual_symbols: code.indices.symbol_count(),
                expected_bits_per_symbol: self.bit_width,
                actual_bits_per_symbol: code.indices.bits_per_symbol(),
            });
        }

        let rotated = dequantize_levels(&code.indices, code.delta)?;
        let centered = self.rotation.apply_transpose(&rotated);
        Ok(centered
            .into_iter()
            .zip(&self.center)
            .map(|(value, center)| value + center)
            .collect())
    }

    pub fn quantize_batch<T>(&self, vectors: &[T]) -> Result<Vec<RaBitQCode>>
    where
        T: AsRef<[f64]> + Sync,
    {
        let results = vectors
            .par_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn dequantize_batch(&self, codes: &[RaBitQCode]) -> Result<Vec<Vec<f64>>> {
        let results = codes
            .par_iter()
            .map(|code| self.dequantize(code))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn approximate_inner_product(&self, code: &RaBitQCode, query: &[f64]) -> Result<f64> {
        validate_dimension(self.dimension, query.len())?;
        let decoded = self.dequantize(code)?;
        Ok(dot(&decoded, query))
    }
}

impl RaBitQCode {
    pub fn delta(&self) -> f64 {
        self.delta
    }

    pub fn indices(&self) -> &PackedBits {
        &self.indices
    }

    pub fn unpack_indices(&self) -> Result<Vec<u64>> {
        self.indices.unpack_values()
    }

    pub fn storage_bytes(&self) -> usize {
        self.indices.storage_bytes() + std::mem::size_of::<f64>()
    }
}

fn dataset_mean(dataset: &[Vec<f64>]) -> Result<Vec<f64>> {
    let first = dataset.first().ok_or(TurboQuantError::EmptyDataset)?;
    let dimension = first.len();
    if dimension < 2 {
        return Err(TurboQuantError::InvalidDimension(dimension));
    }

    let mut sums = vec![0.0; dimension];
    for (row_index, row) in dataset.iter().enumerate() {
        if row.len() != dimension {
            return Err(TurboQuantError::InconsistentRowDimension {
                line: row_index + 1,
                expected: dimension,
                actual: row.len(),
            });
        }
        for (slot, &value) in row.iter().enumerate() {
            sums[slot] += value;
        }
    }

    let scale = 1.0 / dataset.len() as f64;
    for value in &mut sums {
        *value *= scale;
    }
    Ok(sums)
}

fn scalar_quantize(vector: &[f64], bit_width: u8, search_steps: usize) -> (f64, Vec<u64>) {
    let level_count = 1_u64 << bit_width;
    let shift = (level_count as f64 - 1.0) / 2.0;
    let max_abs = vector
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);

    if max_abs == 0.0 {
        return (0.0, vec![shift.round() as u64; vector.len()]);
    }

    let min_delta = (max_abs / (4.0 * level_count as f64)).max(1e-12);
    let max_delta = (2.0 * max_abs).max(min_delta);

    let mut best_delta = max_delta;
    let mut best_codes = quantize_with_delta(vector, best_delta, shift, level_count - 1);
    let mut best_error = quantization_error(vector, &best_codes, best_delta, shift);

    let total_steps = search_steps.max(2);
    for step in 0..total_steps {
        let ratio = step as f64 / (total_steps - 1) as f64;
        let delta = min_delta * (max_delta / min_delta).powf(ratio);
        let codes = quantize_with_delta(vector, delta, shift, level_count - 1);
        let error = quantization_error(vector, &codes, delta, shift);
        if error < best_error {
            best_error = error;
            best_delta = delta;
            best_codes = codes;
        }
    }

    (best_delta, best_codes)
}

fn quantize_with_delta(vector: &[f64], delta: f64, shift: f64, max_level: u64) -> Vec<u64> {
    vector
        .iter()
        .map(|&value| {
            ((value / delta) + shift)
                .round()
                .clamp(0.0, max_level as f64) as u64
        })
        .collect()
}

fn quantization_error(vector: &[f64], codes: &[u64], delta: f64, shift: f64) -> f64 {
    vector
        .iter()
        .zip(codes)
        .map(|(&value, &code)| {
            let reconstructed = delta * (code as f64 - shift);
            let residual = value - reconstructed;
            residual * residual
        })
        .sum()
}

fn dequantize_levels(indices: &PackedBits, delta: f64) -> Result<Vec<f64>> {
    let shift = ((1_u64 << indices.bits_per_symbol()) as f64 - 1.0) / 2.0;
    let mut decoded = Vec::with_capacity(indices.symbol_count());
    for slot in 0..indices.symbol_count() {
        decoded.push(delta * (indices.get(slot)? as f64 - shift));
    }
    Ok(decoded)
}
