use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::error::{Result, TurboQuantError};
use crate::lloyd_max::{LloydMaxOptions, ScalarCodebook};
use crate::math::{dot, l2_norm, normalize, squared_l2_distance, validate_dimension};
use crate::packed::PackedBits;
use crate::rotation::{Rotation, RotationBackend};

#[derive(Debug, Clone)]
pub struct MseCode {
    input_norm: f64,
    indices: PackedBits,
}

#[derive(Debug, Clone)]
pub struct TurboQuantMse {
    dimension: usize,
    bit_width: u8,
    rotation: Rotation,
    codebook: ScalarCodebook,
}

impl TurboQuantMse {
    pub fn new(dimension: usize, bit_width: u8, seed: u64) -> Result<Self> {
        Self::new_with_options(dimension, bit_width, seed, LloydMaxOptions::default())
    }

    pub fn new_with_rotation_backend(
        dimension: usize,
        bit_width: u8,
        seed: u64,
        rotation_backend: RotationBackend,
    ) -> Result<Self> {
        Self::new_with_options_and_rotation_backend(
            dimension,
            bit_width,
            seed,
            LloydMaxOptions::default(),
            rotation_backend,
        )
    }

    pub fn new_with_options(
        dimension: usize,
        bit_width: u8,
        seed: u64,
        options: LloydMaxOptions,
    ) -> Result<Self> {
        Self::new_with_options_and_rotation_backend(
            dimension,
            bit_width,
            seed,
            options,
            RotationBackend::DenseGaussian,
        )
    }

    pub fn new_with_options_and_rotation_backend(
        dimension: usize,
        bit_width: u8,
        seed: u64,
        options: LloydMaxOptions,
        rotation_backend: RotationBackend,
    ) -> Result<Self> {
        if dimension < 2 {
            return Err(TurboQuantError::InvalidDimension(dimension));
        }
        if bit_width > 20 {
            return Err(TurboQuantError::UnsupportedBitWidth(bit_width));
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let rotation = Rotation::new_with_backend(dimension, &mut rng, rotation_backend);
        let codebook = ScalarCodebook::solve(dimension, bit_width, options)?;

        Ok(Self {
            dimension,
            bit_width,
            rotation,
            codebook,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn rotation_backend(&self) -> RotationBackend {
        self.rotation.backend()
    }

    pub fn codebook(&self) -> &[f64] {
        self.codebook.centroids()
    }

    pub fn boundaries(&self) -> &[f64] {
        self.codebook.boundaries()
    }

    pub fn encoded_bits_per_vector(&self) -> usize {
        self.dimension * self.bit_width as usize
    }

    pub fn quantize(&self, vector: &[f64]) -> Result<MseCode> {
        validate_dimension(self.dimension, vector.len())?;
        let (input_norm, unit_vector) = normalize(vector);

        if input_norm == 0.0 {
            return Ok(MseCode {
                input_norm,
                indices: PackedBits::zeros(self.dimension, self.bit_width)?,
            });
        }

        let indices = self.quantize_unit(&unit_vector);
        Ok(MseCode {
            input_norm,
            indices,
        })
    }

    pub fn dequantize(&self, code: &MseCode) -> Result<Vec<f64>> {
        if code.input_norm() == 0.0 {
            return Ok(vec![0.0; self.dimension]);
        }

        let decoded = self.dequantize_unit(code.indices())?;
        Ok(decoded
            .into_iter()
            .map(|value| value * code.input_norm())
            .collect())
    }

    pub fn quantize_batch<T>(&self, vectors: &[T]) -> Result<Vec<MseCode>>
    where
        T: AsRef<[f64]> + Sync,
    {
        let results = vectors
            .par_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn dequantize_batch(&self, codes: &[MseCode]) -> Result<Vec<Vec<f64>>> {
        let results = codes
            .par_iter()
            .map(|code| self.dequantize(code))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub(crate) fn quantize_flat_batch(
        &self,
        values: &[f64],
        row_count: usize,
    ) -> Result<Vec<MseCode>> {
        let expected = row_count * self.dimension;
        if values.len() != expected {
            return Err(TurboQuantError::TensorElementCountMismatch {
                expected,
                actual: values.len(),
            });
        }

        let mut norms = vec![0.0; row_count];
        let mut normalized = values.to_vec();
        for row_index in 0..row_count {
            let start = row_index * self.dimension;
            let end = start + self.dimension;
            let norm = l2_norm(&normalized[start..end]);
            norms[row_index] = norm;
            if norm != 0.0 {
                let inv_norm = 1.0 / norm;
                for value in &mut normalized[start..end] {
                    *value *= inv_norm;
                }
            }
        }

        let indices_batch = self.quantize_unit_flat_batch(&normalized, row_count)?;
        let mut codes = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            codes.push(MseCode {
                input_norm: norms[row_index],
                indices: indices_batch[row_index].clone(),
            });
        }
        Ok(codes)
    }

    pub(crate) fn dequantize_flat_batch(&self, codes: &[MseCode]) -> Result<Vec<f64>> {
        let rotated = self.dequantize_unit_flat_batch(codes)?;
        let mut decoded =
            self.rotation
                .apply_transpose_batch_flat(&rotated, codes.len(), self.dimension);

        for (row_index, code) in codes.iter().enumerate() {
            let start = row_index * self.dimension;
            let end = start + self.dimension;
            for value in &mut decoded[start..end] {
                *value *= code.input_norm();
            }
        }
        Ok(decoded)
    }

    pub fn reconstruction_error(&self, vector: &[f64], code: &MseCode) -> Result<f64> {
        validate_dimension(self.dimension, vector.len())?;
        let decoded = self.dequantize(code)?;
        Ok(squared_l2_distance(vector, &decoded))
    }

    pub fn approximate_inner_product(&self, code: &MseCode, query: &[f64]) -> Result<f64> {
        validate_dimension(self.dimension, query.len())?;
        let decoded = self.dequantize(code)?;
        Ok(dot(query, &decoded))
    }

    pub(crate) fn quantize_unit(&self, unit_vector: &[f64]) -> PackedBits {
        let rotated = self.rotation.apply(unit_vector);
        let indices = rotated
            .iter()
            .map(|&coordinate| self.codebook.find_index(coordinate) as u64)
            .collect::<Vec<_>>();
        PackedBits::pack_values(&indices, self.bit_width).expect("validated bit width")
    }

    pub(crate) fn quantize_unit_flat_batch(
        &self,
        values: &[f64],
        row_count: usize,
    ) -> Result<Vec<PackedBits>> {
        let expected = row_count * self.dimension;
        if values.len() != expected {
            return Err(TurboQuantError::TensorElementCountMismatch {
                expected,
                actual: values.len(),
            });
        }

        let rotated = self
            .rotation
            .apply_batch_flat(values, row_count, self.dimension);
        let mut indices_batch = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let start = row_index * self.dimension;
            let end = start + self.dimension;
            let indices = rotated[start..end]
                .iter()
                .map(|&coordinate| self.codebook.find_index(coordinate) as u64)
                .collect::<Vec<_>>();
            indices_batch.push(PackedBits::pack_values(&indices, self.bit_width)?);
        }
        Ok(indices_batch)
    }

    pub(crate) fn dequantize_unit(&self, indices: &PackedBits) -> Result<Vec<f64>> {
        if indices.symbol_count() != self.dimension || indices.bits_per_symbol() != self.bit_width {
            return Err(TurboQuantError::PackedLayoutMismatch {
                expected_symbols: self.dimension,
                actual_symbols: indices.symbol_count(),
                expected_bits_per_symbol: self.bit_width,
                actual_bits_per_symbol: indices.bits_per_symbol(),
            });
        }

        let mut rotated = Vec::with_capacity(self.dimension);
        for slot in 0..self.dimension {
            rotated.push(self.codebook.centroid(indices.get(slot)? as usize)?);
        }

        Ok(self.rotation.apply_transpose(&rotated))
    }

    pub(crate) fn dequantize_unit_flat_batch(&self, codes: &[MseCode]) -> Result<Vec<f64>> {
        let indices = codes.iter().map(|code| code.indices()).collect::<Vec<_>>();
        self.dequantize_unit_flat_batch_from_indices(&indices)
    }

    pub(crate) fn dequantize_unit_flat_batch_from_indices(
        &self,
        indices_batch: &[&PackedBits],
    ) -> Result<Vec<f64>> {
        let mut rotated = vec![0.0; indices_batch.len() * self.dimension];
        let centroids = self.codebook.centroids();
        let mut idx_buf = vec![0u64; self.dimension];
        for (row_index, indices) in indices_batch.iter().enumerate() {
            if indices.symbol_count() != self.dimension
                || indices.bits_per_symbol() != self.bit_width
            {
                return Err(TurboQuantError::PackedLayoutMismatch {
                    expected_symbols: self.dimension,
                    actual_symbols: indices.symbol_count(),
                    expected_bits_per_symbol: self.bit_width,
                    actual_bits_per_symbol: indices.bits_per_symbol(),
                });
            }
            indices.unpack_values_into(&mut idx_buf)?;
            let row_start = row_index * self.dimension;
            for slot in 0..self.dimension {
                rotated[row_start + slot] = centroids[idx_buf[slot] as usize];
            }
        }
        Ok(rotated)
    }

    #[cfg(test)]
    pub(crate) fn rotate_then_unrotate(&self, vector: &[f64]) -> Result<Vec<f64>> {
        validate_dimension(self.dimension, vector.len())?;
        let rotated = self.rotation.apply(vector);
        Ok(self.rotation.apply_transpose(&rotated))
    }
}

impl MseCode {
    pub fn input_norm(&self) -> f64 {
        self.input_norm
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
