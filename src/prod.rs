use rayon::prelude::*;
use std::f64::consts::PI;

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::{Result, TurboQuantError};
use crate::lloyd_max::LloydMaxOptions;
use crate::math::{dot, l2_norm, normalize, subtract, validate_dimension};
use crate::mse::TurboQuantMse;
use crate::packed::PackedBits;
use crate::rotation::{
    Rotation, RotationBackend, apply_matrix, apply_matrix_transpose, gaussian_matrix,
};

#[derive(Debug, Clone)]
pub struct ProdCode {
    input_norm: f64,
    mse_indices: PackedBits,
    qjl_signs: PackedBits,
    residual_norm: f64,
}

#[derive(Debug, Clone)]
pub struct TurboQuantProd {
    dimension: usize,
    bit_width: u8,
    mse: TurboQuantMse,
    sketch: Sketch,
}

#[derive(Debug, Clone)]
enum Sketch {
    DenseGaussian(DMatrix<f64>),
    Structured(Rotation),
}

impl TurboQuantProd {
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
        if bit_width == 0 {
            return Err(TurboQuantError::ProdBitWidthTooSmall);
        }
        if dimension < 2 {
            return Err(TurboQuantError::InvalidDimension(dimension));
        }

        let mse = TurboQuantMse::new_with_options_and_rotation_backend(
            dimension,
            bit_width.saturating_sub(1),
            seed ^ 0x9E37_79B9_7F4A_7C15,
            options,
            rotation_backend,
        )?;
        let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_D192_ED03);
        let sketch = match rotation_backend {
            RotationBackend::DenseGaussian => {
                Sketch::DenseGaussian(gaussian_matrix(dimension, &mut rng))
            }
            RotationBackend::WalshHadamard => Sketch::Structured(Rotation::new_with_backend(
                dimension,
                &mut rng,
                RotationBackend::WalshHadamard,
            )),
        };

        Ok(Self {
            dimension,
            bit_width,
            mse,
            sketch,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn rotation_backend(&self) -> RotationBackend {
        self.mse.rotation_backend()
    }

    pub fn quantize_batch<T>(&self, vectors: &[T]) -> Result<Vec<ProdCode>>
    where
        T: AsRef<[f64]> + Sync,
    {
        let results = vectors
            .par_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn dequantize_batch(&self, codes: &[ProdCode]) -> Result<Vec<Vec<f64>>> {
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
    ) -> Result<Vec<ProdCode>> {
        let expected = row_count * self.dimension;
        if values.len() != expected {
            return Err(TurboQuantError::TensorElementCountMismatch {
                expected,
                actual: values.len(),
            });
        }

        let mut input_norms = vec![0.0; row_count];
        let mut normalized = vec![0.0; values.len()];
        for row_index in 0..row_count {
            let start = row_index * self.dimension;
            let end = start + self.dimension;
            let (norm, unit) = normalize(&values[start..end]);
            input_norms[row_index] = norm;
            normalized[start..end].copy_from_slice(&unit);
        }

        let mse_indices = self.mse.quantize_unit_flat_batch(&normalized, row_count)?;
        let mse_index_refs = mse_indices.iter().collect::<Vec<_>>();
        let mse_decoded = self
            .mse
            .dequantize_unit_flat_batch_from_indices(&mse_index_refs)?;

        let mut residual = vec![0.0; values.len()];
        let mut residual_norms = vec![0.0; row_count];
        for row_index in 0..row_count {
            let start = row_index * self.dimension;
            let end = start + self.dimension;
            for slot in 0..self.dimension {
                residual[start + slot] = normalized[start + slot] - mse_decoded[start + slot];
            }
            residual_norms[row_index] = l2_norm(&residual[start..end]);
        }

        let projected = self
            .sketch
            .apply_batch_flat(&residual, row_count, self.dimension);
        let mut codes = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let start = row_index * self.dimension;
            let end = start + self.dimension;
            let qjl_signs = if residual_norms[row_index] == 0.0 {
                PackedBits::pack_values(&vec![1; self.dimension], 1)?
            } else {
                let signs = projected[start..end]
                    .iter()
                    .map(|&value| if value >= 0.0 { 1_i8 } else { -1_i8 })
                    .collect::<Vec<_>>();
                PackedBits::pack_signs(&signs)?
            };
            codes.push(ProdCode {
                input_norm: input_norms[row_index],
                mse_indices: mse_indices[row_index].clone(),
                qjl_signs,
                residual_norm: residual_norms[row_index],
            });
        }

        Ok(codes)
    }

    pub(crate) fn dequantize_flat_batch(&self, codes: &[ProdCode]) -> Result<Vec<f64>> {
        let mse_indices = codes
            .iter()
            .map(|code| code.mse_indices())
            .collect::<Vec<_>>();
        let mse_decoded = self
            .mse
            .dequantize_unit_flat_batch_from_indices(&mse_indices)?;

        let mut signs_flat = vec![0.0; codes.len() * self.dimension];
        for (row_index, code) in codes.iter().enumerate() {
            if code.qjl_signs().symbol_count() != self.dimension
                || code.qjl_signs().bits_per_symbol() != 1
            {
                return Err(TurboQuantError::PackedLayoutMismatch {
                    expected_symbols: self.dimension,
                    actual_symbols: code.qjl_signs().symbol_count(),
                    expected_bits_per_symbol: 1,
                    actual_bits_per_symbol: code.qjl_signs().bits_per_symbol(),
                });
            }
            let row_start = row_index * self.dimension;
            for slot in 0..self.dimension {
                signs_flat[row_start + slot] = if code.qjl_signs().get(slot)? == 0 {
                    -1.0
                } else {
                    1.0
                };
            }
        }

        let projected =
            self.sketch
                .apply_transpose_batch_flat(&signs_flat, codes.len(), self.dimension);
        let mut decoded = vec![0.0; codes.len() * self.dimension];
        let qjl_scale = (PI / 2.0).sqrt() / self.dimension as f64;
        for (row_index, code) in codes.iter().enumerate() {
            let row_start = row_index * self.dimension;
            let row_end = row_start + self.dimension;
            let residual_scale = qjl_scale * code.residual_norm();
            for slot in 0..self.dimension {
                decoded[row_start + slot] = code.input_norm()
                    * (mse_decoded[row_start + slot]
                        + residual_scale * projected[row_start + slot]);
            }
            if code.input_norm() == 0.0 {
                decoded[row_start..row_end].fill(0.0);
            }
        }

        Ok(decoded)
    }

    pub fn estimate_inner_products_batch<T>(
        &self,
        codes: &[ProdCode],
        queries: &[T],
    ) -> Result<Vec<f64>>
    where
        T: AsRef<[f64]> + Sync,
    {
        if codes.len() != queries.len() {
            return Err(TurboQuantError::BatchLengthMismatch {
                expected: codes.len(),
                actual: queries.len(),
            });
        }

        let results = codes
            .par_iter()
            .zip(queries.par_iter())
            .map(|(code, query)| self.estimate_inner_product(code, query.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn quantize(&self, vector: &[f64]) -> Result<ProdCode> {
        validate_dimension(self.dimension, vector.len())?;
        let (input_norm, unit_vector) = normalize(vector);

        if input_norm == 0.0 {
            return Ok(ProdCode {
                input_norm,
                mse_indices: PackedBits::zeros(self.dimension, self.bit_width.saturating_sub(1))?,
                qjl_signs: PackedBits::pack_values(&vec![1; self.dimension], 1)?,
                residual_norm: 0.0,
            });
        }

        let mse_indices = self.mse.quantize_unit(&unit_vector);
        let mse_decoded = self.mse.dequantize_unit(&mse_indices)?;
        let residual = subtract(&unit_vector, &mse_decoded);
        let residual_norm = l2_norm(&residual);

        let qjl_signs = if residual_norm == 0.0 {
            PackedBits::pack_values(&vec![1; self.dimension], 1)?
        } else {
            let signs = self
                .sketch
                .apply(&residual)
                .into_iter()
                .map(|value| if value >= 0.0 { 1_i8 } else { -1_i8 })
                .collect::<Vec<_>>();
            PackedBits::pack_signs(&signs)?
        };

        Ok(ProdCode {
            input_norm,
            mse_indices,
            qjl_signs,
            residual_norm,
        })
    }

    pub fn dequantize(&self, code: &ProdCode) -> Result<Vec<f64>> {
        if code.mse_indices.symbol_count() != self.dimension
            || code.mse_indices.bits_per_symbol() != self.bit_width.saturating_sub(1)
        {
            return Err(TurboQuantError::PackedLayoutMismatch {
                expected_symbols: self.dimension,
                actual_symbols: code.mse_indices.symbol_count(),
                expected_bits_per_symbol: self.bit_width.saturating_sub(1),
                actual_bits_per_symbol: code.mse_indices.bits_per_symbol(),
            });
        }
        if code.qjl_signs.symbol_count() != self.dimension || code.qjl_signs.bits_per_symbol() != 1
        {
            return Err(TurboQuantError::PackedLayoutMismatch {
                expected_symbols: self.dimension,
                actual_symbols: code.qjl_signs.symbol_count(),
                expected_bits_per_symbol: 1,
                actual_bits_per_symbol: code.qjl_signs.bits_per_symbol(),
            });
        }
        if code.input_norm() == 0.0 {
            return Ok(vec![0.0; self.dimension]);
        }

        let mse_decoded = self.mse.dequantize_unit(code.mse_indices())?;
        let qjl_component = self.qjl_dequantize(code.qjl_signs(), code.residual_norm());
        let decoded = mse_decoded
            .into_iter()
            .zip(qjl_component)
            .map(|(mse, qjl)| code.input_norm() * (mse + qjl))
            .collect();

        Ok(decoded)
    }

    pub fn estimate_inner_product(&self, code: &ProdCode, query: &[f64]) -> Result<f64> {
        validate_dimension(self.dimension, query.len())?;
        let decoded = self.dequantize(code)?;
        Ok(dot(query, &decoded))
    }

    fn qjl_dequantize(&self, signs: &PackedBits, residual_norm: f64) -> Vec<f64> {
        if residual_norm == 0.0 {
            return vec![0.0; self.dimension];
        }

        let sign_vector = signs
            .unpack_signs()
            .expect("validated packed QJL signs")
            .into_iter()
            .map(|value| if value >= 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let projected = self.sketch.apply_transpose(&sign_vector);
        let scale = (PI / 2.0).sqrt() * residual_norm / self.dimension as f64;

        projected.into_iter().map(|value| value * scale).collect()
    }
}

impl Sketch {
    fn apply(&self, vector: &[f64]) -> Vec<f64> {
        match self {
            Sketch::DenseGaussian(matrix) => apply_matrix(matrix, vector),
            Sketch::Structured(rotation) => rotation.apply(vector),
        }
    }

    fn apply_transpose(&self, vector: &[f64]) -> Vec<f64> {
        match self {
            Sketch::DenseGaussian(matrix) => apply_matrix_transpose(matrix, vector),
            Sketch::Structured(rotation) => rotation.apply_transpose(vector),
        }
    }

    fn apply_batch_flat(&self, values: &[f64], row_count: usize, dimension: usize) -> Vec<f64> {
        match self {
            Sketch::DenseGaussian(matrix) => values
                .par_chunks(dimension)
                .flat_map_iter(|row| apply_matrix(matrix, row))
                .collect(),
            Sketch::Structured(rotation) => rotation.apply_batch_flat(values, row_count, dimension),
        }
    }

    fn apply_transpose_batch_flat(
        &self,
        values: &[f64],
        row_count: usize,
        dimension: usize,
    ) -> Vec<f64> {
        match self {
            Sketch::DenseGaussian(matrix) => values
                .par_chunks(dimension)
                .flat_map_iter(|row| apply_matrix_transpose(matrix, row))
                .collect(),
            Sketch::Structured(rotation) => {
                rotation.apply_transpose_batch_flat(values, row_count, dimension)
            }
        }
    }
}

impl ProdCode {
    pub fn input_norm(&self) -> f64 {
        self.input_norm
    }

    pub fn residual_norm(&self) -> f64 {
        self.residual_norm
    }

    pub fn mse_indices(&self) -> &PackedBits {
        &self.mse_indices
    }

    pub fn qjl_signs(&self) -> &PackedBits {
        &self.qjl_signs
    }

    pub fn unpack_mse_indices(&self) -> Result<Vec<u64>> {
        self.mse_indices.unpack_values()
    }

    pub fn unpack_qjl_signs(&self) -> Result<Vec<i8>> {
        self.qjl_signs.unpack_signs()
    }

    pub fn storage_bytes(&self) -> usize {
        self.mse_indices.storage_bytes()
            + self.qjl_signs.storage_bytes()
            + 2 * std::mem::size_of::<f64>()
    }
}
