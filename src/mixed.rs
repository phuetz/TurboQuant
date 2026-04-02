use std::cmp::Ordering;

use rayon::prelude::*;

use crate::error::{Result, TurboQuantError};
use crate::math::{squared_l2_distance, validate_dimension};
use crate::{MseCode, ProdCode, TurboQuantMse, TurboQuantProd};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutlierSplitPlan {
    dimension: usize,
    outlier_indices: Vec<usize>,
    regular_indices: Vec<usize>,
    outlier_bit_width: u8,
    regular_bit_width: u8,
}

#[derive(Debug, Clone)]
pub struct MixedMseCode {
    outlier_code: Option<MseCode>,
    regular_code: Option<MseCode>,
}

#[derive(Debug, Clone)]
pub struct MixedProdCode {
    outlier_code: Option<ProdCode>,
    regular_code: Option<ProdCode>,
}

#[derive(Debug, Clone)]
pub struct TurboQuantMixedMse {
    plan: OutlierSplitPlan,
    outlier_quantizer: Option<TurboQuantMse>,
    regular_quantizer: Option<TurboQuantMse>,
}

#[derive(Debug, Clone)]
pub struct TurboQuantMixedProd {
    plan: OutlierSplitPlan,
    outlier_quantizer: Option<TurboQuantProd>,
    regular_quantizer: Option<TurboQuantProd>,
}

impl OutlierSplitPlan {
    pub fn new(
        dimension: usize,
        mut outlier_indices: Vec<usize>,
        outlier_bit_width: u8,
        regular_bit_width: u8,
    ) -> Result<Self> {
        if dimension < 2 {
            return Err(TurboQuantError::InvalidDimension(dimension));
        }

        outlier_indices.sort_unstable();
        let mut seen = vec![false; dimension];
        for &index in &outlier_indices {
            if index >= dimension {
                return Err(TurboQuantError::InvalidChannelIndex { index, dimension });
            }
            if seen[index] {
                return Err(TurboQuantError::DuplicateChannelIndex { index });
            }
            seen[index] = true;
        }

        let regular_indices = (0..dimension)
            .filter(|&index| !seen[index])
            .collect::<Vec<_>>();
        validate_partition_lengths(outlier_indices.len(), regular_indices.len())?;

        Ok(Self {
            dimension,
            outlier_indices,
            regular_indices,
            outlier_bit_width,
            regular_bit_width,
        })
    }

    pub fn from_channel_rms<T>(
        samples: &[T],
        outlier_count: usize,
        outlier_bit_width: u8,
        regular_bit_width: u8,
    ) -> Result<Self>
    where
        T: AsRef<[f64]>,
    {
        let first = samples
            .first()
            .ok_or(TurboQuantError::EmptyCalibrationSet)?;
        let dimension = first.as_ref().len();
        validate_dimension(dimension, first.as_ref().len())?;
        if outlier_count > dimension {
            return Err(TurboQuantError::InvalidChannelPartition {
                outliers: outlier_count,
                regular: 0,
            });
        }

        let mut scores = vec![0.0; dimension];
        for sample in samples {
            let values = sample.as_ref();
            validate_dimension(dimension, values.len())?;
            for (index, &value) in values.iter().enumerate() {
                scores[index] += value * value;
            }
        }

        let mut ranked = (0..dimension).collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| {
            scores[right]
                .partial_cmp(&scores[left])
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.cmp(&right))
        });

        let mut outlier_indices = ranked.into_iter().take(outlier_count).collect::<Vec<_>>();
        outlier_indices.sort_unstable();
        Self::new(
            dimension,
            outlier_indices,
            outlier_bit_width,
            regular_bit_width,
        )
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn outlier_indices(&self) -> &[usize] {
        &self.outlier_indices
    }

    pub fn regular_indices(&self) -> &[usize] {
        &self.regular_indices
    }

    pub fn outlier_bit_width(&self) -> u8 {
        self.outlier_bit_width
    }

    pub fn regular_bit_width(&self) -> u8 {
        self.regular_bit_width
    }

    pub fn effective_bit_width(&self) -> f64 {
        let total_bits = self.outlier_indices.len() as f64 * self.outlier_bit_width as f64
            + self.regular_indices.len() as f64 * self.regular_bit_width as f64;
        total_bits / self.dimension as f64
    }
}

impl TurboQuantMixedMse {
    pub fn new(plan: OutlierSplitPlan, seed: u64) -> Result<Self> {
        Ok(Self {
            outlier_quantizer: build_mse_quantizer(
                plan.outlier_indices.len(),
                plan.outlier_bit_width,
                seed ^ 0xA24B_AED4_963E_E407,
            )?,
            regular_quantizer: build_mse_quantizer(
                plan.regular_indices.len(),
                plan.regular_bit_width,
                seed ^ 0x9FB2_1C65_1E98_DF25,
            )?,
            plan,
        })
    }

    pub fn plan(&self) -> &OutlierSplitPlan {
        &self.plan
    }

    pub fn effective_bit_width(&self) -> f64 {
        self.plan.effective_bit_width()
    }

    pub fn quantize(&self, vector: &[f64]) -> Result<MixedMseCode> {
        validate_dimension(self.plan.dimension, vector.len())?;
        let outlier_code = self
            .outlier_quantizer
            .as_ref()
            .map(|quantizer| quantizer.quantize(&gather(vector, self.plan.outlier_indices())))
            .transpose()?;
        let regular_code = self
            .regular_quantizer
            .as_ref()
            .map(|quantizer| quantizer.quantize(&gather(vector, self.plan.regular_indices())))
            .transpose()?;

        Ok(MixedMseCode {
            outlier_code,
            regular_code,
        })
    }

    pub fn dequantize(&self, code: &MixedMseCode) -> Result<Vec<f64>> {
        let mut reconstructed = vec![0.0; self.plan.dimension];

        if let Some(quantizer) = &self.outlier_quantizer {
            let decoded = quantizer.dequantize(
                code.outlier_code
                    .as_ref()
                    .expect("mixed MSE outlier partition is internally consistent"),
            )?;
            scatter(
                &mut reconstructed,
                self.plan.outlier_indices(),
                &decoded,
                self.plan.dimension,
            )?;
        }

        if let Some(quantizer) = &self.regular_quantizer {
            let decoded = quantizer.dequantize(
                code.regular_code
                    .as_ref()
                    .expect("mixed MSE regular partition is internally consistent"),
            )?;
            scatter(
                &mut reconstructed,
                self.plan.regular_indices(),
                &decoded,
                self.plan.dimension,
            )?;
        }

        Ok(reconstructed)
    }

    pub fn quantize_batch<T>(&self, vectors: &[T]) -> Result<Vec<MixedMseCode>>
    where
        T: AsRef<[f64]> + Sync,
    {
        let results = vectors
            .par_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn dequantize_batch(&self, codes: &[MixedMseCode]) -> Result<Vec<Vec<f64>>> {
        let results = codes
            .par_iter()
            .map(|code| self.dequantize(code))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn reconstruction_error(&self, vector: &[f64], code: &MixedMseCode) -> Result<f64> {
        validate_dimension(self.plan.dimension, vector.len())?;
        let decoded = self.dequantize(code)?;
        Ok(squared_l2_distance(vector, &decoded))
    }
}

impl TurboQuantMixedProd {
    pub fn new(plan: OutlierSplitPlan, seed: u64) -> Result<Self> {
        Ok(Self {
            outlier_quantizer: build_prod_quantizer(
                plan.outlier_indices.len(),
                plan.outlier_bit_width,
                seed ^ 0x3C79_AC49_2BA7_B653,
            )?,
            regular_quantizer: build_prod_quantizer(
                plan.regular_indices.len(),
                plan.regular_bit_width,
                seed ^ 0x1C69_B3F7_4AC4_AE35,
            )?,
            plan,
        })
    }

    pub fn plan(&self) -> &OutlierSplitPlan {
        &self.plan
    }

    pub fn effective_bit_width(&self) -> f64 {
        self.plan.effective_bit_width()
    }

    pub fn quantize(&self, vector: &[f64]) -> Result<MixedProdCode> {
        validate_dimension(self.plan.dimension, vector.len())?;
        let outlier_code = self
            .outlier_quantizer
            .as_ref()
            .map(|quantizer| quantizer.quantize(&gather(vector, self.plan.outlier_indices())))
            .transpose()?;
        let regular_code = self
            .regular_quantizer
            .as_ref()
            .map(|quantizer| quantizer.quantize(&gather(vector, self.plan.regular_indices())))
            .transpose()?;

        Ok(MixedProdCode {
            outlier_code,
            regular_code,
        })
    }

    pub fn dequantize(&self, code: &MixedProdCode) -> Result<Vec<f64>> {
        let mut reconstructed = vec![0.0; self.plan.dimension];

        if let Some(quantizer) = &self.outlier_quantizer {
            let decoded = quantizer.dequantize(
                code.outlier_code
                    .as_ref()
                    .expect("mixed prod outlier partition is internally consistent"),
            )?;
            scatter(
                &mut reconstructed,
                self.plan.outlier_indices(),
                &decoded,
                self.plan.dimension,
            )?;
        }

        if let Some(quantizer) = &self.regular_quantizer {
            let decoded = quantizer.dequantize(
                code.regular_code
                    .as_ref()
                    .expect("mixed prod regular partition is internally consistent"),
            )?;
            scatter(
                &mut reconstructed,
                self.plan.regular_indices(),
                &decoded,
                self.plan.dimension,
            )?;
        }

        Ok(reconstructed)
    }

    pub fn quantize_batch<T>(&self, vectors: &[T]) -> Result<Vec<MixedProdCode>>
    where
        T: AsRef<[f64]> + Sync,
    {
        let results = vectors
            .par_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn estimate_inner_product(&self, code: &MixedProdCode, query: &[f64]) -> Result<f64> {
        validate_dimension(self.plan.dimension, query.len())?;

        let mut estimate = 0.0;
        if let Some(quantizer) = &self.outlier_quantizer {
            estimate += quantizer.estimate_inner_product(
                code.outlier_code
                    .as_ref()
                    .expect("mixed prod outlier partition is internally consistent"),
                &gather(query, self.plan.outlier_indices()),
            )?;
        }
        if let Some(quantizer) = &self.regular_quantizer {
            estimate += quantizer.estimate_inner_product(
                code.regular_code
                    .as_ref()
                    .expect("mixed prod regular partition is internally consistent"),
                &gather(query, self.plan.regular_indices()),
            )?;
        }

        Ok(estimate)
    }

    pub fn estimate_inner_products_batch<T>(
        &self,
        codes: &[MixedProdCode],
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
}

impl MixedMseCode {
    pub fn outlier_code(&self) -> Option<&MseCode> {
        self.outlier_code.as_ref()
    }

    pub fn regular_code(&self) -> Option<&MseCode> {
        self.regular_code.as_ref()
    }

    pub fn storage_bytes(&self) -> usize {
        self.outlier_code
            .as_ref()
            .map(MseCode::storage_bytes)
            .unwrap_or(0)
            + self
                .regular_code
                .as_ref()
                .map(MseCode::storage_bytes)
                .unwrap_or(0)
    }
}

impl MixedProdCode {
    pub fn outlier_code(&self) -> Option<&ProdCode> {
        self.outlier_code.as_ref()
    }

    pub fn regular_code(&self) -> Option<&ProdCode> {
        self.regular_code.as_ref()
    }

    pub fn storage_bytes(&self) -> usize {
        self.outlier_code
            .as_ref()
            .map(ProdCode::storage_bytes)
            .unwrap_or(0)
            + self
                .regular_code
                .as_ref()
                .map(ProdCode::storage_bytes)
                .unwrap_or(0)
    }
}

fn build_mse_quantizer(
    dimension: usize,
    bit_width: u8,
    seed: u64,
) -> Result<Option<TurboQuantMse>> {
    if dimension == 0 {
        Ok(None)
    } else {
        Ok(Some(TurboQuantMse::new(dimension, bit_width, seed)?))
    }
}

fn build_prod_quantizer(
    dimension: usize,
    bit_width: u8,
    seed: u64,
) -> Result<Option<TurboQuantProd>> {
    if dimension == 0 {
        Ok(None)
    } else {
        Ok(Some(TurboQuantProd::new(dimension, bit_width, seed)?))
    }
}

fn validate_partition_lengths(outliers: usize, regular: usize) -> Result<()> {
    if (outliers == 1) || (regular == 1) {
        return Err(TurboQuantError::InvalidChannelPartition { outliers, regular });
    }
    Ok(())
}

fn gather(vector: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&index| vector[index]).collect()
}

fn scatter(target: &mut [f64], indices: &[usize], values: &[f64], dimension: usize) -> Result<()> {
    validate_dimension(indices.len(), values.len())?;
    validate_dimension(dimension, target.len())?;
    for (&index, &value) in indices.iter().zip(values) {
        target[index] = value;
    }
    Ok(())
}
