use std::cmp::Ordering;

use rayon::prelude::*;

use crate::error::{Result, TurboQuantError};
use crate::rotation::RotationBackend;
use crate::{
    MixedMseCode, MixedProdCode, MseCode, OutlierSplitPlan, ProdCode, TurboQuantMixedMse,
    TurboQuantMixedProd, TurboQuantMse, TurboQuantProd,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvTensorShape {
    batch_size: usize,
    kv_heads: usize,
    sequence_length: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvTensor {
    shape: KvTensorShape,
    values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct KvTensorCodeRows<T> {
    shape: KvTensorShape,
    rows: Vec<T>,
}

pub type KvMseTensorCode = KvTensorCodeRows<MseCode>;
pub type KvProdTensorCode = KvTensorCodeRows<ProdCode>;
pub type KvMixedMseTensorCode = KvTensorCodeRows<MixedMseCode>;
pub type KvMixedProdTensorCode = KvTensorCodeRows<MixedProdCode>;

#[derive(Debug, Clone)]
pub struct KvMseQuantizer {
    inner: TurboQuantMse,
}

#[derive(Debug, Clone)]
pub struct KvProdQuantizer {
    inner: TurboQuantProd,
}

#[derive(Debug, Clone)]
pub struct KvMixedMseQuantizer {
    inner: TurboQuantMixedMse,
}

#[derive(Debug, Clone)]
pub struct KvMixedProdQuantizer {
    inner: TurboQuantMixedProd,
}

#[derive(Debug, Clone)]
pub enum KvQuantizer {
    Mse(KvMseQuantizer),
    Prod(KvProdQuantizer),
    MixedMse(KvMixedMseQuantizer),
    MixedProd(KvMixedProdQuantizer),
}

#[derive(Debug, Clone)]
pub enum KvQuantizerSpec {
    Mse { bit_width: u8 },
    Prod { bit_width: u8 },
    FastMse { bit_width: u8 },
    FastProd { bit_width: u8 },
    MixedMse { plan: OutlierSplitPlan },
    MixedProd { plan: OutlierSplitPlan },
}

#[derive(Debug, Clone)]
pub enum KvTensorCode {
    Mse(KvMseTensorCode),
    Prod(KvProdTensorCode),
    MixedMse(KvMixedMseTensorCode),
    MixedProd(KvMixedProdTensorCode),
}

#[derive(Debug, Clone)]
pub struct QuantizedKvCacheLayer {
    batch_size: usize,
    kv_heads: usize,
    head_dim: usize,
    residual_length: usize,
    key_quantizer: KvQuantizer,
    value_quantizer: KvQuantizer,
    quantized_keys: Option<KvTensorCode>,
    quantized_values: Option<KvTensorCode>,
    residual_keys: KvTensor,
    residual_values: KvTensor,
}

#[derive(Debug, Clone)]
pub struct FullPrecisionKvCacheLayer {
    batch_size: usize,
    kv_heads: usize,
    head_dim: usize,
    keys: KvTensor,
    values: KvTensor,
}

#[derive(Debug, Clone)]
pub enum KvCacheLayer {
    Quantized(QuantizedKvCacheLayer),
    FullPrecision(FullPrecisionKvCacheLayer),
}

#[derive(Debug, Clone)]
pub struct TurboQuantKvCacheConfig {
    pub num_layers: usize,
    pub batch_size: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub residual_length: usize,
    pub key_spec: KvQuantizerSpec,
    pub value_spec: KvQuantizerSpec,
    pub seed: u64,
    pub skip_layers: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct TurboQuantKvCache {
    layers: Vec<KvCacheLayer>,
}

#[derive(Debug, Clone)]
pub struct KvLayerNormAnalysis {
    norms: Vec<f64>,
    median_norm: f64,
    max_norm: f64,
    max_norm_layer: usize,
    skip_layers: Vec<usize>,
}

pub trait CodeStorage {
    fn storage_bytes(&self) -> usize;
}

impl CodeStorage for MseCode {
    fn storage_bytes(&self) -> usize {
        MseCode::storage_bytes(self)
    }
}

impl CodeStorage for ProdCode {
    fn storage_bytes(&self) -> usize {
        ProdCode::storage_bytes(self)
    }
}

impl CodeStorage for MixedMseCode {
    fn storage_bytes(&self) -> usize {
        MixedMseCode::storage_bytes(self)
    }
}

impl CodeStorage for MixedProdCode {
    fn storage_bytes(&self) -> usize {
        MixedProdCode::storage_bytes(self)
    }
}

impl KvTensorShape {
    pub fn new(
        batch_size: usize,
        kv_heads: usize,
        sequence_length: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if batch_size == 0 {
            return Err(TurboQuantError::InvalidTensorShape {
                axis: "batch_size",
                value: batch_size,
            });
        }
        if kv_heads == 0 {
            return Err(TurboQuantError::InvalidTensorShape {
                axis: "kv_heads",
                value: kv_heads,
            });
        }
        if head_dim == 0 {
            return Err(TurboQuantError::InvalidTensorShape {
                axis: "head_dim",
                value: head_dim,
            });
        }

        Ok(Self {
            batch_size,
            kv_heads,
            sequence_length,
            head_dim,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn lane_count(&self) -> usize {
        self.batch_size * self.kv_heads
    }

    pub fn row_count(&self) -> usize {
        self.lane_count() * self.sequence_length
    }

    pub fn element_count(&self) -> usize {
        self.row_count() * self.head_dim
    }
}

impl KvTensor {
    pub fn new(shape: KvTensorShape, values: Vec<f64>) -> Result<Self> {
        let expected = shape.element_count();
        let actual = values.len();
        if actual != expected {
            return Err(TurboQuantError::TensorElementCountMismatch { expected, actual });
        }
        Ok(Self { shape, values })
    }

    pub fn zeros(shape: KvTensorShape) -> Self {
        Self {
            shape,
            values: vec![0.0; shape.element_count()],
        }
    }

    pub fn shape(&self) -> KvTensorShape {
        self.shape
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    pub fn storage_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>()
    }

    pub fn mean_vector_norm(&self) -> f64 {
        if self.shape.row_count() == 0 {
            return 0.0;
        }

        self.values
            .chunks(self.shape.head_dim)
            .map(|row| row.iter().map(|value| value * value).sum::<f64>().sqrt())
            .sum::<f64>()
            / self.shape.row_count() as f64
    }

    pub fn concat_seq(&self, other: &Self) -> Result<Self> {
        validate_layout(self.shape, other.shape)?;

        let lhs_block_len = self.shape.sequence_length * self.shape.head_dim;
        let rhs_block_len = other.shape.sequence_length * other.shape.head_dim;
        let mut values = Vec::with_capacity(self.values.len() + other.values.len());

        for lane in 0..self.shape.lane_count() {
            let lhs_start = lane * lhs_block_len;
            values.extend_from_slice(&self.values[lhs_start..lhs_start + lhs_block_len]);

            let rhs_start = lane * rhs_block_len;
            values.extend_from_slice(&other.values[rhs_start..rhs_start + rhs_block_len]);
        }

        KvTensor::new(
            KvTensorShape {
                sequence_length: self.shape.sequence_length + other.shape.sequence_length,
                ..self.shape
            },
            values,
        )
    }

    pub fn slice_seq(&self, start: usize, end: usize) -> Result<Self> {
        if start > end || end > self.shape.sequence_length {
            return Err(TurboQuantError::InvalidSequenceRange {
                start,
                end,
                sequence_length: self.shape.sequence_length,
            });
        }

        let slice_seq = end - start;
        let lane_block_len = self.shape.sequence_length * self.shape.head_dim;
        let slice_block_len = slice_seq * self.shape.head_dim;
        let mut values = Vec::with_capacity(self.shape.lane_count() * slice_block_len);

        for lane in 0..self.shape.lane_count() {
            let lane_start = lane * lane_block_len + start * self.shape.head_dim;
            values.extend_from_slice(&self.values[lane_start..lane_start + slice_block_len]);
        }

        KvTensor::new(
            KvTensorShape {
                sequence_length: slice_seq,
                ..self.shape
            },
            values,
        )
    }
}

impl<T> KvTensorCodeRows<T> {
    pub fn shape(&self) -> KvTensorShape {
        self.shape
    }

    pub fn row_codes(&self) -> &[T] {
        &self.rows
    }
}

impl<T: Clone> KvTensorCodeRows<T> {
    fn new(shape: KvTensorShape, rows: Vec<T>) -> Result<Self> {
        let expected = shape.row_count();
        let actual = rows.len();
        if actual != expected {
            return Err(TurboQuantError::TensorRowCountMismatch { expected, actual });
        }

        Ok(Self { shape, rows })
    }

    fn append_seq(&self, other: &Self) -> Result<Self> {
        validate_layout(self.shape, other.shape)?;

        let rows = append_lane_grouped(
            &self.rows,
            self.shape.sequence_length,
            &other.rows,
            other.shape.sequence_length,
            self.shape.lane_count(),
        );

        Self::new(
            KvTensorShape {
                sequence_length: self.shape.sequence_length + other.shape.sequence_length,
                ..self.shape
            },
            rows,
        )
    }
}

impl<T: CodeStorage> KvTensorCodeRows<T> {
    pub fn storage_bytes(&self) -> usize {
        self.rows.iter().map(CodeStorage::storage_bytes).sum()
    }
}

impl KvMseQuantizer {
    pub fn new(head_dim: usize, bit_width: u8, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: TurboQuantMse::new(head_dim, bit_width, seed)?,
        })
    }

    pub fn new_fast(head_dim: usize, bit_width: u8, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: TurboQuantMse::new_with_rotation_backend(
                head_dim,
                bit_width,
                seed,
                RotationBackend::WalshHadamard,
            )?,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.inner.dimension()
    }

    pub fn quantize_tensor(&self, tensor: &KvTensor) -> Result<KvMseTensorCode> {
        validate_head_dim(self.head_dim(), tensor.shape.head_dim())?;
        let codes = tensor
            .values()
            .par_chunks(tensor.shape.head_dim())
            .map(|row| self.inner.quantize(row))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        KvTensorCodeRows::new(tensor.shape, codes)
    }

    pub fn dequantize_tensor(&self, code: &KvMseTensorCode) -> Result<KvTensor> {
        validate_head_dim(self.head_dim(), code.shape.head_dim())?;
        let rows = self.inner.dequantize_batch(code.row_codes())?;
        KvTensor::new(code.shape(), rows.into_iter().flatten().collect())
    }
}

impl KvProdQuantizer {
    pub fn new(head_dim: usize, bit_width: u8, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: TurboQuantProd::new(head_dim, bit_width, seed)?,
        })
    }

    pub fn new_fast(head_dim: usize, bit_width: u8, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: TurboQuantProd::new_with_rotation_backend(
                head_dim,
                bit_width,
                seed,
                RotationBackend::WalshHadamard,
            )?,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.inner.dimension()
    }

    pub fn quantize_tensor(&self, tensor: &KvTensor) -> Result<KvProdTensorCode> {
        validate_head_dim(self.head_dim(), tensor.shape.head_dim())?;
        let codes = self
            .inner
            .quantize_flat_batch(tensor.values(), tensor.shape.row_count())?;
        KvTensorCodeRows::new(tensor.shape, codes)
    }

    pub fn dequantize_tensor(&self, code: &KvProdTensorCode) -> Result<KvTensor> {
        validate_head_dim(self.head_dim(), code.shape.head_dim())?;
        let values = self.inner.dequantize_flat_batch(code.row_codes())?;
        KvTensor::new(code.shape(), values)
    }
}

impl KvMixedMseQuantizer {
    pub fn new(plan: OutlierSplitPlan, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: TurboQuantMixedMse::new(plan, seed)?,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.inner.plan().dimension()
    }

    pub fn quantize_tensor(&self, tensor: &KvTensor) -> Result<KvMixedMseTensorCode> {
        validate_head_dim(self.head_dim(), tensor.shape.head_dim())?;
        let codes = tensor
            .values()
            .par_chunks(tensor.shape.head_dim())
            .map(|row| self.inner.quantize(row))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        KvTensorCodeRows::new(tensor.shape, codes)
    }

    pub fn dequantize_tensor(&self, code: &KvMixedMseTensorCode) -> Result<KvTensor> {
        validate_head_dim(self.head_dim(), code.shape.head_dim())?;
        let rows = code
            .row_codes()
            .iter()
            .map(|row| self.inner.dequantize(row))
            .collect::<Result<Vec<_>>>()?;
        KvTensor::new(code.shape(), rows.into_iter().flatten().collect())
    }
}

impl KvMixedProdQuantizer {
    pub fn new(plan: OutlierSplitPlan, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: TurboQuantMixedProd::new(plan, seed)?,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.inner.plan().dimension()
    }

    pub fn quantize_tensor(&self, tensor: &KvTensor) -> Result<KvMixedProdTensorCode> {
        validate_head_dim(self.head_dim(), tensor.shape.head_dim())?;
        let codes = tensor
            .values()
            .par_chunks(tensor.shape.head_dim())
            .map(|row| self.inner.quantize(row))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        KvTensorCodeRows::new(tensor.shape, codes)
    }

    pub fn dequantize_tensor(&self, code: &KvMixedProdTensorCode) -> Result<KvTensor> {
        validate_head_dim(self.head_dim(), code.shape.head_dim())?;
        let rows = code
            .row_codes()
            .iter()
            .map(|row| self.inner.dequantize(row))
            .collect::<Result<Vec<_>>>()?;
        KvTensor::new(code.shape(), rows.into_iter().flatten().collect())
    }
}

impl KvQuantizerSpec {
    pub fn build(&self, head_dim: usize, seed: u64) -> Result<KvQuantizer> {
        match self {
            KvQuantizerSpec::Mse { bit_width } => Ok(KvQuantizer::Mse(KvMseQuantizer::new(
                head_dim, *bit_width, seed,
            )?)),
            KvQuantizerSpec::Prod { bit_width } => Ok(KvQuantizer::Prod(KvProdQuantizer::new(
                head_dim, *bit_width, seed,
            )?)),
            KvQuantizerSpec::FastMse { bit_width } => Ok(KvQuantizer::Mse(
                KvMseQuantizer::new_fast(head_dim, *bit_width, seed)?,
            )),
            KvQuantizerSpec::FastProd { bit_width } => Ok(KvQuantizer::Prod(
                KvProdQuantizer::new_fast(head_dim, *bit_width, seed)?,
            )),
            KvQuantizerSpec::MixedMse { plan } => {
                validate_head_dim(plan.dimension(), head_dim)?;
                Ok(KvQuantizer::MixedMse(KvMixedMseQuantizer::new(
                    plan.clone(),
                    seed,
                )?))
            }
            KvQuantizerSpec::MixedProd { plan } => {
                validate_head_dim(plan.dimension(), head_dim)?;
                Ok(KvQuantizer::MixedProd(KvMixedProdQuantizer::new(
                    plan.clone(),
                    seed,
                )?))
            }
        }
    }
}

impl KvQuantizer {
    pub fn head_dim(&self) -> usize {
        match self {
            KvQuantizer::Mse(quantizer) => quantizer.head_dim(),
            KvQuantizer::Prod(quantizer) => quantizer.head_dim(),
            KvQuantizer::MixedMse(quantizer) => quantizer.head_dim(),
            KvQuantizer::MixedProd(quantizer) => quantizer.head_dim(),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            KvQuantizer::Mse(_) => "mse",
            KvQuantizer::Prod(_) => "prod",
            KvQuantizer::MixedMse(_) => "mixed_mse",
            KvQuantizer::MixedProd(_) => "mixed_prod",
        }
    }

    pub fn quantize(&self, tensor: &KvTensor) -> Result<KvTensorCode> {
        match self {
            KvQuantizer::Mse(quantizer) => {
                Ok(KvTensorCode::Mse(quantizer.quantize_tensor(tensor)?))
            }
            KvQuantizer::Prod(quantizer) => {
                Ok(KvTensorCode::Prod(quantizer.quantize_tensor(tensor)?))
            }
            KvQuantizer::MixedMse(quantizer) => {
                Ok(KvTensorCode::MixedMse(quantizer.quantize_tensor(tensor)?))
            }
            KvQuantizer::MixedProd(quantizer) => {
                Ok(KvTensorCode::MixedProd(quantizer.quantize_tensor(tensor)?))
            }
        }
    }

    pub fn dequantize(&self, code: &KvTensorCode) -> Result<KvTensor> {
        match (self, code) {
            (KvQuantizer::Mse(quantizer), KvTensorCode::Mse(code)) => {
                quantizer.dequantize_tensor(code)
            }
            (KvQuantizer::Prod(quantizer), KvTensorCode::Prod(code)) => {
                quantizer.dequantize_tensor(code)
            }
            (KvQuantizer::MixedMse(quantizer), KvTensorCode::MixedMse(code)) => {
                quantizer.dequantize_tensor(code)
            }
            (KvQuantizer::MixedProd(quantizer), KvTensorCode::MixedProd(code)) => {
                quantizer.dequantize_tensor(code)
            }
            _ => Err(TurboQuantError::KvQuantizerMismatch {
                expected: self.kind_name(),
                actual: code.kind_name(),
            }),
        }
    }
}

impl KvTensorCode {
    pub fn shape(&self) -> KvTensorShape {
        match self {
            KvTensorCode::Mse(code) => code.shape(),
            KvTensorCode::Prod(code) => code.shape(),
            KvTensorCode::MixedMse(code) => code.shape(),
            KvTensorCode::MixedProd(code) => code.shape(),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            KvTensorCode::Mse(_) => "mse",
            KvTensorCode::Prod(_) => "prod",
            KvTensorCode::MixedMse(_) => "mixed_mse",
            KvTensorCode::MixedProd(_) => "mixed_prod",
        }
    }

    pub fn storage_bytes(&self) -> usize {
        match self {
            KvTensorCode::Mse(code) => code.storage_bytes(),
            KvTensorCode::Prod(code) => code.storage_bytes(),
            KvTensorCode::MixedMse(code) => code.storage_bytes(),
            KvTensorCode::MixedProd(code) => code.storage_bytes(),
        }
    }

    pub fn append_seq(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (KvTensorCode::Mse(lhs), KvTensorCode::Mse(rhs)) => {
                Ok(KvTensorCode::Mse(lhs.append_seq(rhs)?))
            }
            (KvTensorCode::Prod(lhs), KvTensorCode::Prod(rhs)) => {
                Ok(KvTensorCode::Prod(lhs.append_seq(rhs)?))
            }
            (KvTensorCode::MixedMse(lhs), KvTensorCode::MixedMse(rhs)) => {
                Ok(KvTensorCode::MixedMse(lhs.append_seq(rhs)?))
            }
            (KvTensorCode::MixedProd(lhs), KvTensorCode::MixedProd(rhs)) => {
                Ok(KvTensorCode::MixedProd(lhs.append_seq(rhs)?))
            }
            _ => Err(TurboQuantError::KvQuantizerMismatch {
                expected: self.kind_name(),
                actual: other.kind_name(),
            }),
        }
    }
}

impl QuantizedKvCacheLayer {
    pub fn new(
        batch_size: usize,
        kv_heads: usize,
        head_dim: usize,
        residual_length: usize,
        key_quantizer: KvQuantizer,
        value_quantizer: KvQuantizer,
    ) -> Result<Self> {
        validate_head_dim(head_dim, key_quantizer.head_dim())?;
        validate_head_dim(head_dim, value_quantizer.head_dim())?;
        let empty = empty_tensor(batch_size, kv_heads, head_dim)?;

        Ok(Self {
            batch_size,
            kv_heads,
            head_dim,
            residual_length,
            key_quantizer,
            value_quantizer,
            quantized_keys: None,
            quantized_values: None,
            residual_keys: empty.clone(),
            residual_values: empty,
        })
    }

    pub fn update(&mut self, new_keys: &KvTensor, new_values: &KvTensor) -> Result<()> {
        validate_layer_update_shape(
            self.batch_size,
            self.kv_heads,
            self.head_dim,
            new_keys.shape(),
            new_values.shape(),
        )?;

        let combined_keys = self.residual_keys.concat_seq(new_keys)?;
        let combined_values = self.residual_values.concat_seq(new_values)?;
        let total_seq = combined_keys.shape().sequence_length();
        let overflow = total_seq.saturating_sub(self.residual_length);

        if overflow > 0 {
            let key_prefix = combined_keys.slice_seq(0, overflow)?;
            let value_prefix = combined_values.slice_seq(0, overflow)?;
            self.quantized_keys = Some(append_quantized_prefix(
                self.quantized_keys.take(),
                &self.key_quantizer,
                &key_prefix,
            )?);
            self.quantized_values = Some(append_quantized_prefix(
                self.quantized_values.take(),
                &self.value_quantizer,
                &value_prefix,
            )?);
        }

        self.residual_keys = combined_keys.slice_seq(overflow, total_seq)?;
        self.residual_values = combined_values.slice_seq(overflow, total_seq)?;
        Ok(())
    }

    pub fn materialize_keys(&self) -> Result<KvTensor> {
        materialize_layer_part(
            &self.quantized_keys,
            &self.key_quantizer,
            &self.residual_keys,
            self.batch_size,
            self.kv_heads,
            self.head_dim,
        )
    }

    pub fn materialize_values(&self) -> Result<KvTensor> {
        materialize_layer_part(
            &self.quantized_values,
            &self.value_quantizer,
            &self.residual_values,
            self.batch_size,
            self.kv_heads,
            self.head_dim,
        )
    }

    pub fn storage_bytes(&self) -> usize {
        self.quantized_keys
            .as_ref()
            .map(KvTensorCode::storage_bytes)
            .unwrap_or(0)
            + self
                .quantized_values
                .as_ref()
                .map(KvTensorCode::storage_bytes)
                .unwrap_or(0)
            + self.residual_keys.storage_bytes()
            + self.residual_values.storage_bytes()
    }

    pub fn seq_length(&self) -> usize {
        self.quantized_prefix_length() + self.residual_keys.shape().sequence_length()
    }

    pub fn quantized_prefix_length(&self) -> usize {
        self.quantized_keys
            .as_ref()
            .map(|code| code.shape().sequence_length())
            .unwrap_or(0)
    }
}

impl FullPrecisionKvCacheLayer {
    pub fn new(batch_size: usize, kv_heads: usize, head_dim: usize) -> Result<Self> {
        let empty = empty_tensor(batch_size, kv_heads, head_dim)?;
        Ok(Self {
            batch_size,
            kv_heads,
            head_dim,
            keys: empty.clone(),
            values: empty,
        })
    }

    pub fn update(&mut self, new_keys: &KvTensor, new_values: &KvTensor) -> Result<()> {
        validate_layer_update_shape(
            self.batch_size,
            self.kv_heads,
            self.head_dim,
            new_keys.shape(),
            new_values.shape(),
        )?;
        self.keys = self.keys.concat_seq(new_keys)?;
        self.values = self.values.concat_seq(new_values)?;
        Ok(())
    }

    pub fn materialize_keys(&self) -> KvTensor {
        self.keys.clone()
    }

    pub fn materialize_values(&self) -> KvTensor {
        self.values.clone()
    }

    pub fn storage_bytes(&self) -> usize {
        self.keys.storage_bytes() + self.values.storage_bytes()
    }

    pub fn seq_length(&self) -> usize {
        self.keys.shape().sequence_length()
    }
}

impl KvCacheLayer {
    pub fn update(&mut self, new_keys: &KvTensor, new_values: &KvTensor) -> Result<()> {
        match self {
            KvCacheLayer::Quantized(layer) => layer.update(new_keys, new_values),
            KvCacheLayer::FullPrecision(layer) => layer.update(new_keys, new_values),
        }
    }

    pub fn materialize(&self) -> Result<(KvTensor, KvTensor)> {
        match self {
            KvCacheLayer::Quantized(layer) => {
                Ok((layer.materialize_keys()?, layer.materialize_values()?))
            }
            KvCacheLayer::FullPrecision(layer) => {
                Ok((layer.materialize_keys(), layer.materialize_values()))
            }
        }
    }

    pub fn storage_bytes(&self) -> usize {
        match self {
            KvCacheLayer::Quantized(layer) => layer.storage_bytes(),
            KvCacheLayer::FullPrecision(layer) => layer.storage_bytes(),
        }
    }

    pub fn seq_length(&self) -> usize {
        match self {
            KvCacheLayer::Quantized(layer) => layer.seq_length(),
            KvCacheLayer::FullPrecision(layer) => layer.seq_length(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, KvCacheLayer::Quantized(_))
    }
}

impl TurboQuantKvCache {
    pub fn new(config: TurboQuantKvCacheConfig) -> Result<Self> {
        if config.num_layers == 0 {
            return Err(TurboQuantError::InvalidTensorShape {
                axis: "num_layers",
                value: config.num_layers,
            });
        }

        let mut skipped = vec![false; config.num_layers];
        for &layer_index in &config.skip_layers {
            if layer_index >= config.num_layers {
                return Err(TurboQuantError::LayerIndexOutOfRange {
                    layer_index,
                    layer_count: config.num_layers,
                });
            }
            if skipped[layer_index] {
                return Err(TurboQuantError::DuplicateLayerIndex { layer_index });
            }
            skipped[layer_index] = true;
        }

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_index in 0..config.num_layers {
            if skipped[layer_index] {
                layers.push(KvCacheLayer::FullPrecision(FullPrecisionKvCacheLayer::new(
                    config.batch_size,
                    config.kv_heads,
                    config.head_dim,
                )?));
                continue;
            }

            let key_quantizer = config.key_spec.build(
                config.head_dim,
                derive_seed(config.seed, layer_index, 0x4D59_6473),
            )?;
            let value_quantizer = config.value_spec.build(
                config.head_dim,
                derive_seed(config.seed, layer_index, 0x5661_6C73),
            )?;
            layers.push(KvCacheLayer::Quantized(QuantizedKvCacheLayer::new(
                config.batch_size,
                config.kv_heads,
                config.head_dim,
                config.residual_length,
                key_quantizer,
                value_quantizer,
            )?));
        }

        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, layer_index: usize) -> Result<&KvCacheLayer> {
        self.layers
            .get(layer_index)
            .ok_or(TurboQuantError::LayerIndexOutOfRange {
                layer_index,
                layer_count: self.layers.len(),
            })
    }

    pub fn update(
        &mut self,
        layer_index: usize,
        new_keys: &KvTensor,
        new_values: &KvTensor,
    ) -> Result<()> {
        let layer_count = self.layers.len();
        let layer =
            self.layers
                .get_mut(layer_index)
                .ok_or(TurboQuantError::LayerIndexOutOfRange {
                    layer_index,
                    layer_count,
                })?;
        layer.update(new_keys, new_values)
    }

    pub fn materialize_layer(&self, layer_index: usize) -> Result<(KvTensor, KvTensor)> {
        self.layer(layer_index)?.materialize()
    }

    pub fn get_seq_length(&self, layer_index: usize) -> Result<usize> {
        Ok(self.layer(layer_index)?.seq_length())
    }

    pub fn storage_bytes(&self) -> usize {
        self.layers.iter().map(KvCacheLayer::storage_bytes).sum()
    }
}

impl KvLayerNormAnalysis {
    pub fn norms(&self) -> &[f64] {
        &self.norms
    }

    pub fn median_norm(&self) -> f64 {
        self.median_norm
    }

    pub fn max_norm(&self) -> f64 {
        self.max_norm
    }

    pub fn max_norm_layer(&self) -> usize {
        self.max_norm_layer
    }

    pub fn skip_layers(&self) -> &[usize] {
        &self.skip_layers
    }
}

pub fn analyze_kv_layer_norms(
    key_layers: &[KvTensor],
    norm_threshold: f64,
) -> Result<KvLayerNormAnalysis> {
    if key_layers.is_empty() {
        return Err(TurboQuantError::EmptyLayerSet);
    }

    let norms = key_layers
        .iter()
        .map(KvTensor::mean_vector_norm)
        .collect::<Vec<_>>();

    let mut sorted = norms.clone();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let median_norm = sorted[sorted.len() / 2];

    let (max_norm_layer, max_norm) = norms
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .expect("non-empty layer norms");

    let skip_layers = norms
        .iter()
        .enumerate()
        .filter_map(|(layer_index, &norm)| {
            if norm > norm_threshold * median_norm {
                Some(layer_index)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    Ok(KvLayerNormAnalysis {
        norms,
        median_norm,
        max_norm,
        max_norm_layer,
        skip_layers,
    })
}

pub fn calibrate_skip_layers(key_layers: &[KvTensor], norm_threshold: f64) -> Result<Vec<usize>> {
    Ok(analyze_kv_layer_norms(key_layers, norm_threshold)?.skip_layers)
}

fn append_quantized_prefix(
    existing: Option<KvTensorCode>,
    quantizer: &KvQuantizer,
    prefix: &KvTensor,
) -> Result<KvTensorCode> {
    let new_prefix = quantizer.quantize(prefix)?;
    match existing {
        Some(existing) => existing.append_seq(&new_prefix),
        None => Ok(new_prefix),
    }
}

fn materialize_layer_part(
    quantized_prefix: &Option<KvTensorCode>,
    quantizer: &KvQuantizer,
    residual: &KvTensor,
    batch_size: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<KvTensor> {
    let prefix = match quantized_prefix {
        Some(code) => quantizer.dequantize(code)?,
        None => empty_tensor(batch_size, kv_heads, head_dim)?,
    };
    prefix.concat_seq(residual)
}

fn empty_tensor(batch_size: usize, kv_heads: usize, head_dim: usize) -> Result<KvTensor> {
    Ok(KvTensor::zeros(KvTensorShape::new(
        batch_size, kv_heads, 0, head_dim,
    )?))
}

fn validate_layout(expected: KvTensorShape, actual: KvTensorShape) -> Result<()> {
    if expected.batch_size == actual.batch_size
        && expected.kv_heads == actual.kv_heads
        && expected.head_dim == actual.head_dim
    {
        Ok(())
    } else {
        Err(TurboQuantError::KvTensorLayoutMismatch {
            expected_batch: expected.batch_size,
            expected_heads: expected.kv_heads,
            expected_dim: expected.head_dim,
            actual_batch: actual.batch_size,
            actual_heads: actual.kv_heads,
            actual_dim: actual.head_dim,
        })
    }
}

fn validate_head_dim(expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(TurboQuantError::DimensionMismatch { expected, actual })
    }
}

fn validate_layer_update_shape(
    batch_size: usize,
    kv_heads: usize,
    head_dim: usize,
    key_shape: KvTensorShape,
    value_shape: KvTensorShape,
) -> Result<()> {
    if key_shape.batch_size != batch_size
        || key_shape.kv_heads != kv_heads
        || key_shape.head_dim != head_dim
    {
        return Err(TurboQuantError::KvTensorLayoutMismatch {
            expected_batch: batch_size,
            expected_heads: kv_heads,
            expected_dim: head_dim,
            actual_batch: key_shape.batch_size,
            actual_heads: key_shape.kv_heads,
            actual_dim: key_shape.head_dim,
        });
    }
    if value_shape.batch_size != batch_size
        || value_shape.kv_heads != kv_heads
        || value_shape.head_dim != head_dim
    {
        return Err(TurboQuantError::KvTensorLayoutMismatch {
            expected_batch: batch_size,
            expected_heads: kv_heads,
            expected_dim: head_dim,
            actual_batch: value_shape.batch_size,
            actual_heads: value_shape.kv_heads,
            actual_dim: value_shape.head_dim,
        });
    }
    if key_shape != value_shape {
        return Err(TurboQuantError::KvTensorShapeMismatch {
            expected_batch: key_shape.batch_size,
            expected_heads: key_shape.kv_heads,
            expected_seq: key_shape.sequence_length,
            expected_dim: key_shape.head_dim,
            actual_batch: value_shape.batch_size,
            actual_heads: value_shape.kv_heads,
            actual_seq: value_shape.sequence_length,
            actual_dim: value_shape.head_dim,
        });
    }
    Ok(())
}

fn append_lane_grouped<T: Clone>(
    left: &[T],
    left_seq: usize,
    right: &[T],
    right_seq: usize,
    lane_count: usize,
) -> Vec<T> {
    let mut combined = Vec::with_capacity(left.len() + right.len());
    for lane in 0..lane_count {
        let left_start = lane * left_seq;
        combined.extend_from_slice(&left[left_start..left_start + left_seq]);

        let right_start = lane * right_seq;
        combined.extend_from_slice(&right[right_start..right_start + right_seq]);
    }
    combined
}

fn derive_seed(base_seed: u64, layer_index: usize, salt: u64) -> u64 {
    base_seed ^ salt ^ ((layer_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}
