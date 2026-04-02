//! A Rust implementation of TurboQuant from
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
//! (arXiv:2504.19874).
//!
//! This crate implements the two quantizers described in the paper:
//! - `TurboQuantMse`: random rotation plus an optimal scalar Lloyd-Max codebook
//!   for the coordinate distribution induced by the unit sphere.
//! - `TurboQuantProd`: a `(b - 1)`-bit `TurboQuantMse` stage followed by a
//!   1-bit Quantized JL (QJL) sketch of the residual.

pub mod data;
mod error;
pub mod experiment;
mod kv;
mod lloyd_max;
mod math;
mod mixed;
mod mse;
mod packed;
mod pq;
mod prod;
mod rabitq;
mod rotation;

pub use error::{Result, TurboQuantError};
pub use kv::{
    FullPrecisionKvCacheLayer, KvCacheLayer, KvLayerNormAnalysis, KvMixedMseQuantizer,
    KvMixedMseTensorCode, KvMixedProdQuantizer, KvMixedProdTensorCode, KvMseQuantizer,
    KvMseTensorCode, KvProdQuantizer, KvProdTensorCode, KvQuantizer, KvQuantizerSpec, KvTensor,
    KvTensorCode, KvTensorCodeRows, KvTensorShape, QuantizedKvCacheLayer, TurboQuantKvCache,
    TurboQuantKvCacheConfig, analyze_kv_layer_norms, calibrate_skip_layers,
};
pub use lloyd_max::{LloydMaxOptions, ScalarCodebook};
pub use mixed::{
    MixedMseCode, MixedProdCode, OutlierSplitPlan, TurboQuantMixedMse, TurboQuantMixedProd,
};
pub use mse::{MseCode, TurboQuantMse};
pub use packed::PackedBits;
pub use pq::{PqCode, ProductQuantizer};
pub use prod::{ProdCode, TurboQuantProd};
pub use rabitq::{RaBitQCode, RaBitQQuantizer};
pub use rotation::RotationBackend;

#[cfg(test)]
mod tests {
    use super::{RotationBackend, TurboQuantMse, TurboQuantProd};
    use crate::math::squared_l2_distance;

    fn unit_vector(values: &[f64]) -> Vec<f64> {
        let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
        values.iter().map(|v| v / norm).collect()
    }

    #[test]
    fn mse_one_bit_codebook_matches_paper_scaling() {
        let quantizer = TurboQuantMse::new(1_024, 1, 7).unwrap();
        let codebook = quantizer.codebook();
        let expected = (2.0 / std::f64::consts::PI).sqrt() / (1_024.0f64).sqrt();

        assert!((codebook[0] + expected).abs() < 2e-3);
        assert!((codebook[1] - expected).abs() < 2e-3);
    }

    #[test]
    fn mse_reconstruction_error_improves_with_more_bits() {
        let vector = unit_vector(&[0.3, -0.7, 0.2, 1.2, -0.8, 0.1, 0.6, -0.4]);

        let mse_0 = TurboQuantMse::new(8, 0, 42).unwrap();
        let mse_1 = TurboQuantMse::new(8, 1, 42).unwrap();
        let mse_2 = TurboQuantMse::new(8, 2, 42).unwrap();

        let err_0 = mse_0
            .reconstruction_error(&vector, &mse_0.quantize(&vector).unwrap())
            .unwrap();
        let err_1 = mse_1
            .reconstruction_error(&vector, &mse_1.quantize(&vector).unwrap())
            .unwrap();
        let err_2 = mse_2
            .reconstruction_error(&vector, &mse_2.quantize(&vector).unwrap())
            .unwrap();

        assert!(err_1 <= err_0);
        assert!(err_2 <= err_1);
    }

    #[test]
    fn product_quantizer_supports_one_bit_budget() {
        let vector = unit_vector(&[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.1, 0.9]);
        let quantizer = TurboQuantProd::new(8, 1, 11).unwrap();
        let code = quantizer.quantize(&vector).unwrap();
        let decoded = quantizer.dequantize(&code).unwrap();

        assert_eq!(decoded.len(), vector.len());
        assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn implementation_uses_exact_dense_gaussian_rotation() {
        let quantizer = TurboQuantMse::new(8, 2, 5).unwrap();
        assert_eq!(quantizer.rotation_backend(), RotationBackend::DenseGaussian);
    }

    #[test]
    fn batch_quantization_matches_scalar_path() {
        let vectors = vec![
            unit_vector(&[0.3, -0.7, 0.2, 1.2, -0.8, 0.1, 0.6, -0.4]),
            unit_vector(&[-0.2, 0.4, -1.0, 0.3, 0.9, -0.6, 0.2, 0.1]),
        ];
        let quantizer = TurboQuantMse::new(8, 2, 17).unwrap();

        let batch_codes = quantizer.quantize_batch(&vectors).unwrap();
        let scalar_codes = vectors
            .iter()
            .map(|vector| quantizer.quantize(vector).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(batch_codes.len(), scalar_codes.len());
        for index in 0..batch_codes.len() {
            assert_eq!(
                batch_codes[index].unpack_indices().unwrap(),
                scalar_codes[index].unpack_indices().unwrap()
            );
            assert_eq!(
                batch_codes[index].input_norm(),
                scalar_codes[index].input_norm()
            );
        }
    }

    #[test]
    fn gaussian_rotation_is_orthogonal_in_practice() {
        let quantizer = TurboQuantMse::new(16, 2, 123).unwrap();
        let vector = unit_vector(&[
            0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6, -0.7, 0.9, -0.4, 0.3, -0.2, 0.1, 0.5, -0.8,
        ]);

        let recovered = quantizer.rotate_then_unrotate(&vector).unwrap();

        assert!(squared_l2_distance(&vector, &recovered) < 1e-20);
    }

    #[test]
    fn walsh_hadamard_rotation_is_orthogonal_in_practice() {
        let quantizer =
            TurboQuantMse::new_with_rotation_backend(16, 2, 123, RotationBackend::WalshHadamard)
                .unwrap();
        let vector = unit_vector(&[
            0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6, -0.7, 0.9, -0.4, 0.3, -0.2, 0.1, 0.5, -0.8,
        ]);

        let recovered = quantizer.rotate_then_unrotate(&vector).unwrap();

        assert!(squared_l2_distance(&vector, &recovered) < 1e-20);
    }
}
