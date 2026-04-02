//! Stress tests for TurboQuant: edge cases, high dimensions, large batches,
//! statistical properties, and error handling.

use turboquant::{
    KvQuantizerSpec, KvTensor, KvTensorShape, OutlierSplitPlan, PackedBits, QuantizedKvCacheLayer,
    TurboQuantError, TurboQuantKvCache, TurboQuantKvCacheConfig, TurboQuantMixedMse,
    TurboQuantMixedProd, TurboQuantMse, TurboQuantProd,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unit_vector(values: &[f64]) -> Vec<f64> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    values.iter().map(|value| value / norm).collect()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(l, r)| l * r).sum()
}

/// Generate a deterministic pseudo-random unit vector of a given dimension.
fn random_unit_vector(dimension: usize, seed: u64) -> Vec<f64> {
    // Simple LCG to avoid pulling in rand in the test harness.
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let raw: Vec<f64> = (0..dimension)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1, 1].
            (state as i64 as f64) / (i64::MAX as f64)
        })
        .collect();
    unit_vector(&raw)
}

fn sequential_tensor(shape: KvTensorShape, start: f64) -> KvTensor {
    let values = (0..shape.element_count())
        .map(|index| start + index as f64)
        .collect::<Vec<_>>();
    KvTensor::new(shape, values).unwrap()
}

// ===========================================================================
// 1. High dimensions
// ===========================================================================

#[test]
fn high_dimension_mse_round_trip_d512() {
    for bit_width in [1_u8, 2, 3, 4] {
        let dimension = 512;
        let vector = random_unit_vector(dimension, 100 + bit_width as u64);
        let quantizer = TurboQuantMse::new(dimension, bit_width, 7).unwrap();
        let code = quantizer.quantize(&vector).unwrap();
        let decoded = quantizer.dequantize(&code).unwrap();

        assert_eq!(decoded.len(), dimension);
        assert!(decoded.iter().all(|v| v.is_finite()));

        let error = quantizer.reconstruction_error(&vector, &code).unwrap();
        assert!(error.is_finite());
        assert!(error < 2.0, "MSE too high for d={dimension} b={bit_width}: {error}");
    }
}

#[test]
fn high_dimension_prod_round_trip_d512() {
    for bit_width in [1_u8, 2, 3, 4] {
        let dimension = 512;
        let vector = random_unit_vector(dimension, 200 + bit_width as u64);
        let quantizer = TurboQuantProd::new(dimension, bit_width, 13).unwrap();
        let code = quantizer.quantize(&vector).unwrap();
        let decoded = quantizer.dequantize(&code).unwrap();

        assert_eq!(decoded.len(), dimension);
        assert!(decoded.iter().all(|v| v.is_finite()));
    }
}

// ===========================================================================
// 2. Extreme bit widths
// ===========================================================================

#[test]
fn extreme_bit_width_mse_b8() {
    let vector = random_unit_vector(32, 301);
    let quantizer = TurboQuantMse::new(32, 8, 42).unwrap();
    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), 32);
    assert!(decoded.iter().all(|v| v.is_finite()));
}

#[test]
fn extreme_bit_width_mse_b12() {
    let vector = random_unit_vector(16, 302);
    let quantizer = TurboQuantMse::new(16, 12, 42).unwrap();
    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), 16);
    assert!(decoded.iter().all(|v| v.is_finite()));
}

#[test]
fn extreme_bit_width_mse_b14() {
    let vector = random_unit_vector(4, 303);
    let quantizer = TurboQuantMse::new(4, 14, 42).unwrap();
    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), 4);
    assert!(decoded.iter().all(|v| v.is_finite()));

    // At 14 bits the reconstruction should be nearly perfect for a small vector.
    let error = quantizer.reconstruction_error(&vector, &code).unwrap();
    assert!(error < 1e-5, "b=14 reconstruction error unexpectedly high: {error}");
}

#[test]
fn extreme_bit_width_prod_b8() {
    let vector = random_unit_vector(32, 304);
    let quantizer = TurboQuantProd::new(32, 8, 42).unwrap();
    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), 32);
    assert!(decoded.iter().all(|v| v.is_finite()));
}

// ===========================================================================
// 3. Large batch quantization
// ===========================================================================

#[test]
fn large_batch_quantization_10k_vectors() {
    let dimension = 128;
    let batch_size = 10_000;
    let vectors: Vec<Vec<f64>> = (0..batch_size)
        .map(|index| random_unit_vector(dimension, index as u64))
        .collect();

    let quantizer = TurboQuantMse::new(dimension, 2, 77).unwrap();
    let codes = quantizer.quantize_batch(&vectors).unwrap();
    assert_eq!(codes.len(), batch_size);

    let decoded = quantizer.dequantize_batch(&codes).unwrap();
    assert_eq!(decoded.len(), batch_size);
    for vector in &decoded {
        assert_eq!(vector.len(), dimension);
        assert!(vector.iter().all(|v| v.is_finite()));
    }
}

// ===========================================================================
// 4. MSE monotonicity
// ===========================================================================

#[test]
fn mse_monotonically_decreases_with_bits() {
    for dimension in [64, 128, 256, 512] {
        let vector = random_unit_vector(dimension, dimension as u64);
        let mut previous_error = f64::INFINITY;

        for bit_width in 1..=4_u8 {
            let quantizer = TurboQuantMse::new(dimension, bit_width, 42).unwrap();
            let code = quantizer.quantize(&vector).unwrap();
            let error = quantizer.reconstruction_error(&vector, &code).unwrap();

            assert!(
                error <= previous_error,
                "MSE did not decrease: d={dimension} b={bit_width} error={error} >= prev={previous_error}"
            );
            previous_error = error;
        }
    }
}

// ===========================================================================
// 5. Prod unbiasedness
// ===========================================================================

#[test]
fn prod_estimator_unbiased_over_1000_random_pairs() {
    let dimension = 32;
    let trials = 1000_usize;
    let mut total_bias = 0.0;

    for trial in 0..trials {
        let vector = random_unit_vector(dimension, trial as u64 * 2);
        let query = random_unit_vector(dimension, trial as u64 * 2 + 1);
        let expected = dot(&vector, &query);

        let quantizer = TurboQuantProd::new(dimension, 3, trial as u64).unwrap();
        let code = quantizer.quantize(&vector).unwrap();
        let estimate = quantizer.estimate_inner_product(&code, &query).unwrap();

        total_bias += estimate - expected;
    }

    let mean_bias = total_bias / trials as f64;
    assert!(
        mean_bias.abs() < 0.05,
        "mean bias {mean_bias} exceeds threshold 0.05 over {trials} trials"
    );
}

// ===========================================================================
// 6. KV cache stress
// ===========================================================================

#[test]
fn kv_cache_large_sequence_length() {
    let head_dim = 64;
    let prefill_seq = 4096;
    let decode_seq = 256;

    let key_quantizer = KvQuantizerSpec::FastMse { bit_width: 2 }
        .build(head_dim, 21)
        .unwrap();
    let value_quantizer = KvQuantizerSpec::FastProd { bit_width: 2 }
        .build(head_dim, 42)
        .unwrap();
    let mut layer =
        QuantizedKvCacheLayer::new(1, 2, head_dim, 64, key_quantizer, value_quantizer).unwrap();

    let prefill_keys =
        sequential_tensor(KvTensorShape::new(1, 2, prefill_seq, head_dim).unwrap(), 0.0);
    let prefill_values = sequential_tensor(
        KvTensorShape::new(1, 2, prefill_seq, head_dim).unwrap(),
        1e6,
    );
    layer.update(&prefill_keys, &prefill_values).unwrap();
    assert_eq!(layer.seq_length(), prefill_seq);

    let decode_keys =
        sequential_tensor(KvTensorShape::new(1, 2, decode_seq, head_dim).unwrap(), 2e6);
    let decode_values =
        sequential_tensor(KvTensorShape::new(1, 2, decode_seq, head_dim).unwrap(), 3e6);
    layer.update(&decode_keys, &decode_values).unwrap();
    assert_eq!(layer.seq_length(), prefill_seq + decode_seq);

    let materialized_keys = layer.materialize_keys().unwrap();
    assert_eq!(
        materialized_keys.shape().sequence_length(),
        prefill_seq + decode_seq
    );
}

#[test]
fn kv_cache_many_layers() {
    let num_layers = 64;
    let head_dim = 8;

    let mut cache = TurboQuantKvCache::new(TurboQuantKvCacheConfig {
        num_layers,
        batch_size: 1,
        kv_heads: 2,
        head_dim,
        residual_length: 2,
        key_spec: KvQuantizerSpec::FastMse { bit_width: 2 },
        value_spec: KvQuantizerSpec::FastProd { bit_width: 2 },
        seed: 99,
        skip_layers: vec![],
    })
    .unwrap();

    let keys = sequential_tensor(KvTensorShape::new(1, 2, 8, head_dim).unwrap(), 0.0);
    let values = sequential_tensor(KvTensorShape::new(1, 2, 8, head_dim).unwrap(), 100.0);

    for layer_index in 0..num_layers {
        cache.update(layer_index, &keys, &values).unwrap();
    }

    assert_eq!(cache.layer_count(), num_layers);
    for layer_index in 0..num_layers {
        assert!(cache.layer(layer_index).unwrap().is_quantized());
    }
}

#[test]
fn kv_cache_all_quantizer_specs() {
    let head_dim = 8;
    let plan = OutlierSplitPlan::new(head_dim, vec![0, 1, 2, 3], 3, 2).unwrap();

    let specs: Vec<(KvQuantizerSpec, KvQuantizerSpec)> = vec![
        (
            KvQuantizerSpec::Mse { bit_width: 2 },
            KvQuantizerSpec::Mse { bit_width: 2 },
        ),
        (
            KvQuantizerSpec::Prod { bit_width: 3 },
            KvQuantizerSpec::Prod { bit_width: 3 },
        ),
        (
            KvQuantizerSpec::FastMse { bit_width: 2 },
            KvQuantizerSpec::FastMse { bit_width: 2 },
        ),
        (
            KvQuantizerSpec::FastProd { bit_width: 3 },
            KvQuantizerSpec::FastProd { bit_width: 3 },
        ),
        (
            KvQuantizerSpec::MixedMse {
                plan: plan.clone(),
            },
            KvQuantizerSpec::MixedMse {
                plan: plan.clone(),
            },
        ),
        (
            KvQuantizerSpec::MixedProd {
                plan: plan.clone(),
            },
            KvQuantizerSpec::MixedProd {
                plan: plan.clone(),
            },
        ),
    ];

    for (key_spec, value_spec) in specs {
        let mut cache = TurboQuantKvCache::new(TurboQuantKvCacheConfig {
            num_layers: 2,
            batch_size: 1,
            kv_heads: 2,
            head_dim,
            residual_length: 2,
            key_spec,
            value_spec,
            seed: 7,
            skip_layers: vec![],
        })
        .unwrap();

        let keys = sequential_tensor(KvTensorShape::new(1, 2, 4, head_dim).unwrap(), 0.0);
        let values = sequential_tensor(KvTensorShape::new(1, 2, 4, head_dim).unwrap(), 50.0);

        for layer_index in 0..2 {
            cache.update(layer_index, &keys, &values).unwrap();
        }

        for layer_index in 0..2 {
            let (mat_keys, mat_values) = cache.materialize_layer(layer_index).unwrap();
            assert_eq!(mat_keys.shape().sequence_length(), 4);
            assert_eq!(mat_values.shape().sequence_length(), 4);
        }
    }
}

// ===========================================================================
// 7. Mixed precision edge cases
// ===========================================================================

#[test]
fn mixed_mse_minimum_outliers() {
    // 2 outliers, 6 regular -- the minimum non-empty partition size is 2.
    let plan = OutlierSplitPlan::new(8, vec![0, 7], 4, 2).unwrap();
    let quantizer = TurboQuantMixedMse::new(plan, 31).unwrap();
    let vector = unit_vector(&[0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6]);

    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), 8);
    assert!(decoded.iter().all(|v| v.is_finite()));
}

#[test]
fn mixed_prod_asymmetric_bit_widths() {
    // outlier = 8 bits, regular = 1 bit -- very asymmetric.
    let plan = OutlierSplitPlan::new(8, vec![0, 1, 2, 3], 8, 1).unwrap();
    let quantizer = TurboQuantMixedProd::new(plan, 17).unwrap();
    let vector = unit_vector(&[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.1, 0.9]);
    let query = unit_vector(&[0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6]);

    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();
    let estimate = quantizer.estimate_inner_product(&code, &query).unwrap();

    assert_eq!(decoded.len(), 8);
    assert!(decoded.iter().all(|v| v.is_finite()));
    assert!(estimate.is_finite());
}

#[test]
fn mixed_mse_maximum_outliers() {
    // All outliers (6 outlier, 2 regular -- maximum while keeping both partitions >= 2).
    let plan = OutlierSplitPlan::new(8, vec![0, 1, 2, 3, 4, 5], 4, 2).unwrap();
    let quantizer = TurboQuantMixedMse::new(plan, 41).unwrap();
    let vector = unit_vector(&[0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6]);

    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), 8);
    assert!(decoded.iter().all(|v| v.is_finite()));
    assert!(
        quantizer
            .reconstruction_error(&vector, &code)
            .unwrap()
            .is_finite()
    );
}

// ===========================================================================
// 8. Determinism
// ===========================================================================

#[test]
fn determinism_mse_100_runs_same_seed() {
    let dimension = 64;
    let bit_width = 3;
    let seed = 12345_u64;
    let vector = random_unit_vector(dimension, 999);

    let reference_quantizer = TurboQuantMse::new(dimension, bit_width, seed).unwrap();
    let reference_code = reference_quantizer.quantize(&vector).unwrap();
    let reference_indices = reference_code.unpack_indices().unwrap();
    let reference_decoded = reference_quantizer.dequantize(&reference_code).unwrap();

    for _ in 0..100 {
        let quantizer = TurboQuantMse::new(dimension, bit_width, seed).unwrap();
        let code = quantizer.quantize(&vector).unwrap();

        assert_eq!(code.unpack_indices().unwrap(), reference_indices);
        assert_eq!(code.input_norm(), reference_code.input_norm());
        assert_eq!(quantizer.dequantize(&code).unwrap(), reference_decoded);
    }
}

#[test]
fn determinism_prod_100_runs_same_seed() {
    let dimension = 32;
    let bit_width = 2;
    let seed = 54321_u64;
    let vector = random_unit_vector(dimension, 888);

    let reference_quantizer = TurboQuantProd::new(dimension, bit_width, seed).unwrap();
    let reference_code = reference_quantizer.quantize(&vector).unwrap();
    let reference_mse_indices = reference_code.unpack_mse_indices().unwrap();
    let reference_qjl_signs = reference_code.unpack_qjl_signs().unwrap();
    let reference_decoded = reference_quantizer.dequantize(&reference_code).unwrap();

    for _ in 0..100 {
        let quantizer = TurboQuantProd::new(dimension, bit_width, seed).unwrap();
        let code = quantizer.quantize(&vector).unwrap();

        assert_eq!(code.unpack_mse_indices().unwrap(), reference_mse_indices);
        assert_eq!(code.unpack_qjl_signs().unwrap(), reference_qjl_signs);
        assert_eq!(code.input_norm(), reference_code.input_norm());
        assert_eq!(code.residual_norm(), reference_code.residual_norm());
        assert_eq!(quantizer.dequantize(&code).unwrap(), reference_decoded);
    }
}

// ===========================================================================
// 9. PackedBits stress
// ===========================================================================

#[test]
fn packed_bits_high_bit_width_b16() {
    let symbol_count = 100;
    let bits_per_symbol = 16_u8;
    let max_value = (1_u64 << bits_per_symbol) - 1;

    let values: Vec<u64> = (0..symbol_count)
        .map(|index| (index as u64 * 659) % (max_value + 1))
        .collect();

    let packed = PackedBits::pack_values(&values, bits_per_symbol).unwrap();
    assert_eq!(packed.symbol_count(), symbol_count);
    assert_eq!(packed.bits_per_symbol(), bits_per_symbol);

    let unpacked = packed.unpack_values().unwrap();
    assert_eq!(unpacked, values);
}

#[test]
fn packed_bits_high_bit_width_b20() {
    let symbol_count = 200;
    let bits_per_symbol = 20_u8;
    let max_value = (1_u64 << bits_per_symbol) - 1;

    let values: Vec<u64> = (0..symbol_count)
        .map(|index| (index as u64 * 104729) % (max_value + 1))
        .collect();

    let packed = PackedBits::pack_values(&values, bits_per_symbol).unwrap();
    assert_eq!(packed.symbol_count(), symbol_count);
    assert_eq!(packed.bits_per_symbol(), bits_per_symbol);

    let unpacked = packed.unpack_values().unwrap();
    assert_eq!(unpacked, values);
}

#[test]
fn packed_bits_large_symbol_count() {
    let symbol_count = 100_000;
    let bits_per_symbol = 3_u8;

    let values: Vec<u64> = (0..symbol_count)
        .map(|index| (index as u64) % (1 << bits_per_symbol))
        .collect();

    let packed = PackedBits::pack_values(&values, bits_per_symbol).unwrap();
    assert_eq!(packed.symbol_count(), symbol_count);

    let unpacked = packed.unpack_values().unwrap();
    assert_eq!(unpacked, values);
}

#[test]
fn packed_bits_boundary_values() {
    // Pack the maximum representable value at each bit width.
    for bits in [1_u8, 2, 4, 8, 16, 20, 32, 63] {
        let max_value = (1_u64 << bits) - 1;
        let values = vec![max_value; 10];
        let packed = PackedBits::pack_values(&values, bits).unwrap();
        let unpacked = packed.unpack_values().unwrap();
        assert_eq!(unpacked, values, "round-trip failed for bits={bits}");
    }
}

// ===========================================================================
// 10. Error handling
// ===========================================================================

#[test]
fn error_invalid_dimension() {
    let result = TurboQuantMse::new(0, 2, 1);
    assert!(matches!(result, Err(TurboQuantError::InvalidDimension(0))));

    let result = TurboQuantMse::new(1, 2, 1);
    assert!(matches!(result, Err(TurboQuantError::InvalidDimension(1))));

    let result = TurboQuantProd::new(0, 2, 1);
    assert!(matches!(result, Err(TurboQuantError::InvalidDimension(0))));

    let result = TurboQuantProd::new(1, 2, 1);
    assert!(matches!(result, Err(TurboQuantError::InvalidDimension(1))));
}

#[test]
fn error_unsupported_bit_width() {
    let result = TurboQuantMse::new(8, 21, 1);
    assert!(matches!(
        result,
        Err(TurboQuantError::UnsupportedBitWidth(21))
    ));

    let result = TurboQuantMse::new(8, 64, 1);
    assert!(matches!(
        result,
        Err(TurboQuantError::UnsupportedBitWidth(64))
    ));
}

#[test]
fn error_dimension_mismatch() {
    let quantizer = TurboQuantMse::new(8, 2, 1).unwrap();
    let wrong_length = vec![1.0; 16];
    let result = quantizer.quantize(&wrong_length);
    assert!(matches!(
        result,
        Err(TurboQuantError::DimensionMismatch {
            expected: 8,
            actual: 16
        })
    ));
}

#[test]
fn error_prod_bit_width_too_small() {
    let result = TurboQuantProd::new(8, 0, 1);
    assert!(matches!(result, Err(TurboQuantError::ProdBitWidthTooSmall)));
}

#[test]
fn error_batch_length_mismatch() {
    let quantizer = TurboQuantProd::new(8, 2, 5).unwrap();
    let codes = quantizer
        .quantize_batch(&[vec![0.0; 8], vec![0.0; 8]])
        .unwrap();
    let result = quantizer.estimate_inner_products_batch(&codes, &[vec![1.0; 8]]);
    assert!(matches!(
        result,
        Err(TurboQuantError::BatchLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn error_packed_value_out_of_range() {
    let result = PackedBits::pack_values(&[8], 3);
    assert!(matches!(
        result,
        Err(TurboQuantError::PackedValueOutOfRange {
            value: 8,
            bits_per_symbol: 3
        })
    ));
}

#[test]
fn error_packed_index_out_of_range() {
    let packed = PackedBits::pack_values(&[1, 2, 3], 4).unwrap();
    let result = packed.get(3);
    assert!(matches!(
        result,
        Err(TurboQuantError::PackedIndexOutOfRange {
            index: 3,
            symbol_count: 3
        })
    ));
}

#[test]
fn error_packed_unsupported_bit_width() {
    let result = PackedBits::zeros(10, 64);
    assert!(matches!(
        result,
        Err(TurboQuantError::UnsupportedBitWidth(64))
    ));
}

#[test]
fn error_duplicate_channel_index() {
    let result = OutlierSplitPlan::new(8, vec![1, 1], 3, 2);
    assert!(matches!(
        result,
        Err(TurboQuantError::DuplicateChannelIndex { index: 1 })
    ));
}

#[test]
fn error_invalid_channel_index() {
    let result = OutlierSplitPlan::new(8, vec![0, 10], 3, 2);
    assert!(matches!(
        result,
        Err(TurboQuantError::InvalidChannelIndex {
            index: 10,
            dimension: 8
        })
    ));
}

#[test]
fn error_invalid_channel_partition_singleton() {
    // 7 outliers + 1 regular = singleton regular partition.
    let result = OutlierSplitPlan::new(8, vec![0, 1, 2, 3, 4, 5, 6], 3, 2);
    assert!(matches!(
        result,
        Err(TurboQuantError::InvalidChannelPartition {
            outliers: 7,
            regular: 1
        })
    ));

    // 1 outlier + 7 regular = singleton outlier partition.
    let result = OutlierSplitPlan::new(8, vec![3], 3, 2);
    assert!(matches!(
        result,
        Err(TurboQuantError::InvalidChannelPartition {
            outliers: 1,
            regular: 7
        })
    ));
}

#[test]
fn error_kv_tensor_shape_zero_axes() {
    assert!(matches!(
        KvTensorShape::new(0, 2, 4, 8),
        Err(TurboQuantError::InvalidTensorShape {
            axis: "batch_size",
            value: 0
        })
    ));
    assert!(matches!(
        KvTensorShape::new(1, 0, 4, 8),
        Err(TurboQuantError::InvalidTensorShape {
            axis: "kv_heads",
            value: 0
        })
    ));
    assert!(matches!(
        KvTensorShape::new(1, 2, 4, 0),
        Err(TurboQuantError::InvalidTensorShape {
            axis: "head_dim",
            value: 0
        })
    ));
}

#[test]
fn error_kv_tensor_element_count_mismatch() {
    let shape = KvTensorShape::new(1, 2, 3, 8).unwrap();
    let result = KvTensor::new(shape, vec![0.0; 10]);
    assert!(matches!(
        result,
        Err(TurboQuantError::TensorElementCountMismatch { .. })
    ));
}

#[test]
fn error_mixed_batch_length_mismatch() {
    let plan = OutlierSplitPlan::new(8, vec![0, 1, 2, 3], 3, 2).unwrap();
    let quantizer = TurboQuantMixedProd::new(plan, 17).unwrap();
    let codes = quantizer
        .quantize_batch(&[
            unit_vector(&[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.1, 0.9]),
            unit_vector(&[-0.4, 0.2, -0.8, 0.6, 0.3, -0.9, 0.1, 0.5]),
        ])
        .unwrap();

    let result = quantizer.estimate_inner_products_batch(
        &codes,
        &[unit_vector(&[0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6])],
    );
    assert!(matches!(
        result,
        Err(TurboQuantError::BatchLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
}
