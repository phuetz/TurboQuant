use turboquant::{
    KvQuantizerSpec, KvTensor, KvTensorShape, QuantizedKvCacheLayer, TurboQuantKvCache,
    TurboQuantKvCacheConfig, analyze_kv_layer_norms, calibrate_skip_layers,
};

fn sequential_tensor(shape: KvTensorShape, start: f64) -> KvTensor {
    let values = (0..shape.element_count())
        .map(|index| start + index as f64)
        .collect::<Vec<_>>();
    KvTensor::new(shape, values).unwrap()
}

#[test]
fn kv_tensor_concat_and_slice_round_trip() {
    let prefix = sequential_tensor(KvTensorShape::new(1, 2, 2, 4).unwrap(), 0.0);
    let suffix = sequential_tensor(KvTensorShape::new(1, 2, 1, 4).unwrap(), 100.0);

    let combined = prefix.concat_seq(&suffix).unwrap();
    assert_eq!(combined.shape().sequence_length(), 3);
    assert_eq!(combined.slice_seq(0, 2).unwrap(), prefix);
    assert_eq!(combined.slice_seq(2, 3).unwrap(), suffix);
}

#[test]
fn kv_tensor_quantizers_preserve_shape() {
    let tensor = sequential_tensor(KvTensorShape::new(1, 2, 3, 8).unwrap(), 1.0);

    let mse = turboquant::KvMseQuantizer::new(8, 2, 7).unwrap();
    let mse_code = mse.quantize_tensor(&tensor).unwrap();
    let mse_decoded = mse.dequantize_tensor(&mse_code).unwrap();
    assert_eq!(mse_decoded.shape(), tensor.shape());

    let prod = turboquant::KvProdQuantizer::new(8, 3, 11).unwrap();
    let prod_code = prod.quantize_tensor(&tensor).unwrap();
    let prod_decoded = prod.dequantize_tensor(&prod_code).unwrap();
    assert_eq!(prod_decoded.shape(), tensor.shape());

    let plan = turboquant::OutlierSplitPlan::new(8, vec![0, 1], 4, 2).unwrap();
    let mixed = turboquant::KvMixedMseQuantizer::new(plan, 13).unwrap();
    let mixed_code = mixed.quantize_tensor(&tensor).unwrap();
    let mixed_decoded = mixed.dequantize_tensor(&mixed_code).unwrap();
    assert_eq!(mixed_decoded.shape(), tensor.shape());
}

#[test]
fn quantized_kv_cache_layer_keeps_residual_window() {
    let key_quantizer = KvQuantizerSpec::Mse { bit_width: 2 }.build(8, 21).unwrap();
    let value_quantizer = KvQuantizerSpec::Mse { bit_width: 2 }.build(8, 42).unwrap();
    let mut layer = QuantizedKvCacheLayer::new(1, 2, 8, 2, key_quantizer, value_quantizer).unwrap();

    let prefill_keys = sequential_tensor(KvTensorShape::new(1, 2, 3, 8).unwrap(), 0.0);
    let prefill_values = sequential_tensor(KvTensorShape::new(1, 2, 3, 8).unwrap(), 50.0);
    layer.update(&prefill_keys, &prefill_values).unwrap();

    assert_eq!(layer.seq_length(), 3);
    assert_eq!(layer.quantized_prefix_length(), 1);

    let decode_keys = sequential_tensor(KvTensorShape::new(1, 2, 1, 8).unwrap(), 200.0);
    let decode_values = sequential_tensor(KvTensorShape::new(1, 2, 1, 8).unwrap(), 300.0);
    layer.update(&decode_keys, &decode_values).unwrap();

    assert_eq!(layer.seq_length(), 4);
    assert_eq!(layer.quantized_prefix_length(), 2);
    assert_eq!(
        layer.materialize_keys().unwrap().shape().sequence_length(),
        4
    );
    assert_eq!(
        layer
            .materialize_values()
            .unwrap()
            .shape()
            .sequence_length(),
        4
    );
}

#[test]
fn multi_layer_kv_cache_respects_skip_layers() {
    let mut cache = TurboQuantKvCache::new(TurboQuantKvCacheConfig {
        num_layers: 3,
        batch_size: 1,
        kv_heads: 2,
        head_dim: 8,
        residual_length: 2,
        key_spec: KvQuantizerSpec::FastProd { bit_width: 3 },
        value_spec: KvQuantizerSpec::FastMse { bit_width: 2 },
        seed: 7,
        skip_layers: vec![1],
    })
    .unwrap();

    let keys = sequential_tensor(KvTensorShape::new(1, 2, 16, 8).unwrap(), 0.0);
    let values = sequential_tensor(KvTensorShape::new(1, 2, 16, 8).unwrap(), 100.0);
    for layer_index in 0..cache.layer_count() {
        cache.update(layer_index, &keys, &values).unwrap();
    }

    let full_precision_bytes = 3 * 2 * keys.storage_bytes();
    assert!(cache.storage_bytes() < full_precision_bytes);
    assert!(cache.layer(0).unwrap().is_quantized());
    assert!(!cache.layer(1).unwrap().is_quantized());
    assert!(cache.layer(2).unwrap().is_quantized());

    let (layer1_keys, layer1_values) = cache.materialize_layer(1).unwrap();
    assert_eq!(layer1_keys.shape(), keys.shape());
    assert_eq!(layer1_values.shape(), values.shape());
}

#[test]
fn kv_layer_norm_analysis_detects_outliers() {
    let normal = sequential_tensor(KvTensorShape::new(1, 1, 2, 8).unwrap(), 1.0);
    let outlier = KvTensor::new(
        KvTensorShape::new(1, 1, 2, 8).unwrap(),
        normal.values().iter().map(|value| value * 25.0).collect(),
    )
    .unwrap();

    let analysis =
        analyze_kv_layer_norms(&[normal.clone(), outlier.clone(), normal.clone()], 3.0).unwrap();
    assert_eq!(analysis.max_norm_layer(), 1);
    assert_eq!(analysis.skip_layers(), &[1]);

    let skip_layers =
        calibrate_skip_layers(&[normal.clone(), normal, outlier.clone()], 2.0).unwrap();
    assert_eq!(skip_layers, vec![2]);
}
