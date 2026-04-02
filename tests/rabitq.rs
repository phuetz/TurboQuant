use turboquant::{RaBitQQuantizer, RotationBackend};

fn dataset() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.9, 0.1, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.9, 0.1],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.1, 0.0, 0.0, 0.9],
    ]
}

#[test]
fn rabitq_round_trip_is_finite() {
    let data = dataset();
    let quantizer = RaBitQQuantizer::train(&data, 2, 7).unwrap();
    let code = quantizer.quantize(&data[0]).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), data[0].len());
    assert!(decoded.iter().all(|value| value.is_finite()));
}

#[test]
fn rabitq_direct_inner_product_matches_reconstructed_dot() {
    let data = dataset();
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let quantizer = RaBitQQuantizer::train(&data, 2, 7).unwrap();
    let code = quantizer.quantize(&data[0]).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    let direct = quantizer.approximate_inner_product(&code, &query).unwrap();
    let via_decode = decoded
        .iter()
        .zip(&query)
        .map(|(left, right)| left * right)
        .sum::<f64>();

    assert!((direct - via_decode).abs() < 1e-12);
}

#[test]
fn rabitq_training_is_deterministic_for_fixed_seed() {
    let data = dataset();
    let left = RaBitQQuantizer::train(&data, 2, 7).unwrap();
    let right = RaBitQQuantizer::train(&data, 2, 7).unwrap();

    let left_code = left.quantize(&data[0]).unwrap();
    let right_code = right.quantize(&data[0]).unwrap();

    assert_eq!(
        left_code.unpack_indices().unwrap(),
        right_code.unpack_indices().unwrap()
    );
    assert!((left_code.delta() - right_code.delta()).abs() < 1e-12);
}

#[test]
fn rabitq_uses_dense_gaussian_rotation() {
    let quantizer = RaBitQQuantizer::new(8, 2, 9).unwrap();
    assert_eq!(quantizer.rotation_backend(), RotationBackend::DenseGaussian);
}
