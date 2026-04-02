use turboquant::{RotationBackend, TurboQuantError, TurboQuantMse, TurboQuantProd};

fn unit_vector(values: &[f64]) -> Vec<f64> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    values.iter().map(|value| value / norm).collect()
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(l, r)| l * r).sum()
}

#[test]
fn zero_vector_round_trip_is_exact_for_mse_and_prod() {
    let zero = vec![0.0; 8];

    let mse = TurboQuantMse::new(8, 2, 3).unwrap();
    let mse_code = mse.quantize(&zero).unwrap();
    let mse_decoded = mse.dequantize(&mse_code).unwrap();
    assert_eq!(mse_decoded, zero);

    let prod = TurboQuantProd::new(8, 2, 3).unwrap();
    let prod_code = prod.quantize(&zero).unwrap();
    let prod_decoded = prod.dequantize(&prod_code).unwrap();
    assert_eq!(prod_decoded, zero);
}

#[test]
fn all_dimensions_use_exact_dense_gaussian_rotation() {
    let quantizer_8 = TurboQuantMse::new(8, 2, 9).unwrap();
    let quantizer_10 = TurboQuantMse::new(10, 2, 9).unwrap();

    assert_eq!(
        quantizer_8.rotation_backend(),
        RotationBackend::DenseGaussian
    );
    assert_eq!(
        quantizer_10.rotation_backend(),
        RotationBackend::DenseGaussian
    );
}

#[test]
fn walsh_hadamard_rotation_round_trips_when_requested() {
    let vector = unit_vector(&[0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6, -0.7, 0.9]);
    let mse =
        TurboQuantMse::new_with_rotation_backend(10, 2, 5, RotationBackend::WalshHadamard).unwrap();
    let code = mse.quantize(&vector).unwrap();
    let recovered = mse.dequantize(&code).unwrap();

    assert_eq!(mse.rotation_backend(), RotationBackend::WalshHadamard);
    assert_eq!(recovered.len(), vector.len());
    assert!(recovered.iter().all(|value| value.is_finite()));

    let prod = TurboQuantProd::new_with_rotation_backend(10, 2, 5, RotationBackend::WalshHadamard)
        .unwrap();
    assert_eq!(prod.rotation_backend(), RotationBackend::WalshHadamard);
}

#[test]
fn quantization_is_deterministic_for_fixed_seed() {
    let vector = unit_vector(&[0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6]);

    let left = TurboQuantMse::new(8, 2, 99).unwrap();
    let right = TurboQuantMse::new(8, 2, 99).unwrap();

    let left_code = left.quantize(&vector).unwrap();
    let right_code = right.quantize(&vector).unwrap();

    assert_eq!(
        left_code.unpack_indices().unwrap(),
        right_code.unpack_indices().unwrap()
    );
    assert_eq!(
        left.dequantize(&left_code).unwrap(),
        right.dequantize(&right_code).unwrap()
    );
}

#[test]
fn mse_batch_dequantization_matches_scalar_path() {
    let vectors = vec![
        unit_vector(&[0.3, -0.7, 0.2, 1.2, -0.8, 0.1, 0.6, -0.4]),
        unit_vector(&[-0.2, 0.4, -1.0, 0.3, 0.9, -0.6, 0.2, 0.1]),
        unit_vector(&[0.5, 0.1, -0.4, 0.8, -0.7, 0.2, -0.1, 0.6]),
    ];
    let quantizer = TurboQuantMse::new(8, 3, 21).unwrap();

    let codes = quantizer.quantize_batch(&vectors).unwrap();
    let batch_decoded = quantizer.dequantize_batch(&codes).unwrap();
    let scalar_decoded = codes
        .iter()
        .map(|code| quantizer.dequantize(code).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(batch_decoded, scalar_decoded);
}

#[test]
fn prod_batch_inner_products_match_scalar_path() {
    let vectors = vec![
        unit_vector(&[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.1, 0.9]),
        unit_vector(&[-0.4, 0.2, -0.8, 0.6, 0.3, -0.9, 0.1, 0.5]),
    ];
    let queries = vec![
        unit_vector(&[0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6]),
        unit_vector(&[-0.2, 0.4, -0.1, 0.8, -0.5, 0.3, -0.7, 0.2]),
    ];
    let quantizer = TurboQuantProd::new(8, 3, 17).unwrap();

    let codes = quantizer.quantize_batch(&vectors).unwrap();
    let batch_estimates = quantizer
        .estimate_inner_products_batch(&codes, &queries)
        .unwrap();
    let scalar_estimates = codes
        .iter()
        .zip(&queries)
        .map(|(code, query)| quantizer.estimate_inner_product(code, query).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(batch_estimates, scalar_estimates);
}

#[test]
fn prod_batch_length_mismatch_is_reported() {
    let quantizer = TurboQuantProd::new(8, 2, 5).unwrap();
    let codes = quantizer
        .quantize_batch(&[vec![0.0; 8], vec![1.0; 8]])
        .unwrap();
    let err = quantizer
        .estimate_inner_products_batch(&codes, &[vec![1.0; 8]])
        .unwrap_err();

    assert!(matches!(
        err,
        TurboQuantError::BatchLengthMismatch {
            expected: 2,
            actual: 1
        }
    ));
}

#[test]
fn prod_estimator_is_empirically_near_unbiased_across_seeds() {
    let vector = unit_vector(&[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.1, 0.9]);
    let query = unit_vector(&[0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6]);
    let expected = dot(&vector, &query);

    let mut mean_estimate = 0.0;
    let trials = 64usize;
    for seed in 0..trials as u64 {
        let quantizer = TurboQuantProd::new(8, 2, seed).unwrap();
        let code = quantizer.quantize(&vector).unwrap();
        mean_estimate += quantizer.estimate_inner_product(&code, &query).unwrap();
    }
    mean_estimate /= trials as f64;

    assert!((mean_estimate - expected).abs() < 0.15);
}
