use turboquant::ProductQuantizer;

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
fn pq_round_trip_is_finite() {
    let data = dataset();
    let pq = ProductQuantizer::train(&data, 2, 2, 4, 7).unwrap();
    let code = pq.quantize(&data[0]).unwrap();
    let decoded = pq.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), data[0].len());
    assert!(decoded.iter().all(|value| value.is_finite()));
}

#[test]
fn pq_direct_inner_product_matches_reconstructed_dot() {
    let data = dataset();
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let pq = ProductQuantizer::train(&data, 2, 2, 4, 7).unwrap();
    let code = pq.quantize(&data[0]).unwrap();
    let decoded = pq.dequantize(&code).unwrap();

    let direct = pq.approximate_inner_product(&code, &query).unwrap();
    let via_decode = decoded
        .iter()
        .zip(&query)
        .map(|(left, right)| left * right)
        .sum::<f64>();

    assert!((direct - via_decode).abs() < 1e-12);
}

#[test]
fn pq_training_is_deterministic_for_fixed_seed() {
    let data = dataset();
    let left = ProductQuantizer::train(&data, 2, 2, 4, 7).unwrap();
    let right = ProductQuantizer::train(&data, 2, 2, 4, 7).unwrap();

    let left_code = left.quantize(&data[0]).unwrap();
    let right_code = right.quantize(&data[0]).unwrap();

    assert_eq!(
        left_code.unpack_indices().unwrap(),
        right_code.unpack_indices().unwrap()
    );
}
