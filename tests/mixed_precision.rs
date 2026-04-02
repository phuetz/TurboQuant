use turboquant::{OutlierSplitPlan, TurboQuantError, TurboQuantMixedMse, TurboQuantMixedProd};

fn unit_vector(values: &[f64]) -> Vec<f64> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    values.iter().map(|value| value / norm).collect()
}

#[test]
fn split_plan_tracks_effective_bit_width() {
    let plan = OutlierSplitPlan::new(8, vec![1, 3, 5, 7], 3, 2).unwrap();

    assert_eq!(plan.outlier_indices(), &[1, 3, 5, 7]);
    assert_eq!(plan.regular_indices(), &[0, 2, 4, 6]);
    assert!((plan.effective_bit_width() - 2.5).abs() < 1e-12);
}

#[test]
fn rms_calibration_picks_high_energy_channels() {
    let samples = vec![
        vec![8.0, 0.2, 6.0, 0.1, 5.0, 0.2, 0.1, 0.1],
        vec![7.5, 0.1, 6.5, 0.2, 4.5, 0.1, 0.2, 0.1],
        vec![8.2, 0.3, 5.8, 0.1, 4.8, 0.2, 0.1, 0.2],
    ];

    let plan = OutlierSplitPlan::from_channel_rms(&samples, 3, 3, 2).unwrap();
    assert_eq!(plan.outlier_indices(), &[0, 2, 4]);
}

#[test]
fn mixed_mse_round_trip_is_finite() {
    let plan = OutlierSplitPlan::new(8, vec![0, 2, 4, 6], 3, 2).unwrap();
    let quantizer = TurboQuantMixedMse::new(plan, 7).unwrap();
    let vector = unit_vector(&[0.4, -0.3, 1.2, -0.5, 0.8, -0.1, 0.2, 0.6]);

    let code = quantizer.quantize(&vector).unwrap();
    let decoded = quantizer.dequantize(&code).unwrap();

    assert_eq!(decoded.len(), vector.len());
    assert!(decoded.iter().all(|value| value.is_finite()));
    assert!(
        quantizer
            .reconstruction_error(&vector, &code)
            .unwrap()
            .is_finite()
    );
}

#[test]
fn mixed_prod_batch_matches_scalar_path() {
    let plan = OutlierSplitPlan::new(8, vec![0, 2, 4, 6], 3, 2).unwrap();
    let quantizer = TurboQuantMixedProd::new(plan, 17).unwrap();
    let vectors = vec![
        unit_vector(&[0.25, -0.5, 0.75, -1.0, 0.5, -0.25, 0.1, 0.9]),
        unit_vector(&[-0.4, 0.2, -0.8, 0.6, 0.3, -0.9, 0.1, 0.5]),
    ];
    let queries = vec![
        unit_vector(&[0.1, -0.3, 0.5, -0.2, 0.7, -0.4, 0.2, 0.6]),
        unit_vector(&[-0.2, 0.4, -0.1, 0.8, -0.5, 0.3, -0.7, 0.2]),
    ];

    let codes = quantizer.quantize_batch(&vectors).unwrap();
    let batch = quantizer
        .estimate_inner_products_batch(&codes, &queries)
        .unwrap();
    let scalar = codes
        .iter()
        .zip(&queries)
        .map(|(code, query)| quantizer.estimate_inner_product(code, query).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(batch, scalar);
}

#[test]
fn invalid_split_plans_are_rejected() {
    let duplicate = OutlierSplitPlan::new(8, vec![1, 1, 3, 5], 3, 2).unwrap_err();
    assert!(matches!(
        duplicate,
        TurboQuantError::DuplicateChannelIndex { index: 1 }
    ));

    let singleton_partition =
        OutlierSplitPlan::new(8, vec![0, 1, 2, 3, 4, 5, 6], 3, 2).unwrap_err();
    assert!(matches!(
        singleton_partition,
        TurboQuantError::InvalidChannelPartition {
            outliers: 7,
            regular: 1
        }
    ));
}
