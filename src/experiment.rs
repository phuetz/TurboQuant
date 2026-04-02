use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;
use std::time::Instant;

use crate::error::{Result, TurboQuantError};
use crate::math::{dot, validate_dimension};
use crate::{
    OutlierSplitPlan, ProductQuantizer, RaBitQQuantizer, TurboQuantMixedProd, TurboQuantMse,
    TurboQuantProd,
};

#[derive(Debug, Clone, Copy)]
pub struct MseMetrics {
    pub average_mse: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ProdMetrics {
    pub bias: f64,
    pub variance: f64,
    pub mse: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RecallMetrics {
    pub recall_at_k: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RecallPoint {
    pub k: usize,
    pub recall_at_1: f64,
}

#[derive(Debug, Clone)]
pub struct RecallCurveMetrics {
    pub points: Vec<RecallPoint>,
    pub indexing_seconds: f64,
    pub query_seconds: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundMetrics {
    pub measured: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub small_bit_reference: Option<f64>,
}

pub fn evaluate_mse(dimension: usize, bits: u8, samples: usize, seed: u64) -> Result<MseMetrics> {
    validate_non_zero("samples", samples)?;
    let quantizer = TurboQuantMse::new(dimension, bits, seed)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC6A4_A793_5BD1_E995);
    let mut total_mse = 0.0;

    for _ in 0..samples {
        let vector = random_unit_vector(dimension, &mut rng);
        let code = quantizer.quantize(&vector)?;
        total_mse += quantizer.reconstruction_error(&vector, &code)?;
    }

    Ok(MseMetrics {
        average_mse: total_mse / samples as f64,
    })
}

pub fn evaluate_prod(dimension: usize, bits: u8, samples: usize, seed: u64) -> Result<ProdMetrics> {
    validate_non_zero("samples", samples)?;
    let quantizer = TurboQuantProd::new(dimension, bits, seed)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut sum_error = 0.0;
    let mut sum_sq_error = 0.0;

    for _ in 0..samples {
        let vector = random_unit_vector(dimension, &mut rng);
        let query = random_unit_vector(dimension, &mut rng);
        let code = quantizer.quantize(&vector)?;
        let estimate = quantizer.estimate_inner_product(&code, &query)?;
        let exact = dot(&vector, &query);
        let error = estimate - exact;
        sum_error += error;
        sum_sq_error += error * error;
    }

    let bias = sum_error / samples as f64;
    let mse = sum_sq_error / samples as f64;
    let variance = mse - bias * bias;

    Ok(ProdMetrics {
        bias,
        variance,
        mse,
    })
}

pub fn evaluate_recall(
    dimension: usize,
    bits: u8,
    dataset_size: usize,
    query_count: usize,
    top_k: usize,
    seed: u64,
) -> Result<RecallMetrics> {
    validate_non_zero("dataset_size", dataset_size)?;
    validate_non_zero("query_count", query_count)?;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_D192_ED03);
    let dataset = (0..dataset_size)
        .map(|_| random_unit_vector(dimension, &mut rng))
        .collect::<Vec<_>>();
    let queries = (0..query_count)
        .map(|_| random_unit_vector(dimension, &mut rng))
        .collect::<Vec<_>>();
    let curve = evaluate_recall_curve_dataset(&dataset, &queries, bits, &[top_k], seed)?;

    Ok(RecallMetrics {
        recall_at_k: curve.points[0].recall_at_1,
    })
}

pub fn evaluate_recall_curve(
    dimension: usize,
    bits: u8,
    dataset_size: usize,
    query_count: usize,
    ks: &[usize],
    seed: u64,
) -> Result<RecallCurveMetrics> {
    validate_non_zero("dataset_size", dataset_size)?;
    validate_non_zero("query_count", query_count)?;
    validate_ks(ks, dataset_size)?;

    let mut rng = StdRng::seed_from_u64(seed ^ 0xD1B5_4A32_D192_ED03);
    let dataset = (0..dataset_size)
        .map(|_| random_unit_vector(dimension, &mut rng))
        .collect::<Vec<_>>();
    let queries = (0..query_count)
        .map(|_| random_unit_vector(dimension, &mut rng))
        .collect::<Vec<_>>();

    evaluate_recall_curve_dataset(&dataset, &queries, bits, ks, seed)
}

pub fn evaluate_recall_curve_dataset(
    dataset: &[Vec<f64>],
    queries: &[Vec<f64>],
    bits: u8,
    ks: &[usize],
    seed: u64,
) -> Result<RecallCurveMetrics> {
    let dimension = validate_dataset_and_queries(dataset, queries, ks)?;

    let quantizer = TurboQuantProd::new(dimension, bits, seed)?;
    let indexing_start = Instant::now();
    let codes = dataset
        .iter()
        .map(|vector| quantizer.quantize(vector))
        .collect::<Result<Vec<_>>>()?;
    let indexing_seconds = indexing_start.elapsed().as_secs_f64();

    let max_k = *ks.iter().max().expect("validated non-empty ks");
    let query_start = Instant::now();
    let mut hits = vec![0usize; ks.len()];
    for query in queries {
        let exact_best_index = exact_top1_index(dataset, query);

        let mut approx_scores = codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                quantizer
                    .estimate_inner_product(code, query)
                    .map(|score| (index, score))
            })
            .collect::<Result<Vec<_>>>()?;
        update_hits_from_scores(&mut approx_scores, exact_best_index, ks, max_k, &mut hits);
    }
    let query_seconds = query_start.elapsed().as_secs_f64();

    Ok(build_recall_curve_metrics(
        ks,
        queries.len(),
        hits,
        indexing_seconds,
        query_seconds,
    ))
}

pub fn evaluate_mixed_recall_curve_dataset(
    dataset: &[Vec<f64>],
    queries: &[Vec<f64>],
    plan: &OutlierSplitPlan,
    ks: &[usize],
    seed: u64,
) -> Result<RecallCurveMetrics> {
    validate_dataset_and_queries(dataset, queries, ks)?;
    let quantizer = TurboQuantMixedProd::new(plan.clone(), seed)?;

    let indexing_start = Instant::now();
    let codes = quantizer.quantize_batch(dataset)?;
    let indexing_seconds = indexing_start.elapsed().as_secs_f64();

    let max_k = *ks.iter().max().expect("validated non-empty ks");
    let query_start = Instant::now();
    let mut hits = vec![0usize; ks.len()];
    for query in queries {
        let exact_best_index = exact_top1_index(dataset, query);
        let mut approx_scores = codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                quantizer
                    .estimate_inner_product(code, query)
                    .map(|score| (index, score))
            })
            .collect::<Result<Vec<_>>>()?;
        update_hits_from_scores(&mut approx_scores, exact_best_index, ks, max_k, &mut hits);
    }
    let query_seconds = query_start.elapsed().as_secs_f64();

    Ok(build_recall_curve_metrics(
        ks,
        queries.len(),
        hits,
        indexing_seconds,
        query_seconds,
    ))
}

pub fn evaluate_pq_recall_curve_dataset(
    dataset: &[Vec<f64>],
    queries: &[Vec<f64>],
    subspaces: usize,
    bits: u8,
    iterations: usize,
    ks: &[usize],
    seed: u64,
) -> Result<RecallCurveMetrics> {
    validate_dataset_and_queries(dataset, queries, ks)?;
    let indexing_start = Instant::now();
    let quantizer = ProductQuantizer::train(dataset, subspaces, bits, iterations, seed)?;
    let codes = quantizer.quantize_batch(dataset)?;
    let indexing_seconds = indexing_start.elapsed().as_secs_f64();

    let max_k = *ks.iter().max().expect("validated non-empty ks");
    let query_start = Instant::now();
    let mut hits = vec![0usize; ks.len()];
    for query in queries {
        let exact_best_index = exact_top1_index(dataset, query);
        let mut approx_scores = codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                quantizer
                    .approximate_inner_product(code, query)
                    .map(|score| (index, score))
            })
            .collect::<Result<Vec<_>>>()?;
        update_hits_from_scores(&mut approx_scores, exact_best_index, ks, max_k, &mut hits);
    }
    let query_seconds = query_start.elapsed().as_secs_f64();

    Ok(build_recall_curve_metrics(
        ks,
        queries.len(),
        hits,
        indexing_seconds,
        query_seconds,
    ))
}

pub fn evaluate_rabitq_recall_curve_dataset(
    dataset: &[Vec<f64>],
    queries: &[Vec<f64>],
    bits: u8,
    ks: &[usize],
    seed: u64,
) -> Result<RecallCurveMetrics> {
    validate_dataset_and_queries(dataset, queries, ks)?;
    let indexing_start = Instant::now();
    let quantizer = RaBitQQuantizer::train(dataset, bits, seed)?;
    let codes = quantizer.quantize_batch(dataset)?;
    let indexing_seconds = indexing_start.elapsed().as_secs_f64();

    let max_k = *ks.iter().max().expect("validated non-empty ks");
    let query_start = Instant::now();
    let mut hits = vec![0usize; ks.len()];
    for query in queries {
        let exact_best_index = exact_top1_index(dataset, query);
        let mut approx_scores = codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                quantizer
                    .approximate_inner_product(code, query)
                    .map(|score| (index, score))
            })
            .collect::<Result<Vec<_>>>()?;
        update_hits_from_scores(&mut approx_scores, exact_best_index, ks, max_k, &mut hits);
    }
    let query_seconds = query_start.elapsed().as_secs_f64();

    Ok(build_recall_curve_metrics(
        ks,
        queries.len(),
        hits,
        indexing_seconds,
        query_seconds,
    ))
}

pub fn evaluate_mse_bounds(
    dimension: usize,
    bits: u8,
    samples: usize,
    seed: u64,
) -> Result<BoundMetrics> {
    let measured = evaluate_mse(dimension, bits, samples, seed)?.average_mse;
    Ok(BoundMetrics {
        measured,
        lower_bound: mse_lower_bound(bits),
        upper_bound: mse_upper_bound(bits),
        small_bit_reference: mse_small_bit_reference(bits),
    })
}

pub fn evaluate_prod_bounds(
    dimension: usize,
    bits: u8,
    samples: usize,
    seed: u64,
) -> Result<BoundMetrics> {
    let measured = evaluate_prod(dimension, bits, samples, seed)?.mse;
    Ok(BoundMetrics {
        measured,
        lower_bound: prod_lower_bound(dimension, bits),
        upper_bound: prod_upper_bound(dimension, bits),
        small_bit_reference: prod_small_bit_reference(dimension, bits),
    })
}

pub fn mse_lower_bound(bits: u8) -> f64 {
    4.0f64.powi(-(bits as i32))
}

pub fn mse_upper_bound(bits: u8) -> f64 {
    3.0_f64.sqrt() * PI / 2.0 * mse_lower_bound(bits)
}

pub fn prod_lower_bound(dimension: usize, bits: u8) -> f64 {
    mse_lower_bound(bits) / dimension as f64
}

pub fn prod_upper_bound(dimension: usize, bits: u8) -> f64 {
    mse_upper_bound(bits) / dimension as f64
}

pub fn mse_small_bit_reference(bits: u8) -> Option<f64> {
    match bits {
        1 => Some(0.36),
        2 => Some(0.117),
        3 => Some(0.03),
        4 => Some(0.009),
        _ => None,
    }
}

pub fn prod_small_bit_reference(dimension: usize, bits: u8) -> Option<f64> {
    let coefficient = match bits {
        1 => Some(1.57),
        2 => Some(0.56),
        3 => Some(0.18),
        4 => Some(0.047),
        _ => None,
    }?;
    Some(coefficient / dimension as f64)
}

fn random_unit_vector(dimension: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut vector = Vec::with_capacity(dimension);
    let mut norm_sq = 0.0;

    for _ in 0..dimension {
        let value: f64 = StandardNormal.sample(rng);
        vector.push(value);
        norm_sq += value * value;
    }

    let norm = norm_sq.sqrt();
    for value in &mut vector {
        *value /= norm;
    }

    vector
}

fn validate_non_zero(parameter: &'static str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(TurboQuantError::InvalidExperimentParameter {
            parameter,
            detail: "value must be greater than zero",
        });
    }
    Ok(())
}

fn validate_ks(ks: &[usize], dataset_size: usize) -> Result<()> {
    validate_non_zero("ks", ks.len())?;
    for &k in ks {
        validate_non_zero("top_k", k)?;
        if k > dataset_size {
            return Err(TurboQuantError::InvalidExperimentParameter {
                parameter: "top_k",
                detail: "top_k must be less than or equal to dataset_size",
            });
        }
    }
    Ok(())
}

fn validate_dataset_and_queries(
    dataset: &[Vec<f64>],
    queries: &[Vec<f64>],
    ks: &[usize],
) -> Result<usize> {
    if dataset.is_empty() {
        return Err(TurboQuantError::EmptyDataset);
    }
    validate_non_zero("query_count", queries.len())?;
    validate_ks(ks, dataset.len())?;

    let dimension = dataset[0].len();
    for vector in dataset {
        validate_dimension(dimension, vector.len())?;
    }
    for query in queries {
        validate_dimension(dimension, query.len())?;
    }
    Ok(dimension)
}

fn exact_top1_index(dataset: &[Vec<f64>], query: &[f64]) -> usize {
    let mut best_index = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for (index, vector) in dataset.iter().enumerate() {
        let score = dot(query, vector);
        if score > best_score {
            best_score = score;
            best_index = index;
        }
    }

    best_index
}

fn update_hits_from_scores(
    approx_scores: &mut Vec<(usize, f64)>,
    exact_best_index: usize,
    ks: &[usize],
    max_k: usize,
    hits: &mut [usize],
) {
    approx_scores.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap());
    let approx_top = approx_scores
        .iter()
        .take(max_k)
        .map(|&(index, _)| index)
        .collect::<Vec<_>>();

    for (slot, &k) in ks.iter().enumerate() {
        if approx_top[..k].contains(&exact_best_index) {
            hits[slot] += 1;
        }
    }
}

fn build_recall_curve_metrics(
    ks: &[usize],
    query_count: usize,
    hits: Vec<usize>,
    indexing_seconds: f64,
    query_seconds: f64,
) -> RecallCurveMetrics {
    RecallCurveMetrics {
        points: ks
            .iter()
            .enumerate()
            .map(|(slot, &k)| RecallPoint {
                k,
                recall_at_1: hits[slot] as f64 / query_count as f64,
            })
            .collect(),
        indexing_seconds,
        query_seconds,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_mse, evaluate_mse_bounds, evaluate_pq_recall_curve_dataset, evaluate_prod,
        evaluate_prod_bounds, evaluate_rabitq_recall_curve_dataset, evaluate_recall,
        evaluate_recall_curve, mse_lower_bound, mse_upper_bound, prod_lower_bound,
        prod_upper_bound,
    };
    use crate::TurboQuantError;

    #[test]
    fn experiment_metrics_are_finite() {
        let mse = evaluate_mse(8, 2, 8, 7).unwrap();
        let prod = evaluate_prod(8, 2, 8, 7).unwrap();
        let recall = evaluate_recall(8, 2, 32, 8, 4, 7).unwrap();

        assert!(mse.average_mse.is_finite());
        assert!(prod.bias.is_finite());
        assert!(prod.variance.is_finite());
        assert!(prod.mse.is_finite());
        assert!(recall.recall_at_k.is_finite());
    }

    #[test]
    fn experiment_rejects_invalid_parameters() {
        let zero_samples = evaluate_mse(8, 2, 0, 7).unwrap_err();
        assert!(matches!(
            zero_samples,
            TurboQuantError::InvalidExperimentParameter {
                parameter: "samples",
                ..
            }
        ));

        let invalid_topk = evaluate_recall(8, 2, 4, 2, 5, 7).unwrap_err();
        assert!(matches!(
            invalid_topk,
            TurboQuantError::InvalidExperimentParameter {
                parameter: "top_k",
                ..
            }
        ));

        let zero_queries = evaluate_recall(8, 2, 4, 0, 2, 7).unwrap_err();
        assert!(matches!(
            zero_queries,
            TurboQuantError::InvalidExperimentParameter {
                parameter: "query_count",
                ..
            }
        ));

        let prod = evaluate_prod(8, 2, 4, 7).unwrap();
        assert!(prod.mse.is_finite());
    }

    #[test]
    fn theoretical_bounds_match_closed_forms() {
        let mse_bounds = evaluate_mse_bounds(64, 2, 8, 7).unwrap();
        assert!((mse_bounds.lower_bound - 0.0625).abs() < 1e-12);
        assert!((mse_bounds.lower_bound - mse_lower_bound(2)).abs() < 1e-12);
        assert!((mse_bounds.upper_bound - mse_upper_bound(2)).abs() < 1e-12);

        let prod_bounds = evaluate_prod_bounds(64, 2, 8, 7).unwrap();
        assert!((prod_bounds.lower_bound - (0.0625 / 64.0)).abs() < 1e-12);
        assert!((prod_bounds.lower_bound - prod_lower_bound(64, 2)).abs() < 1e-12);
        assert!((prod_bounds.upper_bound - prod_upper_bound(64, 2)).abs() < 1e-12);
    }

    #[test]
    fn recall_curve_is_monotone_in_k() {
        let curve = evaluate_recall_curve(8, 2, 32, 8, &[1, 2, 4, 8], 7).unwrap();
        for pair in curve.points.windows(2) {
            assert!(pair[1].recall_at_1 >= pair[0].recall_at_1);
        }
        assert!(curve.indexing_seconds.is_finite());
        assert!(curve.query_seconds.is_finite());
    }

    #[test]
    fn pq_recall_curve_is_monotone_in_k() {
        let dataset = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.9, 0.1, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
            vec![0.1, 0.0, 0.0, 0.9],
        ];
        let queries = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let curve =
            evaluate_pq_recall_curve_dataset(&dataset, &queries, 2, 2, 4, &[1, 2, 4], 7).unwrap();
        for pair in curve.points.windows(2) {
            assert!(pair[1].recall_at_1 >= pair[0].recall_at_1);
        }
    }

    #[test]
    fn rabitq_recall_curve_is_monotone_in_k() {
        let dataset = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 0.9, 0.1, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
            vec![0.1, 0.0, 0.0, 0.9],
        ];
        let queries = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let curve =
            evaluate_rabitq_recall_curve_dataset(&dataset, &queries, 2, &[1, 2, 4], 7).unwrap();
        for pair in curve.points.windows(2) {
            assert!(pair[1].recall_at_1 >= pair[0].recall_at_1);
        }
    }
}
