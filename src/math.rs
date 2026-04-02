use crate::error::{Result, TurboQuantError};
use wide::f64x4;

pub(crate) fn validate_dimension(expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(TurboQuantError::DimensionMismatch { expected, actual })
    }
}

pub(crate) fn l2_norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

pub(crate) fn normalize(vector: &[f64]) -> (f64, Vec<f64>) {
    let norm = l2_norm(vector);
    if norm == 0.0 {
        (0.0, vec![0.0; vector.len()])
    } else {
        (norm, vector.iter().map(|value| value / norm).collect())
    }
}

pub(crate) fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    let simd_len = lhs.len().min(rhs.len()) / 4 * 4;
    let mut acc = f64x4::from([0.0; 4]);

    for index in (0..simd_len).step_by(4) {
        let lhs_vec = load_f64x4(&lhs[index..index + 4]);
        let rhs_vec = load_f64x4(&rhs[index..index + 4]);
        acc += lhs_vec * rhs_vec;
    }

    let lanes: [f64; 4] = acc.into();
    let mut sum = lanes.into_iter().sum::<f64>();

    for index in simd_len..lhs.len().min(rhs.len()) {
        sum += lhs[index] * rhs[index];
    }

    sum
}

pub(crate) fn squared_l2_distance(lhs: &[f64], rhs: &[f64]) -> f64 {
    let simd_len = lhs.len().min(rhs.len()) / 4 * 4;
    let mut acc = f64x4::from([0.0; 4]);

    for index in (0..simd_len).step_by(4) {
        let lhs_vec = load_f64x4(&lhs[index..index + 4]);
        let rhs_vec = load_f64x4(&rhs[index..index + 4]);
        let delta = lhs_vec - rhs_vec;
        acc += delta * delta;
    }

    let lanes: [f64; 4] = acc.into();
    let mut sum = lanes.into_iter().sum::<f64>();

    for index in simd_len..lhs.len().min(rhs.len()) {
        let delta = lhs[index] - rhs[index];
        sum += delta * delta;
    }

    sum
}

pub(crate) fn subtract(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; lhs.len().min(rhs.len())];
    let simd_len = result.len() / 4 * 4;

    for index in (0..simd_len).step_by(4) {
        let lhs_vec = load_f64x4(&lhs[index..index + 4]);
        let rhs_vec = load_f64x4(&rhs[index..index + 4]);
        store_f64x4(&mut result[index..index + 4], lhs_vec - rhs_vec);
    }

    for index in simd_len..result.len() {
        result[index] = lhs[index] - rhs[index];
    }

    result
}

pub(crate) fn load_f64x4(slice: &[f64]) -> f64x4 {
    f64x4::from([slice[0], slice[1], slice[2], slice[3]])
}

pub(crate) fn store_f64x4(slice: &mut [f64], value: f64x4) {
    let lanes: [f64; 4] = value.into();
    slice[..4].copy_from_slice(&lanes);
}
