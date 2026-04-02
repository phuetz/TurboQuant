use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationBackend {
    DenseGaussian,
    WalshHadamard,
}

#[derive(Debug, Clone)]
pub(crate) enum Rotation {
    DenseGaussian {
        matrix: DMatrix<f64>,
    },
    WalshHadamard {
        dimension: usize,
        padded_dimension: usize,
        signs_pre: Vec<f64>,
        signs_post: Vec<f64>,
        scale: f64,
    },
}

impl Rotation {
    pub(crate) fn new(dimension: usize, rng: &mut StdRng) -> Self {
        Self::new_with_backend(dimension, rng, RotationBackend::DenseGaussian)
    }

    pub(crate) fn new_with_backend(
        dimension: usize,
        rng: &mut StdRng,
        backend: RotationBackend,
    ) -> Self {
        match backend {
            RotationBackend::DenseGaussian => Self::new_dense_gaussian(dimension, rng),
            RotationBackend::WalshHadamard => Self::new_walsh_hadamard(dimension, rng),
        }
    }

    fn new_dense_gaussian(dimension: usize, rng: &mut StdRng) -> Self {
        let gaussian = DMatrix::from_fn(dimension, dimension, |_, _| rng.sample(StandardNormal));
        let qr = gaussian.qr();
        let (mut q, r) = qr.unpack();

        // Flip columns so the orthogonal factor is sampled consistently from
        // the Gaussian QR construction described in the paper.
        for column in 0..dimension {
            let sign = if r[(column, column)] < 0.0 { -1.0 } else { 1.0 };
            if sign < 0.0 {
                for row in 0..dimension {
                    q[(row, column)] *= -1.0;
                }
            }
        }

        Self::DenseGaussian { matrix: q }
    }

    fn new_walsh_hadamard(dimension: usize, rng: &mut StdRng) -> Self {
        let padded_dimension = dimension.next_power_of_two();
        let signs_pre = sample_signs(padded_dimension, rng);
        let signs_post = sample_signs(padded_dimension, rng);
        let scale = 1.0 / (padded_dimension as f64).sqrt();

        Self::WalshHadamard {
            dimension,
            padded_dimension,
            signs_pre,
            signs_post,
            scale,
        }
    }

    pub(crate) fn backend(&self) -> RotationBackend {
        match self {
            Rotation::DenseGaussian { .. } => RotationBackend::DenseGaussian,
            Rotation::WalshHadamard { .. } => RotationBackend::WalshHadamard,
        }
    }

    pub(crate) fn apply(&self, vector: &[f64]) -> Vec<f64> {
        match self {
            Rotation::DenseGaussian { matrix } => {
                let vector = DVector::from_column_slice(vector);
                (matrix * vector).iter().copied().collect()
            }
            Rotation::WalshHadamard {
                dimension,
                padded_dimension,
                signs_pre,
                signs_post,
                scale,
            } => apply_walsh_hadamard(
                vector,
                *dimension,
                *padded_dimension,
                signs_pre,
                signs_post,
                *scale,
            ),
        }
    }

    pub(crate) fn apply_transpose(&self, vector: &[f64]) -> Vec<f64> {
        match self {
            Rotation::DenseGaussian { matrix } => {
                let vector = DVector::from_column_slice(vector);
                (matrix.transpose() * vector).iter().copied().collect()
            }
            Rotation::WalshHadamard {
                dimension,
                padded_dimension,
                signs_pre,
                signs_post,
                scale,
            } => apply_walsh_hadamard(
                vector,
                *dimension,
                *padded_dimension,
                signs_post,
                signs_pre,
                *scale,
            ),
        }
    }

    pub(crate) fn apply_batch_flat(
        &self,
        values: &[f64],
        row_count: usize,
        dimension: usize,
    ) -> Vec<f64> {
        match self {
            Rotation::DenseGaussian { matrix } => {
                // Each row of `values` is a vector to rotate.  Treat them as
                // columns of a dim × row_count matrix (the flat layout already
                // matches column-major storage for that shape).  After GEMM
                // the result columns are the rotated vectors, and as_slice()
                // returns column-major data = the desired flat row-major output.
                let input_cols =
                    DMatrix::from_column_slice(dimension, row_count, values);
                let result = matrix * input_cols;
                result.as_slice().to_vec()
            }
            Rotation::WalshHadamard {
                dimension: rotation_dimension,
                padded_dimension,
                signs_pre,
                signs_post,
                scale,
            } => apply_walsh_hadamard_batch_flat(
                values,
                row_count,
                *rotation_dimension,
                *padded_dimension,
                signs_pre,
                signs_post,
                *scale,
            ),
        }
    }

    pub(crate) fn apply_transpose_batch_flat(
        &self,
        values: &[f64],
        row_count: usize,
        dimension: usize,
    ) -> Vec<f64> {
        match self {
            Rotation::DenseGaussian { matrix } => {
                let input_cols =
                    DMatrix::from_column_slice(dimension, row_count, values);
                let result = matrix.transpose() * input_cols;
                result.as_slice().to_vec()
            }
            Rotation::WalshHadamard {
                dimension: rotation_dimension,
                padded_dimension,
                signs_pre,
                signs_post,
                scale,
            } => apply_walsh_hadamard_batch_flat(
                values,
                row_count,
                *rotation_dimension,
                *padded_dimension,
                signs_post,
                signs_pre,
                *scale,
            ),
        }
    }
}

pub(crate) fn gaussian_matrix(dimension: usize, rng: &mut StdRng) -> DMatrix<f64> {
    DMatrix::from_fn(dimension, dimension, |_, _| rng.sample(StandardNormal))
}

pub(crate) fn apply_matrix(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let vector = DVector::from_column_slice(vector);
    (matrix * vector).iter().copied().collect()
}

pub(crate) fn apply_matrix_transpose(matrix: &DMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let vector = DVector::from_column_slice(vector);
    (matrix.transpose() * vector).iter().copied().collect()
}

fn sample_signs(dimension: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..dimension)
        .map(|_| {
            if rng.random::<u64>() & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn apply_walsh_hadamard(
    vector: &[f64],
    dimension: usize,
    padded_dimension: usize,
    signs_left: &[f64],
    signs_right: &[f64],
    scale: f64,
) -> Vec<f64> {
    let mut padded = vec![0.0; padded_dimension];
    padded[..dimension].copy_from_slice(vector);

    for index in 0..padded_dimension {
        padded[index] *= signs_left[index];
    }

    fast_walsh_hadamard_in_place(&mut padded);

    for index in 0..padded_dimension {
        padded[index] *= scale * signs_right[index];
    }

    padded.truncate(dimension);
    padded
}

fn fast_walsh_hadamard_in_place(values: &mut [f64]) {
    let mut stride = 1;
    while stride < values.len() {
        let step = stride * 2;
        let mut block_start = 0;
        while block_start < values.len() {
            for offset in 0..stride {
                let left = block_start + offset;
                let right = left + stride;
                let a = values[left];
                let b = values[right];
                values[left] = a + b;
                values[right] = a - b;
            }
            block_start += step;
        }
        stride = step;
    }
}

fn apply_walsh_hadamard_batch_flat(
    values: &[f64],
    row_count: usize,
    dimension: usize,
    padded_dimension: usize,
    signs_left: &[f64],
    signs_right: &[f64],
    scale: f64,
) -> Vec<f64> {
    if dimension == padded_dimension {
        let mut output = values.to_vec();
        output
            .par_chunks_mut(dimension)
            .for_each(|row| transform_row_in_place(row, signs_left, signs_right, scale));
        return output;
    }

    let mut padded_output = vec![0.0; row_count * padded_dimension];
    padded_output
        .par_chunks_mut(padded_dimension)
        .zip(values.par_chunks(dimension))
        .for_each(|(target_row, source_row)| {
            target_row[..dimension].copy_from_slice(source_row);
            transform_row_in_place(target_row, signs_left, signs_right, scale);
        });

    let mut compact = vec![0.0; row_count * dimension];
    compact
        .par_chunks_mut(dimension)
        .zip(padded_output.par_chunks(padded_dimension))
        .for_each(|(target_row, source_row)| target_row.copy_from_slice(&source_row[..dimension]));
    compact
}

fn transform_row_in_place(values: &mut [f64], signs_left: &[f64], signs_right: &[f64], scale: f64) {
    for index in 0..values.len() {
        values[index] *= signs_left[index];
    }
    fast_walsh_hadamard_in_place(values);
    for index in 0..values.len() {
        values[index] *= scale * signs_right[index];
    }
}
