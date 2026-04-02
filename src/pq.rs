use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::error::{Result, TurboQuantError};
use crate::math::validate_dimension;
use crate::packed::PackedBits;

#[derive(Debug, Clone)]
pub struct PqCode {
    indices: PackedBits,
}

#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    dimension: usize,
    subspaces: usize,
    bit_width: u8,
    ranges: Vec<(usize, usize)>,
    codebooks: Vec<Vec<Vec<f64>>>,
}

impl ProductQuantizer {
    pub fn train(
        dataset: &[Vec<f64>],
        subspaces: usize,
        bit_width: u8,
        iterations: usize,
        seed: u64,
    ) -> Result<Self> {
        if dataset.is_empty() {
            return Err(TurboQuantError::EmptyDataset);
        }
        if bit_width > 20 {
            return Err(TurboQuantError::UnsupportedBitWidth(bit_width));
        }
        if iterations == 0 {
            return Err(TurboQuantError::InvalidExperimentParameter {
                parameter: "iterations",
                detail: "value must be greater than zero",
            });
        }

        let dimension = dataset[0].len();
        if dimension < 2 {
            return Err(TurboQuantError::InvalidDimension(dimension));
        }
        if subspaces == 0 || subspaces > dimension {
            return Err(TurboQuantError::InvalidSubspaceCount {
                dimension,
                subspaces,
            });
        }

        for (row_index, row) in dataset.iter().enumerate() {
            if row.len() != dimension {
                return Err(TurboQuantError::InconsistentRowDimension {
                    line: row_index + 1,
                    expected: dimension,
                    actual: row.len(),
                });
            }
        }

        let ranges = partition_ranges(dimension, subspaces)?;
        let codebook_size = 1usize << bit_width;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut codebooks = Vec::with_capacity(subspaces);
        for &(start, end) in &ranges {
            codebooks.push(train_subspace_codebook(
                dataset,
                start,
                end,
                codebook_size,
                iterations,
                &mut rng,
            ));
        }

        Ok(Self {
            dimension,
            subspaces,
            bit_width,
            ranges,
            codebooks,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn subspaces(&self) -> usize {
        self.subspaces
    }

    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    pub fn quantize(&self, vector: &[f64]) -> Result<PqCode> {
        validate_dimension(self.dimension, vector.len())?;
        let indices = self
            .ranges
            .iter()
            .zip(&self.codebooks)
            .map(|(&(start, end), codebook)| nearest_centroid_index(&vector[start..end], codebook))
            .map(|index| index as u64)
            .collect::<Vec<_>>();

        Ok(PqCode {
            indices: PackedBits::pack_values(&indices, self.bit_width)
                .expect("PQ bit width is validated"),
        })
    }

    pub fn quantize_batch<T>(&self, vectors: &[T]) -> Result<Vec<PqCode>>
    where
        T: AsRef<[f64]> + Sync,
    {
        let results = vectors
            .par_iter()
            .map(|vector| self.quantize(vector.as_ref()))
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    pub fn dequantize(&self, code: &PqCode) -> Result<Vec<f64>> {
        if code.indices.symbol_count() != self.subspaces
            || code.indices.bits_per_symbol() != self.bit_width
        {
            return Err(TurboQuantError::PackedLayoutMismatch {
                expected_symbols: self.subspaces,
                actual_symbols: code.indices.symbol_count(),
                expected_bits_per_symbol: self.bit_width,
                actual_bits_per_symbol: code.indices.bits_per_symbol(),
            });
        }

        let mut reconstructed = vec![0.0; self.dimension];
        for (subspace, &(start, end)) in self.ranges.iter().enumerate() {
            let centroid_index = code.indices.get(subspace)? as usize;
            let centroid = self.codebooks[subspace].get(centroid_index).ok_or(
                TurboQuantError::InvalidCodebookIndex {
                    index: centroid_index,
                    codebook_len: self.codebooks[subspace].len(),
                },
            )?;
            reconstructed[start..end].copy_from_slice(centroid);
        }

        Ok(reconstructed)
    }

    pub fn approximate_inner_product(&self, code: &PqCode, query: &[f64]) -> Result<f64> {
        validate_dimension(self.dimension, query.len())?;
        if code.indices.symbol_count() != self.subspaces
            || code.indices.bits_per_symbol() != self.bit_width
        {
            return Err(TurboQuantError::PackedLayoutMismatch {
                expected_symbols: self.subspaces,
                actual_symbols: code.indices.symbol_count(),
                expected_bits_per_symbol: self.bit_width,
                actual_bits_per_symbol: code.indices.bits_per_symbol(),
            });
        }

        let mut total = 0.0;
        for (subspace, &(start, end)) in self.ranges.iter().enumerate() {
            let centroid_index = code.indices.get(subspace)? as usize;
            let centroid = self.codebooks[subspace].get(centroid_index).ok_or(
                TurboQuantError::InvalidCodebookIndex {
                    index: centroid_index,
                    codebook_len: self.codebooks[subspace].len(),
                },
            )?;
            total += centroid
                .iter()
                .zip(&query[start..end])
                .map(|(left, right)| left * right)
                .sum::<f64>();
        }
        Ok(total)
    }
}

impl PqCode {
    pub fn indices(&self) -> &PackedBits {
        &self.indices
    }

    pub fn unpack_indices(&self) -> Result<Vec<u64>> {
        self.indices.unpack_values()
    }

    pub fn storage_bytes(&self) -> usize {
        self.indices.storage_bytes()
    }
}

fn partition_ranges(dimension: usize, subspaces: usize) -> Result<Vec<(usize, usize)>> {
    if subspaces == 0 || subspaces > dimension {
        return Err(TurboQuantError::InvalidSubspaceCount {
            dimension,
            subspaces,
        });
    }

    let mut ranges = Vec::with_capacity(subspaces);
    for subspace in 0..subspaces {
        let start = subspace * dimension / subspaces;
        let end = (subspace + 1) * dimension / subspaces;
        if start == end {
            return Err(TurboQuantError::InvalidSubspaceCount {
                dimension,
                subspaces,
            });
        }
        ranges.push((start, end));
    }
    Ok(ranges)
}

fn train_subspace_codebook(
    dataset: &[Vec<f64>],
    start: usize,
    end: usize,
    codebook_size: usize,
    iterations: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let samples = dataset
        .iter()
        .map(|row| row[start..end].to_vec())
        .collect::<Vec<_>>();
    let mut centroids = initialize_centroids(&samples, codebook_size, rng);

    for _ in 0..iterations {
        let mut sums = vec![vec![0.0; end - start]; codebook_size];
        let mut counts = vec![0usize; codebook_size];

        for sample in &samples {
            let centroid = nearest_centroid_index(sample, &centroids);
            counts[centroid] += 1;
            for (slot, &value) in sample.iter().enumerate() {
                sums[centroid][slot] += value;
            }
        }

        for centroid in 0..codebook_size {
            if counts[centroid] == 0 {
                centroids[centroid] = samples[rng.random_range(0..samples.len())].clone();
            } else {
                let scale = 1.0 / counts[centroid] as f64;
                for value in &mut sums[centroid] {
                    *value *= scale;
                }
                centroids[centroid] = sums[centroid].clone();
            }
        }
    }

    centroids
}

fn initialize_centroids(
    samples: &[Vec<f64>],
    codebook_size: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f64>> {
    let mut centroids = Vec::with_capacity(codebook_size);
    for _ in 0..codebook_size {
        centroids.push(samples[rng.random_range(0..samples.len())].clone());
    }
    centroids
}

fn nearest_centroid_index(sample: &[f64], codebook: &[Vec<f64>]) -> usize {
    let mut best_index = 0usize;
    let mut best_distance = f64::INFINITY;

    for (index, centroid) in codebook.iter().enumerate() {
        let distance = sample
            .iter()
            .zip(centroid)
            .map(|(left, right)| {
                let delta = left - right;
                delta * delta
            })
            .sum::<f64>();
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }

    best_index
}
