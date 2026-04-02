use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{OnceLock, RwLock};

use statrs::function::gamma::ln_gamma;

use crate::error::{Result, TurboQuantError};

#[derive(Debug, Clone, Copy)]
pub struct LloydMaxOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub integration_subdivisions: usize,
}

impl Default for LloydMaxOptions {
    fn default() -> Self {
        Self {
            max_iterations: 256,
            tolerance: 1e-10,
            integration_subdivisions: 128,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScalarCodebook {
    centroids: Vec<f64>,
    boundaries: Vec<f64>,
}

impl ScalarCodebook {
    pub(crate) fn solve(dimension: usize, bit_width: u8, options: LloydMaxOptions) -> Result<Self> {
        let key = CodebookCacheKey::new(dimension, bit_width, options);
        if let Some(codebook) = codebook_cache().read().unwrap().get(&key).cloned() {
            return Ok(codebook);
        }

        let codebook = Self::solve_uncached(dimension, bit_width, options)?;
        codebook_cache()
            .write()
            .unwrap()
            .insert(key, codebook.clone());
        Ok(codebook)
    }

    fn solve_uncached(dimension: usize, bit_width: u8, options: LloydMaxOptions) -> Result<Self> {
        if dimension < 2 {
            return Err(TurboQuantError::InvalidDimension(dimension));
        }
        if bit_width > 20 {
            return Err(TurboQuantError::UnsupportedBitWidth(bit_width));
        }
        if bit_width == 0 {
            return Ok(Self::from_centroids(vec![0.0]));
        }

        let distribution = SphereCoordinateDistribution::new(dimension);
        let centroid_count = 1usize << bit_width;
        let step = 2.0 / centroid_count as f64;
        let mut centroids = (0..centroid_count)
            .map(|index| -1.0 + (index as f64 + 0.5) * step)
            .collect::<Vec<_>>();

        for _ in 0..options.max_iterations {
            let boundaries = compute_boundaries(&centroids);
            let mut next = centroids.clone();
            let mut max_delta = 0.0_f64;

            for index in 0..centroid_count {
                let left = boundaries[index];
                let right = boundaries[index + 1];
                let mass = integrate(
                    |x| distribution.pdf(x),
                    left,
                    right,
                    options.integration_subdivisions,
                );
                let mean = if mass > 0.0 {
                    integrate(
                        |x| x * distribution.pdf(x),
                        left,
                        right,
                        options.integration_subdivisions,
                    ) / mass
                } else {
                    0.5 * (left + right)
                };

                next[index] = mean.clamp(-1.0, 1.0);
                max_delta = max_delta.max((next[index] - centroids[index]).abs());
            }

            centroids = next;
            if max_delta <= options.tolerance {
                break;
            }
        }

        Ok(Self::from_centroids(centroids))
    }

    pub fn centroids(&self) -> &[f64] {
        &self.centroids
    }

    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    pub(crate) fn find_index(&self, value: f64) -> usize {
        if self.centroids.len() == 1 {
            return 0;
        }

        let clamped = value.clamp(-1.0, 1.0);
        // Binary search on boundaries: find the first boundary > clamped.
        // boundaries[0] = -1.0, boundaries[last] = 1.0, so the answer is
        // partition_point - 1, clamped to valid range.
        let pos = self.boundaries[1..].partition_point(|&b| b < clamped);
        pos.min(self.centroids.len() - 1)
    }

    pub(crate) fn centroid(&self, index: usize) -> Result<f64> {
        self.centroids
            .get(index)
            .copied()
            .ok_or(TurboQuantError::InvalidCodebookIndex {
                index,
                codebook_len: self.centroids.len(),
            })
    }

    fn from_centroids(mut centroids: Vec<f64>) -> Self {
        centroids.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
        let boundaries = compute_boundaries(&centroids);
        Self {
            centroids,
            boundaries,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct CodebookCacheKey {
    dimension: usize,
    bit_width: u8,
    max_iterations: usize,
    tolerance_bits: u64,
    integration_subdivisions: usize,
}

impl CodebookCacheKey {
    fn new(dimension: usize, bit_width: u8, options: LloydMaxOptions) -> Self {
        Self {
            dimension,
            bit_width,
            max_iterations: options.max_iterations,
            tolerance_bits: options.tolerance.to_bits(),
            integration_subdivisions: options.integration_subdivisions,
        }
    }
}

fn codebook_cache() -> &'static RwLock<HashMap<CodebookCacheKey, ScalarCodebook>> {
    static CACHE: OnceLock<RwLock<HashMap<CodebookCacheKey, ScalarCodebook>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

fn compute_boundaries(centroids: &[f64]) -> Vec<f64> {
    let mut boundaries = Vec::with_capacity(centroids.len() + 1);
    boundaries.push(-1.0);

    for window in centroids.windows(2) {
        boundaries.push(0.5 * (window[0] + window[1]));
    }

    boundaries.push(1.0);
    boundaries
}

#[derive(Debug, Clone, Copy)]
struct SphereCoordinateDistribution {
    log_normalizer: f64,
    exponent: f64,
}

impl SphereCoordinateDistribution {
    fn new(dimension: usize) -> Self {
        let log_normalizer = ln_gamma(dimension as f64 / 2.0)
            - 0.5 * PI.ln()
            - ln_gamma((dimension as f64 - 1.0) / 2.0);
        let exponent = (dimension as f64 - 3.0) / 2.0;
        Self {
            log_normalizer,
            exponent,
        }
    }

    fn pdf(&self, x: f64) -> f64 {
        let clamped = x.clamp(-1.0 + 1e-12, 1.0 - 1e-12);
        let one_minus_square = (1.0 - clamped * clamped).max(1e-16);
        (self.log_normalizer + self.exponent * one_minus_square.ln()).exp()
    }
}

fn integrate<F>(function: F, start: f64, end: f64, subdivisions: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if (end - start).abs() <= f64::EPSILON {
        return 0.0;
    }

    let steps = subdivisions.max(2) + (subdivisions.max(2) % 2);
    let width = (end - start) / steps as f64;
    let mut sum = function(start) + function(end);

    for step in 1..steps {
        let x = start + step as f64 * width;
        let weight = if step % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * function(x);
    }

    sum * width / 3.0
}
