from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable

import torch


DEFAULT_LLOYD_MAX_ITERATIONS = 256
DEFAULT_LLOYD_MAX_TOLERANCE = 1e-10
DEFAULT_LLOYD_MAX_SUBDIVISIONS = 128
DEFAULT_QJL_SCALE = math.sqrt(math.pi / 2.0)


@dataclass(frozen=True)
class QuantizerSpec:
    kind: str
    bit_width: int | None = None
    outlier_count: int | None = None
    outlier_bit_width: int | None = None
    regular_bit_width: int | None = None

    def is_passthrough(self) -> bool:
        return self.kind == "none"

    def is_mixed(self) -> bool:
        return self.kind.startswith("mixed_")

    def validate(self) -> None:
        supported = {"none", "mse", "prod", "mixed_mse", "mixed_prod"}
        if self.kind not in supported:
            raise ValueError(f"unsupported quantizer kind: {self.kind}")
        if self.kind == "none":
            return
        if self.is_mixed():
            if self.outlier_count is None or self.outlier_bit_width is None or self.regular_bit_width is None:
                raise ValueError("mixed quantizer requires outlier_count, outlier_bit_width, regular_bit_width")
            if self.outlier_count < 0:
                raise ValueError("outlier_count must be non-negative")
            if self.outlier_bit_width < 0 or self.regular_bit_width < 0:
                raise ValueError("mixed bit widths must be non-negative")
            return
        if self.bit_width is None or self.bit_width < 0:
            raise ValueError("bit_width must be a non-negative integer")

    def describe(self) -> str:
        if self.kind == "none":
            return "none"
        if self.is_mixed():
            return (
                f"{self.kind}:{self.outlier_count}:{self.outlier_bit_width}:{self.regular_bit_width}"
            )
        return f"{self.kind}:{self.bit_width}"


def parse_quantizer_spec(spec: str) -> QuantizerSpec:
    text = spec.strip().lower()
    if text in {"none", "full", "fp"}:
        return QuantizerSpec(kind="none")

    parts = text.split(":")
    if len(parts) == 2:
        quantizer = QuantizerSpec(kind=parts[0], bit_width=int(parts[1]))
    elif len(parts) == 4:
        quantizer = QuantizerSpec(
            kind=parts[0],
            outlier_count=int(parts[1]),
            outlier_bit_width=int(parts[2]),
            regular_bit_width=int(parts[3]),
        )
    else:
        raise ValueError(
            "quantizer spec must be one of: none, mse:<bits>, prod:<bits>, mixed_mse:<outliers>:<outlier_bits>:<regular_bits>, mixed_prod:<outliers>:<outlier_bits>:<regular_bits>"
        )

    quantizer.validate()
    return quantizer


@dataclass(frozen=True)
class SplitPlan:
    dimension: int
    outlier_indices: tuple[int, ...]
    regular_indices: tuple[int, ...]
    outlier_bit_width: int
    regular_bit_width: int

    @classmethod
    def new(
        cls,
        dimension: int,
        outlier_indices: Iterable[int],
        outlier_bit_width: int,
        regular_bit_width: int,
    ) -> "SplitPlan":
        if dimension < 2:
            raise ValueError(f"dimension must be at least 2, got {dimension}")

        raw_indices = tuple(int(index) for index in outlier_indices)
        unique = sorted(set(raw_indices))
        if len(unique) != len(raw_indices):
            raise ValueError("outlier_indices contains duplicates")
        if any(index < 0 or index >= dimension for index in unique):
            raise ValueError("outlier index is out of range")

        unique_set = set(unique)
        regular = tuple(index for index in range(dimension) if index not in unique_set)
        if len(unique) == 1 or len(regular) == 1:
            raise ValueError("each non-empty partition must have at least 2 channels")

        return cls(
            dimension=dimension,
            outlier_indices=tuple(unique),
            regular_indices=regular,
            outlier_bit_width=outlier_bit_width,
            regular_bit_width=regular_bit_width,
        )

    @classmethod
    def from_channel_rms(
        cls,
        samples: torch.Tensor,
        outlier_count: int,
        outlier_bit_width: int,
        regular_bit_width: int,
    ) -> "SplitPlan":
        if samples.ndim != 2:
            raise ValueError(f"expected 2D calibration tensor, got shape {tuple(samples.shape)}")
        dimension = samples.shape[1]
        if outlier_count > dimension:
            raise ValueError("outlier_count cannot exceed dimension")
        scores = samples.to(torch.float32).pow(2).sum(dim=0)
        ranked = torch.argsort(scores, descending=True)
        outliers = ranked[:outlier_count].tolist()
        return cls.new(dimension, outliers, outlier_bit_width, regular_bit_width)

    def effective_bit_width(self) -> float:
        total = len(self.outlier_indices) * self.outlier_bit_width + len(self.regular_indices) * self.regular_bit_width
        return total / self.dimension


@dataclass
class MseTensorCode:
    indices: torch.Tensor
    norms: torch.Tensor
    original_shape: tuple[int, ...]

    def storage_bytes(self) -> int:
        return self.indices.numel() * self.indices.element_size() + self.norms.numel() * self.norms.element_size()


@dataclass
class ProdTensorCode:
    mse_indices: torch.Tensor
    qjl_signs: torch.Tensor
    norms: torch.Tensor
    residual_norms: torch.Tensor
    original_shape: tuple[int, ...]

    def storage_bytes(self) -> int:
        return (
            self.mse_indices.numel() * self.mse_indices.element_size()
            + self.qjl_signs.numel() * self.qjl_signs.element_size()
            + self.norms.numel() * self.norms.element_size()
            + self.residual_norms.numel() * self.residual_norms.element_size()
        )


@dataclass
class MixedTensorCode:
    outlier_code: MseTensorCode | ProdTensorCode | None
    regular_code: MseTensorCode | ProdTensorCode | None
    plan: SplitPlan
    original_shape: tuple[int, ...]

    def storage_bytes(self) -> int:
        total = 0
        if self.outlier_code is not None:
            total += self.outlier_code.storage_bytes()
        if self.regular_code is not None:
            total += self.regular_code.storage_bytes()
        return total


class TorchTurboQuantMse:
    def __init__(
        self,
        dimension: int,
        bit_width: int,
        seed: int,
        dtype: torch.dtype = torch.float32,
        rotation: str = "dense_gaussian",
    ) -> None:
        if dimension < 2:
            raise ValueError(f"dimension must be at least 2, got {dimension}")
        if bit_width < 0:
            raise ValueError("bit_width must be non-negative")
        self.dimension = dimension
        self.bit_width = bit_width
        self.dtype = dtype
        self.rotation_backend = rotation
        if rotation == "walsh_hadamard":
            self._wht = _WalshHadamardRotation(dimension, seed, dtype=dtype)
            self.rotation = None  # Not a matrix — use _wht.apply() instead
        else:
            self._wht = None
            self.rotation = _gaussian_orthogonal_matrix(dimension, seed, dtype=dtype)
        centroids, boundaries = _solve_codebook(dimension, bit_width)
        self.centroids = torch.tensor(centroids, dtype=dtype)
        self.boundaries = torch.tensor(boundaries, dtype=dtype)

    def _migrate_to(self, device: torch.device) -> None:
        """Lazy device migration so rotation/centroids/boundaries match input device."""
        if self.rotation is not None and self.rotation.device != device:
            self.rotation = self.rotation.to(device)
        if self.centroids.device != device:
            self.centroids = self.centroids.to(device)
        if self.boundaries.device != device:
            self.boundaries = self.boundaries.to(device)
        if self._wht is not None:
            self._wht._migrate_to(device)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        self._migrate_to(x.device)
        if self._wht is not None:
            return self._wht.apply(x)
        return x @ self.rotation.T

    def _rotate_inverse(self, x: torch.Tensor) -> torch.Tensor:
        self._migrate_to(x.device)
        if self._wht is not None:
            return self._wht.apply_transpose(x)
        return x @ self.rotation

    def quantize_tensor(self, tensor: torch.Tensor) -> MseTensorCode:
        flat, shape = _flatten_last_dim(tensor)
        if flat.shape[1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}, got {flat.shape[1]}")

        norms = torch.linalg.norm(flat, dim=-1, keepdim=True)
        safe_norms = norms.clamp_min(1e-12)
        normalized = torch.where(norms > 0, flat / safe_norms, torch.zeros_like(flat))
        rotated = self._rotate(normalized)
        indices = _bucketize_centroids(rotated, self.boundaries).to(torch.int16)
        # Stay on the input device — caller decides if/when to move to CPU for storage.
        return MseTensorCode(indices=indices, norms=norms.squeeze(-1), original_shape=shape)

    def dequantize_tensor(self, code: MseTensorCode) -> torch.Tensor:
        self._migrate_to(code.indices.device)
        rotated = self.centroids[code.indices.long()]
        decoded = self._rotate_inverse(rotated)
        decoded = decoded * code.norms.to(decoded.dtype).unsqueeze(-1)
        return decoded.reshape(code.original_shape)

    def approximate_inner_product(self, code: MseTensorCode, query: torch.Tensor) -> torch.Tensor:
        query_flat, _ = _flatten_last_dim(query)
        decoded = self.dequantize_tensor(code).reshape(query_flat.shape)
        return (decoded * query_flat).sum(dim=-1)

    def encoded_bits_per_vector(self) -> int:
        return self.dimension * self.bit_width


class TorchTurboQuantProd:
    def __init__(
        self,
        dimension: int,
        bit_width: int,
        seed: int,
        dtype: torch.dtype = torch.float32,
        rotation: str = "dense_gaussian",
    ) -> None:
        if dimension < 2:
            raise ValueError(f"dimension must be at least 2, got {dimension}")
        if bit_width < 1:
            raise ValueError("bit_width must be at least 1 for product quantization")
        self.dimension = dimension
        self.bit_width = bit_width
        self.dtype = dtype
        self.mse = TorchTurboQuantMse(dimension, bit_width - 1, seed ^ 0x9E37_79B9_7F4A_7C15, dtype=dtype, rotation=rotation)
        self.sketch = _gaussian_matrix(dimension, seed ^ 0xD1B5_4A32_D192_ED03, dtype=dtype)

    def quantize_tensor(self, tensor: torch.Tensor) -> ProdTensorCode:
        flat, shape = _flatten_last_dim(tensor)
        if flat.shape[1] != self.dimension:
            raise ValueError(f"expected last dimension {self.dimension}, got {flat.shape[1]}")

        norms = torch.linalg.norm(flat, dim=-1, keepdim=True)
        safe_norms = norms.clamp_min(1e-12)
        normalized = torch.where(norms > 0, flat / safe_norms, torch.zeros_like(flat))

        mse_code = self.mse.quantize_tensor(normalized)
        mse_decoded = self.mse.dequantize_tensor(mse_code).reshape(flat.shape)
        residual = normalized - mse_decoded
        residual_norms = torch.linalg.norm(residual, dim=-1, keepdim=True)
        projected = residual @ self.sketch.T
        signs = projected >= 0

        zero_rows = residual_norms.squeeze(-1) == 0
        if zero_rows.any():
            signs[zero_rows] = True

        return ProdTensorCode(
            mse_indices=mse_code.indices,
            qjl_signs=signs,
            norms=norms.squeeze(-1),
            residual_norms=residual_norms.squeeze(-1),
            original_shape=shape,
        )

    def dequantize_tensor(self, code: ProdTensorCode) -> torch.Tensor:
        mse_decoded = self.mse.dequantize_tensor(
            MseTensorCode(indices=code.mse_indices, norms=torch.ones_like(code.norms), original_shape=code.original_shape)
        ).reshape(-1, self.dimension)
        sign_values = torch.where(code.qjl_signs, 1.0, -1.0).to(self.dtype)
        if self.sketch.device != sign_values.device:
            self.sketch = self.sketch.to(sign_values.device)
        projected = sign_values @ self.sketch
        scale = DEFAULT_QJL_SCALE * code.residual_norms.to(self.dtype).unsqueeze(-1) / self.dimension
        qjl = projected * scale
        decoded = (mse_decoded + qjl) * code.norms.to(self.dtype).unsqueeze(-1)
        return decoded.reshape(code.original_shape)

    def approximate_inner_product(self, code: ProdTensorCode, query: torch.Tensor) -> torch.Tensor:
        query_flat, _ = _flatten_last_dim(query)
        decoded = self.dequantize_tensor(code).reshape(query_flat.shape)
        return (decoded * query_flat).sum(dim=-1)

    def encoded_bits_per_vector(self) -> int:
        return self.dimension * self.bit_width


class TorchMixedQuantizer:
    def __init__(self, plan: SplitPlan, seed: int, mode: str, dtype: torch.dtype = torch.float32) -> None:
        self.plan = plan
        self.mode = mode
        self.dtype = dtype
        self.outlier_quantizer = _build_quantizer_from_mode(
            mode,
            len(plan.outlier_indices),
            plan.outlier_bit_width,
            seed ^ 0x3C79_AC49_2BA7_B653,
            dtype,
        )
        self.regular_quantizer = _build_quantizer_from_mode(
            mode,
            len(plan.regular_indices),
            plan.regular_bit_width,
            seed ^ 0x1C69_B3F7_4AC4_AE35,
            dtype,
        )

    def quantize_tensor(self, tensor: torch.Tensor) -> MixedTensorCode:
        flat, shape = _flatten_last_dim(tensor)
        outlier_code = None
        regular_code = None

        if self.outlier_quantizer is not None:
            outlier_values = flat[:, list(self.plan.outlier_indices)]
            outlier_code = self.outlier_quantizer.quantize_tensor(outlier_values)
        if self.regular_quantizer is not None:
            regular_values = flat[:, list(self.plan.regular_indices)]
            regular_code = self.regular_quantizer.quantize_tensor(regular_values)

        return MixedTensorCode(
            outlier_code=outlier_code,
            regular_code=regular_code,
            plan=self.plan,
            original_shape=shape,
        )

    def dequantize_tensor(self, code: MixedTensorCode) -> torch.Tensor:
        flat = torch.zeros((math.prod(code.original_shape[:-1]), code.plan.dimension), dtype=self.dtype)
        if code.outlier_code is not None and self.outlier_quantizer is not None:
            outlier = self.outlier_quantizer.dequantize_tensor(code.outlier_code).reshape(flat.shape[0], -1)
            flat[:, list(code.plan.outlier_indices)] = outlier
        if code.regular_code is not None and self.regular_quantizer is not None:
            regular = self.regular_quantizer.dequantize_tensor(code.regular_code).reshape(flat.shape[0], -1)
            flat[:, list(code.plan.regular_indices)] = regular
        return flat.reshape(code.original_shape)

    def storage_bytes(self, code: MixedTensorCode) -> int:
        return code.storage_bytes()


def build_quantizer(
    spec: QuantizerSpec,
    dimension: int,
    seed: int,
    calibration_samples: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
):
    spec.validate()
    if spec.is_passthrough():
        return None
    if spec.kind == "mse":
        return TorchTurboQuantMse(dimension, spec.bit_width or 0, seed, dtype=dtype)
    if spec.kind == "prod":
        return TorchTurboQuantProd(dimension, spec.bit_width or 0, seed, dtype=dtype)
    if not spec.is_mixed():
        raise ValueError(f"unsupported quantizer kind: {spec.kind}")
    if calibration_samples is None:
        raise ValueError("mixed quantizers require calibration_samples")
    plan = SplitPlan.from_channel_rms(
        calibration_samples,
        spec.outlier_count or 0,
        spec.outlier_bit_width or 0,
        spec.regular_bit_width or 0,
    )
    mode = "prod" if spec.kind.endswith("prod") else "mse"
    return TorchMixedQuantizer(plan, seed, mode=mode, dtype=dtype)


def storage_bytes(code: MseTensorCode | ProdTensorCode | MixedTensorCode) -> int:
    return code.storage_bytes()


def _flatten_last_dim(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least one dimension")
    shape = tuple(int(part) for part in tensor.shape)
    # Stay on the input device — let downstream quantizers migrate their state if needed.
    flat = tensor.detach().to(torch.float32).reshape(-1, shape[-1])
    return flat, shape


def _bucketize_centroids(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    if boundaries.numel() <= 2:
        return torch.zeros_like(values, dtype=torch.long)
    inner = boundaries[1:-1]
    return torch.bucketize(values.clamp(-1.0, 1.0), inner)


def _build_quantizer_from_mode(mode: str, dimension: int, bit_width: int, seed: int, dtype: torch.dtype):
    if dimension == 0:
        return None
    if mode == "mse":
        return TorchTurboQuantMse(dimension, bit_width, seed, dtype=dtype)
    if mode == "prod":
        return TorchTurboQuantProd(dimension, bit_width, seed, dtype=dtype)
    raise ValueError(f"unsupported mode: {mode}")


def _gaussian_orthogonal_matrix(dimension: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    gaussian = torch.randn((dimension, dimension), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(0)


class _WalshHadamardRotation:
    """Structured Walsh-Hadamard rotation: signs_post * FWHT(signs_pre * x) * scale.

    Applied via O(d log d) in-place FWHT, NOT as a dense matrix multiply.
    Mirrors the Rust implementation in src/rotation.rs.
    """

    def __init__(self, dimension: int, seed: int, dtype: torch.dtype):
        self.dimension = dimension
        self.padded_dim = 1
        while self.padded_dim < dimension:
            self.padded_dim *= 2

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.signs_pre = (torch.randint(0, 2, (self.padded_dim,), generator=generator) * 2 - 1).to(dtype)
        self.signs_post = (torch.randint(0, 2, (self.padded_dim,), generator=generator) * 2 - 1).to(dtype)
        self.scale = 1.0 / (self.padded_dim ** 0.5)
        # Dummy .T attribute so code paths expecting `rotation.T` don't crash
        # (we override apply methods instead)
        self.T = self  # sentinel: will not be used for matmul

    def _migrate_to(self, device: torch.device) -> None:
        if self.signs_pre.device != device:
            self.signs_pre = self.signs_pre.to(device)
        if self.signs_post.device != device:
            self.signs_post = self.signs_post.to(device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward rotation: y = scale * signs_post * FWHT(signs_pre * x)."""
        self._migrate_to(x.device)
        return self._transform(x, self.signs_pre, self.signs_post)

    def apply_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation (WHT is self-adjoint, so swap pre/post signs)."""
        self._migrate_to(x.device)
        return self._transform(x, self.signs_post, self.signs_pre)

    def _transform(self, x: torch.Tensor, left_signs: torch.Tensor, right_signs: torch.Tensor) -> torch.Tensor:
        """Core FWHT transform: pad → left_signs → FWHT → right_signs * scale → truncate."""
        orig_shape = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])  # (batch, dim)
        batch = flat.shape[0]

        # Pad to power of 2
        if self.dimension < self.padded_dim:
            padded = torch.zeros(batch, self.padded_dim, dtype=flat.dtype, device=flat.device)
            padded[:, :self.dimension] = flat
        else:
            padded = flat.clone()

        # Multiply by left signs
        padded *= left_signs

        # Fast Walsh-Hadamard Transform in-place (iterative butterfly)
        stride = 1
        while stride < self.padded_dim:
            step = stride * 2
            for start in range(0, self.padded_dim, step):
                left = padded[:, start:start + stride]
                right = padded[:, start + stride:start + step]
                a = left.clone()
                b = right.clone()
                padded[:, start:start + stride] = a + b
                padded[:, start + stride:start + step] = a - b
            stride = step

        # Multiply by right signs and scale
        padded *= right_signs * self.scale

        # Truncate back to original dimension
        result = padded[:, :self.dimension]
        return result.reshape(*orig_shape, self.dimension)


def _gaussian_matrix(dimension: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn((dimension, dimension), generator=generator, dtype=dtype)


@lru_cache(maxsize=None)
def _solve_codebook(
    dimension: int,
    bit_width: int,
    max_iterations: int = DEFAULT_LLOYD_MAX_ITERATIONS,
    tolerance: float = DEFAULT_LLOYD_MAX_TOLERANCE,
    subdivisions: int = DEFAULT_LLOYD_MAX_SUBDIVISIONS,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if bit_width == 0:
        return (0.0,), (-1.0, 1.0)

    centroid_count = 1 << bit_width
    step = 2.0 / centroid_count
    centroids = [-1.0 + (index + 0.5) * step for index in range(centroid_count)]
    distribution = _SphereCoordinateDistribution(dimension)

    for _ in range(max_iterations):
        boundaries = _compute_boundaries(centroids)
        next_centroids = list(centroids)
        max_delta = 0.0
        for index in range(centroid_count):
            left = boundaries[index]
            right = boundaries[index + 1]
            mass = _integrate(distribution.pdf, left, right, subdivisions)
            if mass > 0.0:
                mean = _integrate(lambda value: value * distribution.pdf(value), left, right, subdivisions) / mass
            else:
                mean = 0.5 * (left + right)
            next_centroids[index] = max(-1.0, min(1.0, mean))
            max_delta = max(max_delta, abs(next_centroids[index] - centroids[index]))
        centroids = next_centroids
        if max_delta <= tolerance:
            break

    boundaries = _compute_boundaries(centroids)
    return tuple(centroids), tuple(boundaries)


def _compute_boundaries(centroids: list[float]) -> list[float]:
    boundaries = [-1.0]
    boundaries.extend(0.5 * (left + right) for left, right in zip(centroids, centroids[1:]))
    boundaries.append(1.0)
    return boundaries


class _SphereCoordinateDistribution:
    def __init__(self, dimension: int) -> None:
        self.log_normalizer = (
            math.lgamma(dimension / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((dimension - 1.0) / 2.0)
        )
        self.exponent = (dimension - 3.0) / 2.0

    def pdf(self, value: float) -> float:
        clamped = min(max(value, -1.0 + 1e-12), 1.0 - 1e-12)
        one_minus_square = max(1e-16, 1.0 - clamped * clamped)
        return math.exp(self.log_normalizer + self.exponent * math.log(one_minus_square))


def _integrate(function, start: float, end: float, subdivisions: int) -> float:
    if abs(end - start) <= float.fromhex("0x1.0p-52"):
        return 0.0
    steps = max(2, subdivisions)
    if steps % 2 == 1:
        steps += 1
    width = (end - start) / steps
    total = function(start) + function(end)
    for step in range(1, steps):
        point = start + step * width
        total += (4.0 if step % 2 == 1 else 2.0) * function(point)
    return total * width / 3.0
