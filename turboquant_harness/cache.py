"""HuggingFace Transformers Cache integration for TurboQuant.

Provides TurboQuantCache as a drop-in replacement for DynamicCache with
**real** KV cache compression — the cache is stored quantized and only
decompressed on-the-fly during attention computation.

Requires transformers >= 5.0 (QuantizedLayer / DynamicLayer API).

Usage::

    from turboquant_harness import TurboQuantCache

    cache = TurboQuantCache(model.config, nbits=4)
    output = model.generate(input_ids, past_key_values=cache)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, QuantizedLayer

from .quantization import TorchTurboQuantMse, TorchTurboQuantProd, MseTensorCode, ProdTensorCode
from .packing import pack_uint4, unpack_uint4, pack_uint2, unpack_uint2

CACHE_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_packer(nbits: int):
    if nbits == 4:
        return pack_uint4, unpack_uint4
    if nbits == 2:
        return pack_uint2, unpack_uint2
    return None, None


# ---------------------------------------------------------------------------
# TurboQuantLayer — real quantized storage via QuantizedLayer
# ---------------------------------------------------------------------------

class TurboQuantLayer(QuantizedLayer):
    """Single-layer quantized KV cache using TurboQuant MSE.

    Extends HF ``QuantizedLayer`` so the residual window and the
    quantize/dequantize lifecycle are handled by the parent.
    Storage is packed uint4/uint2 — real memory savings.
    """

    def __init__(
        self,
        dim: int = 128,
        nbits: int = 4,
        residual_length: int = 128,
        seed: int = 42,
        rotation: str = "dense_gaussian",
    ):
        super().__init__(
            nbits=nbits,
            axis_key=0,
            axis_value=0,
            q_group_size=dim,
            residual_length=residual_length,
        )
        self._quantizer = TorchTurboQuantMse(
            dim, nbits, seed, dtype=torch.float32, rotation=rotation,
        )
        self._pack, self._unpack = _get_packer(nbits)

    def _quantize(self, tensor: torch.Tensor, axis: int):
        device = tensor.device
        code = self._quantizer.quantize_tensor(tensor.float())
        indices = code.indices
        if self._pack is not None:
            indices = self._pack(indices.to(torch.uint8))
        return (indices, code.norms, code.original_shape, tensor.dtype, device)

    def _dequantize(self, q_tensor):
        indices, norms, original_shape, original_dtype, device = q_tensor
        if self._unpack is not None:
            indices = self._unpack(indices).to(torch.int16)
        code = MseTensorCode(indices=indices, norms=norms, original_shape=original_shape)
        return self._quantizer.dequantize_tensor(code).to(original_dtype).to(device)


class TurboQuantProdLayer(QuantizedLayer):
    """Single-layer quantized KV cache using TurboQuant Prod (MSE + QJL).

    Better inner-product preservation at the cost of slightly higher storage.
    """

    def __init__(
        self,
        dim: int = 128,
        nbits: int = 4,
        residual_length: int = 128,
        seed: int = 42,
        rotation: str = "dense_gaussian",
    ):
        super().__init__(
            nbits=nbits,
            axis_key=0,
            axis_value=0,
            q_group_size=dim,
            residual_length=residual_length,
        )
        self._quantizer = TorchTurboQuantProd(
            dim, nbits, seed, dtype=torch.float32, rotation=rotation,
        )
        mse_bits = nbits - 1
        self._mse_pack, self._mse_unpack = _get_packer(mse_bits)

    def _quantize(self, tensor: torch.Tensor, axis: int):
        device = tensor.device
        code = self._quantizer.quantize_tensor(tensor.float())
        mse_idx = code.mse_indices
        if self._mse_pack is not None:
            mse_idx = self._mse_pack(mse_idx.to(torch.uint8))
        return (mse_idx, code.qjl_signs, code.norms, code.residual_norms,
                code.original_shape, tensor.dtype, device)

    def _dequantize(self, q_tensor):
        mse_idx, qjl_signs, norms, residual_norms, original_shape, original_dtype, device = q_tensor
        if self._mse_unpack is not None:
            mse_idx = self._mse_unpack(mse_idx).to(torch.int16)
        code = ProdTensorCode(
            mse_indices=mse_idx, qjl_signs=qjl_signs,
            norms=norms, residual_norms=residual_norms,
            original_shape=original_shape,
        )
        return self._quantizer.dequantize_tensor(code).to(original_dtype).to(device)


# ---------------------------------------------------------------------------
# TurboQuantCache — multi-layer container
# ---------------------------------------------------------------------------

class TurboQuantCache(Cache):
    """Drop-in replacement for DynamicCache with **real** TurboQuant compression.

    The KV cache is stored in packed quantized form. Only the most recent
    ``residual_length`` tokens are kept in full precision. Older tokens are
    decompressed on-the-fly when the model computes attention.

    Usage::

        cache = TurboQuantCache(model.config, nbits=4)
        output = model.generate(input_ids, past_key_values=cache)

    Args:
        config: HuggingFace model config.
        nbits: Bits per coordinate (2 or 4).
        residual_length: Recent tokens kept at full precision.
        base_seed: Base seed for per-layer rotation matrices.
        skip_layers: Layer indices to keep in full precision.
        mode: ``"mse"`` (default) or ``"prod"``.
        rotation: ``"dense_gaussian"`` (paper-exact) or ``"walsh_hadamard"``.
    """

    def __init__(
        self,
        config,
        nbits: int = 4,
        residual_length: int = 128,
        base_seed: int = 42,
        skip_layers: set[int] | None = None,
        mode: str = "mse",
        rotation: str = "dense_gaussian",
    ):
        text_config = (
            config.get_text_config(decoder=True)
            if hasattr(config, "get_text_config")
            else config
        )
        num_layers = text_config.num_hidden_layers
        head_dim = getattr(text_config, "head_dim", None) or (
            text_config.hidden_size // text_config.num_attention_heads
        )

        if skip_layers is None:
            skip_layers = {0}

        self._save_meta = {
            "nbits": nbits,
            "residual_length": residual_length,
            "base_seed": base_seed,
            "skip_layers": sorted(skip_layers),
            "mode": mode,
            "rotation": rotation,
            "head_dim": head_dim,
            "num_layers": num_layers,
        }

        layers = []
        for i in range(num_layers):
            if i in skip_layers:
                layers.append(DynamicLayer())
            else:
                if mode == "prod":
                    layers.append(TurboQuantProdLayer(
                        dim=head_dim, nbits=nbits,
                        residual_length=residual_length,
                        seed=base_seed + i, rotation=rotation,
                    ))
                else:
                    layers.append(TurboQuantLayer(
                        dim=head_dim, nbits=nbits,
                        residual_length=residual_length,
                        seed=base_seed + i, rotation=rotation,
                    ))
        super().__init__(layers=layers)

    @staticmethod
    def calibrate_skip_layers(
        model,
        tokenizer,
        calibration_text: str = "The quick brown fox jumps over the lazy dog.",
        norm_threshold: float = 5.0,
    ) -> set[int]:
        """Auto-detect layers with outlier KV norms that should skip quantization.

        Runs a single forward pass and identifies layers where key norms exceed
        ``norm_threshold`` times the median key norm.
        """
        inputs = tokenizer(calibration_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(inputs.input_ids, use_cache=True)

        cache = out.past_key_values
        norms: list[float] = []

        if isinstance(cache, DynamicCache):
            for layer in cache.layers:
                k = getattr(layer, "keys", None)
                if k is not None and k.numel() > 0:
                    norms.append(k.float().norm(dim=-1).mean().item())
                else:
                    norms.append(0.0)
        elif isinstance(cache, Cache):
            for layer in cache.layers:
                k = getattr(layer, "keys", None)
                if k is not None and k.numel() > 0:
                    norms.append(k.float().norm(dim=-1).mean().item())
                else:
                    norms.append(0.0)
        elif isinstance(cache, (tuple, list)):
            for entry in cache:
                if isinstance(entry, (tuple, list)) and len(entry) >= 1:
                    k = entry[0]
                    if k is not None and k.numel() > 0:
                        norms.append(k.float().norm(dim=-1).mean().item())
                    else:
                        norms.append(0.0)

        if not norms:
            return set()
        median_norm = sorted(norms)[len(norms) // 2]
        if median_norm == 0:
            return set()
        return {i for i, n in enumerate(norms) if n > norm_threshold * median_norm}

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def save_to_disk(self, path: str | os.PathLike) -> None:
        """Save the full cache state to a single ``.pt`` file.

        Tensors are moved to CPU during serialization. The resulting file
        is portable across machines and PyTorch versions and can be
        reloaded with :meth:`load_from_disk` on a different host.

        Args:
            path: File path. Parent directory must exist.

        Notes:
            - Format uses ``torch.save`` (pickle-based). Only load files
              from trusted sources.
            - Storage size scales with the quantized payload, not the
              dequantized one. A 4-bit cache of an 8 GB FP16 prefix will
              be roughly 1.1 GB on disk.
        """
        state = {
            "format_version": CACHE_FORMAT_VERSION,
            "config": dict(self._save_meta),
            "layers": [_serialize_layer(layer) for layer in self.layers],
        }
        torch.save(state, str(path))

    @classmethod
    def load_from_disk(
        cls,
        path: str | os.PathLike,
        model_config=None,
        map_location: str | torch.device = "cpu",
    ) -> "TurboQuantCache":
        """Reconstruct a cache previously saved with :meth:`save_to_disk`.

        Args:
            path: File path produced by ``save_to_disk``.
            model_config: Optional HF model config. If provided, layer
                count and head dim are validated against the saved
                metadata. Otherwise the saved metadata is taken as
                authoritative.
            map_location: Where tensors should land. Pass ``"cuda"`` (or
                a specific ``"cuda:0"``) to materialise straight on GPU
                and skip a CPU round-trip.
        """
        state = torch.load(str(path), map_location=map_location, weights_only=False)
        version = state.get("format_version")
        if version != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported cache format version {version!r}; "
                f"this build expects {CACHE_FORMAT_VERSION}."
            )
        cfg = state["config"]

        if model_config is not None:
            text_cfg = (
                model_config.get_text_config(decoder=True)
                if hasattr(model_config, "get_text_config")
                else model_config
            )
            saved = cfg["num_layers"]
            actual = text_cfg.num_hidden_layers
            if saved != actual:
                raise ValueError(
                    f"Layer count mismatch: saved cache has {saved} layers, "
                    f"model config has {actual}."
                )
            saved_head = cfg["head_dim"]
            actual_head = getattr(text_cfg, "head_dim", None) or (
                text_cfg.hidden_size // text_cfg.num_attention_heads
            )
            if saved_head != actual_head:
                raise ValueError(
                    f"head_dim mismatch: saved cache uses {saved_head}, "
                    f"model config has {actual_head}."
                )
            build_config = model_config
        else:
            build_config = _StubConfig(cfg["num_layers"], cfg["head_dim"])

        cache = cls(
            build_config,
            nbits=cfg["nbits"],
            residual_length=cfg["residual_length"],
            base_seed=cfg["base_seed"],
            skip_layers=set(cfg["skip_layers"]),
            mode=cfg["mode"],
            rotation=cfg["rotation"],
        )

        target_device = torch.device(map_location) if isinstance(map_location, str) else map_location
        for layer, layer_state in zip(cache.layers, state["layers"]):
            _restore_layer(layer, layer_state, target_device)
        return cache


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

class _StubConfig:
    """Minimal stand-in for an HF config when reloading without a real model."""

    def __init__(self, num_layers: int, head_dim: int):
        self.num_hidden_layers = num_layers
        self.head_dim = head_dim
        self.hidden_size = head_dim
        self.num_attention_heads = 1

    def get_text_config(self, decoder: bool = True):
        return self


_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def _dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype)


def _dtype_from_str(s: str) -> torch.dtype:
    if s in _DTYPE_MAP:
        return _DTYPE_MAP[s]
    name = s.split(".")[-1]
    if hasattr(torch, name):
        return getattr(torch, name)
    raise ValueError(f"Unknown dtype string: {s!r}")


def _tensor_or_none(t):
    if t is None:
        return None
    return t.detach().cpu().contiguous()


def _serialize_mse_quant(t):
    if t is None:
        return None
    indices, norms, original_shape, original_dtype, _device = t
    return {
        "indices": _tensor_or_none(indices),
        "norms": _tensor_or_none(norms),
        "original_shape": list(original_shape),
        "original_dtype": _dtype_to_str(original_dtype),
    }


def _serialize_prod_quant(t):
    if t is None:
        return None
    mse_idx, qjl_signs, norms, residual_norms, original_shape, original_dtype, _device = t
    return {
        "mse_indices": _tensor_or_none(mse_idx),
        "qjl_signs": _tensor_or_none(qjl_signs),
        "norms": _tensor_or_none(norms),
        "residual_norms": _tensor_or_none(residual_norms),
        "original_shape": list(original_shape),
        "original_dtype": _dtype_to_str(original_dtype),
    }


def _serialize_layer(layer):
    keys = layer.keys.detach().cpu() if getattr(layer, "is_initialized", False) and layer.keys is not None else None
    values = layer.values.detach().cpu() if getattr(layer, "is_initialized", False) and layer.values is not None else None
    cumulative = getattr(layer, "cumulative_length", 0)
    if isinstance(cumulative, torch.Tensor):
        cumulative = int(cumulative.item())

    if isinstance(layer, TurboQuantProdLayer):
        return {
            "kind": "turboquant_prod",
            "quantized_keys": _serialize_prod_quant(getattr(layer, "_quantized_keys", None)),
            "quantized_values": _serialize_prod_quant(getattr(layer, "_quantized_values", None)),
            "keys": keys,
            "values": values,
            "cumulative_length": cumulative,
        }
    if isinstance(layer, TurboQuantLayer):
        return {
            "kind": "turboquant_mse",
            "quantized_keys": _serialize_mse_quant(getattr(layer, "_quantized_keys", None)),
            "quantized_values": _serialize_mse_quant(getattr(layer, "_quantized_values", None)),
            "keys": keys,
            "values": values,
            "cumulative_length": cumulative,
        }
    return {
        "kind": "dynamic",
        "keys": keys,
        "values": values,
        "cumulative_length": cumulative,
    }


def _deserialize_mse_quant(d, device):
    if d is None:
        return None
    return (
        d["indices"].to(device),
        d["norms"].to(device),
        tuple(d["original_shape"]),
        _dtype_from_str(d["original_dtype"]),
        device,
    )


def _deserialize_prod_quant(d, device):
    if d is None:
        return None
    return (
        d["mse_indices"].to(device),
        d["qjl_signs"].to(device),
        d["norms"].to(device),
        d["residual_norms"].to(device),
        tuple(d["original_shape"]),
        _dtype_from_str(d["original_dtype"]),
        device,
    )


def _restore_layer(layer, state, device: torch.device):
    keys = state.get("keys")
    values = state.get("values")

    if keys is not None:
        layer.dtype = keys.dtype
        layer.device = device
        layer.keys = keys.to(device)
        layer.values = values.to(device)
        layer.is_initialized = True

    cumulative = state.get("cumulative_length", 0)
    if hasattr(layer, "cumulative_length"):
        if isinstance(layer.cumulative_length, torch.Tensor):
            layer.cumulative_length.fill_(cumulative)
        else:
            layer.cumulative_length = cumulative

    kind = state["kind"]
    if kind == "turboquant_mse":
        layer._quantized_keys = _deserialize_mse_quant(state["quantized_keys"], device)
        layer._quantized_values = _deserialize_mse_quant(state["quantized_values"], device)
    elif kind == "turboquant_prod":
        layer._quantized_keys = _deserialize_prod_quant(state["quantized_keys"], device)
        layer._quantized_values = _deserialize_prod_quant(state["quantized_values"], device)
    elif kind != "dynamic":
        raise ValueError(f"Unknown layer kind: {kind!r}")
