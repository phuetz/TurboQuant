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

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, QuantizedLayer

from .quantization import TorchTurboQuantMse, TorchTurboQuantProd, MseTensorCode, ProdTensorCode
from .packing import pack_uint4, unpack_uint4, pack_uint2, unpack_uint2


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
