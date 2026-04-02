from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .quantization import QuantizerSpec, build_quantizer, parse_quantizer_spec, storage_bytes


@dataclass
class QuantizedTensorEnvelope:
    payload: Any
    dtype: torch.dtype
    device: torch.device
    shape: tuple[int, ...]

    def storage_bytes(self) -> int:
        if isinstance(self.payload, torch.Tensor):
            return self.payload.numel() * self.payload.element_size()
        return storage_bytes(self.payload)


@dataclass
class QuantizedLayerCache:
    key: QuantizedTensorEnvelope
    value: QuantizedTensorEnvelope
    extras: tuple[Any, ...] = ()

    def storage_bytes(self) -> int:
        return self.key.storage_bytes() + self.value.storage_bytes()


@dataclass
class GenerationResult:
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: list[int]
    cache_bytes_per_step: list[int]
    backend_name: str


class BaseKVBackend(ABC):
    name: str

    @abstractmethod
    def quantize(self, past_key_values: Sequence[Sequence[torch.Tensor]]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def materialize(self, state: Any) -> Sequence[Sequence[torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def estimate_bytes(self, state: Any) -> int:
        raise NotImplementedError


class FullPrecisionKVBackend(BaseKVBackend):
    name = "full"

    def quantize(self, past_key_values: Sequence[Sequence[torch.Tensor]]) -> tuple[tuple[torch.Tensor, ...], ...]:
        return tuple(tuple(tensor.detach() for tensor in layer) for layer in _to_legacy_cache(past_key_values))

    def materialize(self, state: tuple[tuple[torch.Tensor, ...], ...]) -> tuple[tuple[torch.Tensor, ...], ...]:
        return state

    def estimate_bytes(self, state: tuple[tuple[torch.Tensor, ...], ...]) -> int:
        total = 0
        for layer in state:
            for tensor in layer:
                total += tensor.numel() * tensor.element_size()
        return total


class TurboQuantKVBackend(BaseKVBackend):
    def __init__(
        self,
        key_quantizer: QuantizerSpec | str,
        value_quantizer: QuantizerSpec | str,
        seed: int = 7,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.key_spec = parse_quantizer_spec(key_quantizer) if isinstance(key_quantizer, str) else key_quantizer
        self.value_spec = parse_quantizer_spec(value_quantizer) if isinstance(value_quantizer, str) else value_quantizer
        self.key_spec.validate()
        self.value_spec.validate()
        self.seed = seed
        self.dtype = dtype
        self._key_quantizers: dict[int, Any] = {}
        self._value_quantizers: dict[int, Any] = {}
        self.name = f"turboquant[k={self.key_spec.describe()},v={self.value_spec.describe()}]"

    def quantize(self, past_key_values: Sequence[Sequence[torch.Tensor]]) -> tuple[QuantizedLayerCache, ...]:
        legacy = _to_legacy_cache(past_key_values)
        layers = []
        for layer_index, layer in enumerate(legacy):
            if len(layer) < 2:
                raise ValueError("past_key_values layer must contain at least key and value tensors")
            key_tensor, value_tensor = layer[0], layer[1]
            key_quantizer = self._get_quantizer(self._key_quantizers, self.key_spec, key_tensor, layer_index, role_offset=0)
            value_quantizer = self._get_quantizer(
                self._value_quantizers, self.value_spec, value_tensor, layer_index, role_offset=10_000
            )
            layers.append(
                QuantizedLayerCache(
                    key=self._quantize_tensor(key_tensor, key_quantizer),
                    value=self._quantize_tensor(value_tensor, value_quantizer),
                    extras=tuple(item.detach() if isinstance(item, torch.Tensor) else item for item in layer[2:]),
                )
            )
        return tuple(layers)

    def materialize(self, state: tuple[QuantizedLayerCache, ...]) -> tuple[tuple[torch.Tensor, ...], ...]:
        materialized_layers = []
        for layer_index, layer in enumerate(state):
            key_quantizer = self._key_quantizers.get(layer_index)
            value_quantizer = self._value_quantizers.get(layer_index)
            key = self._materialize_tensor(layer.key, key_quantizer)
            value = self._materialize_tensor(layer.value, value_quantizer)
            materialized_layers.append((key, value, *layer.extras))
        return tuple(materialized_layers)

    def estimate_bytes(self, state: tuple[QuantizedLayerCache, ...]) -> int:
        return sum(layer.storage_bytes() for layer in state)

    def _get_quantizer(
        self,
        cache: dict[int, Any],
        spec: QuantizerSpec,
        tensor: torch.Tensor,
        layer_index: int,
        role_offset: int,
    ):
        if layer_index in cache:
            return cache[layer_index]
        if tensor.shape[-1] < 2:
            cache[layer_index] = None
            return None
        if spec.is_passthrough():
            cache[layer_index] = None
            return None
        calibration = tensor.detach().to(torch.float32).reshape(-1, tensor.shape[-1]).cpu() if spec.is_mixed() else None
        quantizer = build_quantizer(
            spec,
            dimension=tensor.shape[-1],
            seed=self.seed + layer_index + role_offset,
            calibration_samples=calibration,
            dtype=self.dtype,
        )
        cache[layer_index] = quantizer
        return quantizer

    def _quantize_tensor(self, tensor: torch.Tensor, quantizer) -> QuantizedTensorEnvelope:
        if quantizer is None:
            payload = tensor.detach().cpu()
        else:
            payload = quantizer.quantize_tensor(tensor.detach())
        return QuantizedTensorEnvelope(
            payload=payload,
            dtype=tensor.dtype,
            device=tensor.device,
            shape=tuple(int(part) for part in tensor.shape),
        )

    def _materialize_tensor(self, envelope: QuantizedTensorEnvelope, quantizer) -> torch.Tensor:
        if quantizer is None:
            tensor = envelope.payload
        else:
            tensor = quantizer.dequantize_tensor(envelope.payload)
        return tensor.to(device=envelope.device, dtype=envelope.dtype).reshape(envelope.shape)


def load_model_and_tokenizer(model_name: str, device: str = "cpu", dtype: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_arg = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype_arg)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_incremental(
    model,
    tokenizer,
    prompt: str,
    backend: BaseKVBackend,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop_strings: Sequence[str] = (),
) -> GenerationResult:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    generated_tokens: list[int] = []
    cache_bytes: list[int] = []

    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("model did not return past_key_values; use a decoder-only causal LM with use_cache=True")

        state = backend.quantize(past_key_values)
        cache_bytes.append(backend.estimate_bytes(state))
        next_token = _select_next_token(outputs.logits[:, -1, :], do_sample=do_sample, temperature=temperature, top_p=top_p)

        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            generated_tokens.append(token_id)

            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if eos_token_id is not None and token_id == eos_token_id:
                break
            if stop_strings and any(stop in generated_text for stop in stop_strings):
                break

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=model.device, dtype=attention_mask.dtype)],
                    dim=-1,
                )

            materialized = backend.materialize(state)
            outputs = model(
                input_ids=next_token.view(1, 1).to(model.device),
                attention_mask=attention_mask,
                past_key_values=materialized,
                use_cache=True,
            )
            state = backend.quantize(outputs.past_key_values)
            cache_bytes.append(backend.estimate_bytes(state))
            next_token = _select_next_token(outputs.logits[:, -1, :], do_sample=do_sample, temperature=temperature, top_p=top_p)

    return GenerationResult(
        prompt=prompt,
        generated_text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
        prompt_tokens=int(input_ids.shape[-1]),
        generated_tokens=generated_tokens,
        cache_bytes_per_step=cache_bytes,
        backend_name=backend.name,
    )


def _select_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if not do_sample:
        return torch.argmax(logits, dim=-1)

    if temperature <= 0.0:
        raise ValueError("temperature must be positive when sampling")
    scaled = logits / temperature
    probabilities = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sample = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sample).squeeze(-1)

    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def _resolve_dtype(dtype: str):
    text = dtype.strip().lower()
    if text == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if text not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[text]


def _to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values
