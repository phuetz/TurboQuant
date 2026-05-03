"""Tests for TurboQuantCache.save_to_disk / load_from_disk."""

import os
import tempfile
import unittest

import torch
from transformers import AutoConfig

from turboquant_harness import TurboQuantCache
from turboquant_harness.cache import (
    CACHE_FORMAT_VERSION,
    TurboQuantLayer,
    TurboQuantProdLayer,
)


class _FakeConfig:
    """Minimal HF-compatible config for unit-test KV shapes."""

    def __init__(self, num_layers=4, head_dim=64, num_heads=4):
        self.num_hidden_layers = num_layers
        self.head_dim = head_dim
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads
        self.hidden_size = head_dim * num_heads

    def get_text_config(self, decoder=True):
        return self


def _populate_cache(cache: TurboQuantCache, batch=1, num_kv_heads=4, head_dim=64,
                    seq=200, dtype=torch.float32) -> tuple[list, list]:
    """Push synthetic KV through every layer until a full quantize-and-flush
    cycle has happened, then return the full per-layer K/V tensors that the
    cache "saw" (for ground-truth comparison).
    """
    torch.manual_seed(0)
    full_keys = []
    full_values = []
    for i, layer in enumerate(cache.layers):
        k = torch.randn(batch, num_kv_heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, num_kv_heads, seq, head_dim, dtype=dtype)
        layer.update(k, v)
        full_keys.append(k)
        full_values.append(v)
    return full_keys, full_values


class CacheSaveLoadTests(unittest.TestCase):
    """Round-trip and structural tests for cache persistence."""

    def test_save_load_round_trip_mse(self):
        cfg = _FakeConfig(num_layers=4, head_dim=64, num_heads=4)
        cache = TurboQuantCache(cfg, nbits=4, residual_length=16, mode="mse")
        _populate_cache(cache, head_dim=64, seq=80)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache.save_to_disk(p)
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 0)

            reloaded = TurboQuantCache.load_from_disk(p, model_config=cfg)

        # Same number of layers, same kinds
        self.assertEqual(len(reloaded.layers), len(cache.layers))
        for orig, rl in zip(cache.layers, reloaded.layers):
            self.assertEqual(type(orig), type(rl))

        # cumulative_length preserved
        for orig, rl in zip(cache.layers, reloaded.layers):
            self.assertEqual(orig.get_seq_length(), rl.get_seq_length())

        # Quantized payload bytes preserved
        for orig, rl in zip(cache.layers, reloaded.layers):
            if isinstance(orig, TurboQuantLayer):
                # _quantized_keys = (indices, norms, shape, dtype, device)
                self.assertTrue(torch.equal(orig._quantized_keys[0].cpu(),
                                            rl._quantized_keys[0].cpu()))
                self.assertTrue(torch.equal(orig._quantized_values[0].cpu(),
                                            rl._quantized_values[0].cpu()))

    def test_save_load_round_trip_prod(self):
        cfg = _FakeConfig(num_layers=3, head_dim=64, num_heads=4)
        cache = TurboQuantCache(cfg, nbits=4, residual_length=16, mode="prod")
        _populate_cache(cache, head_dim=64, seq=80)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache_prod.pt")
            cache.save_to_disk(p)
            reloaded = TurboQuantCache.load_from_disk(p, model_config=cfg)

        for orig, rl in zip(cache.layers, reloaded.layers):
            self.assertEqual(type(orig), type(rl))
            if isinstance(orig, TurboQuantProdLayer):
                self.assertTrue(torch.equal(orig._quantized_keys[0].cpu(),
                                            rl._quantized_keys[0].cpu()))

    def test_load_without_config_uses_stub(self):
        """Loading without an HF config should still work and produce
        a usable cache with the correct shape/skill set."""
        cfg = _FakeConfig(num_layers=4, head_dim=32, num_heads=4)
        cache = TurboQuantCache(cfg, nbits=4, residual_length=16)
        _populate_cache(cache, head_dim=32, seq=64)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache.save_to_disk(p)
            rl = TurboQuantCache.load_from_disk(p)  # no config

        self.assertEqual(len(rl.layers), 4)
        self.assertEqual(rl._save_meta["head_dim"], 32)
        self.assertEqual(rl._save_meta["nbits"], 4)

    def test_layer_count_mismatch_raises(self):
        cfg_save = _FakeConfig(num_layers=4, head_dim=32, num_heads=4)
        cfg_wrong = _FakeConfig(num_layers=8, head_dim=32, num_heads=4)
        cache = TurboQuantCache(cfg_save, nbits=4, residual_length=16)
        _populate_cache(cache, head_dim=32, seq=48)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache.save_to_disk(p)
            with self.assertRaises(ValueError):
                TurboQuantCache.load_from_disk(p, model_config=cfg_wrong)

    def test_format_version_in_state(self):
        cfg = _FakeConfig(num_layers=2, head_dim=32, num_heads=4)
        cache = TurboQuantCache(cfg, nbits=4, residual_length=16)
        _populate_cache(cache, head_dim=32, seq=48)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache.save_to_disk(p)
            raw = torch.load(p, weights_only=False)

        self.assertEqual(raw["format_version"], CACHE_FORMAT_VERSION)
        self.assertIn("config", raw)
        self.assertIn("layers", raw)
        self.assertEqual(len(raw["layers"]), 2)

    def test_dequantize_consistency_after_reload(self):
        """The dequantized prefix should byte-equal the original after
        a save/load cycle (same seed, same indices, same codebook)."""
        cfg = _FakeConfig(num_layers=2, head_dim=64, num_heads=4)
        cache = TurboQuantCache(cfg, nbits=4, residual_length=16)
        _populate_cache(cache, head_dim=64, seq=64)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache.save_to_disk(p)
            rl = TurboQuantCache.load_from_disk(p, model_config=cfg)

        for orig, rec in zip(cache.layers, rl.layers):
            if isinstance(orig, TurboQuantLayer):
                deq_orig_k = orig._dequantize(orig._quantized_keys)
                deq_rec_k = rec._dequantize(rec._quantized_keys)
                self.assertTrue(torch.equal(deq_orig_k, deq_rec_k),
                                "Dequantized keys should match exactly across save/load")

    def test_disk_size_smaller_than_fp16_estimate(self):
        """A 4-bit quantized 200-token cache should land well below the
        equivalent FP16 raw payload (sanity check, not the headline ratio)."""
        cfg = _FakeConfig(num_layers=8, head_dim=128, num_heads=4)
        seq = 1024
        cache = TurboQuantCache(cfg, nbits=4, residual_length=128)
        _populate_cache(cache, num_kv_heads=4, head_dim=128, seq=seq, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache.save_to_disk(p)
            disk_bytes = os.path.getsize(p)

        # Equivalent FP16 raw payload (K and V, all layers, all tokens)
        fp16_bytes = 2 * 8 * 4 * seq * 128 * 2  # 2 (K+V) * layers * heads * seq * head_dim * fp16
        self.assertLess(disk_bytes, fp16_bytes,
                        f"On-disk size {disk_bytes} should be < raw FP16 {fp16_bytes}")


class ForwardPassRoundTripTest(unittest.TestCase):
    """Integration test: with a real model, save+load must produce a
    bitwise-identical forward pass on the next-token logits.

    Uses a tiny config (Qwen2.5-1.5B already cached on disk) and a single
    forward pass (no `.generate()` continuation, whose position-id
    semantics are brittle when given a pre-filled cache).
    """

    def test_forward_pass_logits_match_after_save_load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:  # pragma: no cover
            self.skipTest("transformers not installed")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B",
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B", local_files_only=True
            )
        except Exception as e:  # pragma: no cover
            self.skipTest(f"Qwen2.5-1.5B weights not cached locally: {e}")

        tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        prompt = "The quick brown fox jumps over the lazy dog and runs away"
        ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Prefill the original cache with the full prompt.
        cache_orig = TurboQuantCache(model.config, nbits=4, residual_length=8)
        with torch.no_grad():
            model(ids, past_key_values=cache_orig, use_cache=True)

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "cache.pt")
            cache_orig.save_to_disk(p)
            cache_loaded = TurboQuantCache.load_from_disk(p, model_config=model.config)

        # Append the same single token through both caches and compare logits.
        # If the cache state (quantized prefix + residual + cumulative_length)
        # is restored faithfully, the forward pass logits must match bitwise.
        new_token = ids[:, -1:].clone()
        with torch.no_grad():
            out_orig = model(new_token, past_key_values=cache_orig, use_cache=True)
            out_loaded = model(new_token, past_key_values=cache_loaded, use_cache=True)

        self.assertTrue(
            torch.equal(out_orig.logits, out_loaded.logits),
            "Forward-pass logits must be bitwise-equal across save/load",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
