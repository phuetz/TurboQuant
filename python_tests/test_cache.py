"""Tests for TurboQuant HF Cache integration, bit packing, and Walsh-Hadamard."""

import unittest
import torch


class PackingTests(unittest.TestCase):

    def test_uint4_round_trip(self):
        from turboquant_harness.packing import pack_uint4, unpack_uint4
        indices = torch.randint(0, 16, (3, 5, 128), dtype=torch.uint8)
        packed = pack_uint4(indices)
        self.assertEqual(packed.shape, (3, 5, 64))
        recovered = unpack_uint4(packed)
        self.assertTrue(torch.equal(indices, recovered))

    def test_uint2_round_trip(self):
        from turboquant_harness.packing import pack_uint2, unpack_uint2
        indices = torch.randint(0, 4, (2, 4, 128), dtype=torch.uint8)
        packed = pack_uint2(indices)
        self.assertEqual(packed.shape, (2, 4, 32))
        recovered = unpack_uint2(packed)
        self.assertTrue(torch.equal(indices, recovered))

    def test_uint4_storage_savings(self):
        from turboquant_harness.packing import pack_uint4
        indices = torch.randint(0, 16, (1024, 128), dtype=torch.uint8)
        packed = pack_uint4(indices)
        self.assertEqual(packed.numel(), indices.numel() // 2)

    def test_uint2_storage_savings(self):
        from turboquant_harness.packing import pack_uint2
        indices = torch.randint(0, 4, (1024, 128), dtype=torch.uint8)
        packed = pack_uint2(indices)
        self.assertEqual(packed.numel(), indices.numel() // 4)


class WalshHadamardTests(unittest.TestCase):

    def test_wht_rotation_preserves_norm(self):
        from turboquant_harness.quantization import TorchTurboQuantMse
        q = TorchTurboQuantMse(16, 2, seed=42, rotation="walsh_hadamard")
        v = torch.randn(16)
        v_unit = v / v.norm()
        rotated = q._rotate(v_unit.unsqueeze(0)).squeeze(0)
        self.assertAlmostEqual(rotated.norm().item(), 1.0, places=4)

    def test_wht_quantizer_round_trip_finite(self):
        from turboquant_harness.quantization import TorchTurboQuantMse
        q = TorchTurboQuantMse(32, 4, seed=7, rotation="walsh_hadamard")
        tensor = torch.randn(2, 4, 32)
        code = q.quantize_tensor(tensor)
        decoded = q.dequantize_tensor(code)
        self.assertEqual(decoded.shape, tensor.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_wht_mse_comparable_to_gaussian(self):
        from turboquant_harness.quantization import TorchTurboQuantMse
        torch.manual_seed(99)
        tensor = torch.randn(100, 64)
        q_gauss = TorchTurboQuantMse(64, 4, seed=7, rotation="dense_gaussian")
        q_wht = TorchTurboQuantMse(64, 4, seed=7, rotation="walsh_hadamard")
        dec_g = q_gauss.dequantize_tensor(q_gauss.quantize_tensor(tensor))
        dec_w = q_wht.dequantize_tensor(q_wht.quantize_tensor(tensor))
        mse_g = (tensor - dec_g).pow(2).mean().item()
        mse_w = (tensor - dec_w).pow(2).mean().item()
        self.assertLess(mse_w, mse_g * 3.0)
        self.assertGreater(mse_w, 0.0)


class CacheConstructionTests(unittest.TestCase):

    def _mock_config(self, num_layers=4, hidden_size=64, num_heads=2, head_dim=None):
        class C:
            pass
        c = C()
        c.num_hidden_layers = num_layers
        c.hidden_size = hidden_size
        c.num_attention_heads = num_heads
        if head_dim is not None:
            c.head_dim = head_dim
        return c

    def test_cache_creates_correct_layer_count(self):
        from turboquant_harness.cache import TurboQuantCache, TurboQuantLayer, DynamicLayer
        config = self._mock_config(num_layers=4, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=4, skip_layers={0})
        self.assertEqual(len(cache.layers), 4)
        self.assertIsInstance(cache.layers[0], DynamicLayer)
        self.assertIsInstance(cache.layers[1], TurboQuantLayer)

    def test_cache_explicit_head_dim(self):
        from turboquant_harness.cache import TurboQuantCache, TurboQuantLayer
        config = self._mock_config(num_layers=2, hidden_size=2048, num_heads=16, head_dim=256)
        cache = TurboQuantCache(config, nbits=4, skip_layers=set())
        layer = cache.layers[0]
        self.assertIsInstance(layer, TurboQuantLayer)
        self.assertEqual(layer._quantizer.dimension, 256)

    def test_cache_update_and_get_seq_length(self):
        from turboquant_harness.cache import TurboQuantCache
        config = self._mock_config(num_layers=2, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=4, residual_length=8, skip_layers=set())
        keys = torch.randn(1, 2, 4, 32)
        vals = torch.randn(1, 2, 4, 32)
        out_k, out_v = cache.update(keys, vals, layer_idx=0)
        self.assertEqual(out_k.shape[-2], 4)
        self.assertEqual(cache.get_seq_length(), 4)

    def test_cache_quantizes_on_overflow(self):
        from turboquant_harness.cache import TurboQuantCache, TurboQuantLayer
        config = self._mock_config(num_layers=2, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=4, residual_length=4, skip_layers=set())

        # Push 6 tokens in 3 batches of 2
        for _ in range(3):
            keys = torch.randn(1, 2, 2, 32)
            vals = torch.randn(1, 2, 2, 32)
            out_k, out_v = cache.update(keys, vals, layer_idx=0)

        # Layer should have quantized prefix + residual
        layer = cache.layers[0]
        self.assertIsNotNone(layer._quantized_keys)
        self.assertEqual(out_k.shape[-2], 6)  # Full materialized sequence
        self.assertTrue(torch.isfinite(out_k).all())
        self.assertTrue(torch.isfinite(out_v).all())

    def test_cache_skip_layer_no_quantization(self):
        from turboquant_harness.cache import TurboQuantCache, DynamicLayer
        config = self._mock_config(num_layers=2, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=4, residual_length=2, skip_layers={0})

        for _ in range(5):
            keys = torch.randn(1, 2, 2, 32)
            vals = torch.randn(1, 2, 2, 32)
            cache.update(keys, vals, layer_idx=0)

        layer0 = cache.layers[0]
        self.assertIsInstance(layer0, DynamicLayer)
        # Skip layer keeps everything in full precision
        self.assertEqual(cache.get_seq_length(), 10)

    def test_cache_prod_mode(self):
        from turboquant_harness.cache import TurboQuantCache, TurboQuantProdLayer
        config = self._mock_config(num_layers=2, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=4, residual_length=4, skip_layers=set(), mode="prod")
        self.assertIsInstance(cache.layers[0], TurboQuantProdLayer)

        for _ in range(4):
            keys = torch.randn(1, 2, 2, 32)
            vals = torch.randn(1, 2, 2, 32)
            out_k, out_v = cache.update(keys, vals, layer_idx=0)

        self.assertEqual(out_k.shape[-2], 8)
        self.assertTrue(torch.isfinite(out_k).all())

    def test_cache_2bit(self):
        from turboquant_harness.cache import TurboQuantCache
        config = self._mock_config(num_layers=2, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=2, residual_length=4, skip_layers=set())

        for _ in range(4):
            keys = torch.randn(1, 2, 2, 32)
            vals = torch.randn(1, 2, 2, 32)
            out_k, out_v = cache.update(keys, vals, layer_idx=0)

        self.assertEqual(out_k.shape[-2], 8)
        self.assertTrue(torch.isfinite(out_k).all())

    def test_cache_wht_mode(self):
        from turboquant_harness.cache import TurboQuantCache
        config = self._mock_config(num_layers=2, hidden_size=64, num_heads=2)
        cache = TurboQuantCache(config, nbits=4, residual_length=4, skip_layers=set(), rotation="walsh_hadamard")

        for _ in range(4):
            keys = torch.randn(1, 2, 2, 32)
            vals = torch.randn(1, 2, 2, 32)
            out_k, out_v = cache.update(keys, vals, layer_idx=0)

        self.assertEqual(out_k.shape[-2], 8)
        self.assertTrue(torch.isfinite(out_k).all())

    def test_layer_quantize_dequantize_round_trip(self):
        from turboquant_harness.cache import TurboQuantLayer
        layer = TurboQuantLayer(dim=32, nbits=4, residual_length=8, seed=7)
        tensor = torch.randn(1, 4, 6, 32)
        q = layer._quantize(tensor, axis=0)
        recovered = layer._dequantize(q)
        self.assertEqual(recovered.shape, tensor.shape)
        self.assertTrue(torch.isfinite(recovered).all())
        # Verify reconstruction is reasonable (not zeros)
        self.assertGreater(recovered.abs().mean().item(), 0.01)

    def test_prod_layer_round_trip(self):
        from turboquant_harness.cache import TurboQuantProdLayer
        layer = TurboQuantProdLayer(dim=32, nbits=4, residual_length=8, seed=7)
        tensor = torch.randn(1, 2, 4, 32)
        q = layer._quantize(tensor, axis=0)
        recovered = layer._dequantize(q)
        self.assertEqual(recovered.shape, tensor.shape)
        self.assertTrue(torch.isfinite(recovered).all())


if __name__ == "__main__":
    unittest.main()
