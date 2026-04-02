from __future__ import annotations

import unittest

import torch

from turboquant_harness.hf_runner import TurboQuantKVBackend
from turboquant_harness.longbench import (
    _classification_score,
    _rouge_l_f1,
    _token_f1,
    build_longbench_prompt,
    score_prediction,
)
from turboquant_harness.needle import build_needle_prompt
from turboquant_harness.quantization import (
    QuantizerSpec,
    SplitPlan,
    TorchTurboQuantMse,
    parse_quantizer_spec,
)


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        ids = [slot + 1 for slot, _ in enumerate(text.split())]

        class _Result:
            input_ids = ids

        return _Result()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{token}" for token in ids)


class HarnessTests(unittest.TestCase):
    def test_parse_quantizer_spec(self):
        spec = parse_quantizer_spec("mixed_prod:32:3:2")
        self.assertEqual(spec.kind, "mixed_prod")
        self.assertEqual(spec.outlier_count, 32)
        self.assertEqual(spec.outlier_bit_width, 3)
        self.assertEqual(spec.regular_bit_width, 2)
        self.assertTrue(spec.is_mixed())

        none_spec = parse_quantizer_spec("none")
        self.assertTrue(none_spec.is_passthrough())

    def test_split_plan_effective_bits(self):
        plan = SplitPlan.new(8, [0, 2, 4, 6], 3, 2)
        self.assertAlmostEqual(plan.effective_bit_width(), 2.5)

    def test_turboquant_mse_round_trip_shape(self):
        quantizer = TorchTurboQuantMse(8, 2, seed=7)
        tensor = torch.randn(2, 3, 8)
        code = quantizer.quantize_tensor(tensor)
        decoded = quantizer.dequantize_tensor(code)
        self.assertEqual(decoded.shape, tensor.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_turboquant_kv_backend_round_trip_shape(self):
        backend = TurboQuantKVBackend("prod:2", "mse:2", seed=7)
        key = torch.randn(1, 2, 4, 8)
        value = torch.randn(1, 2, 4, 8)
        state = backend.quantize(((key, value),))
        materialized = backend.materialize(state)
        self.assertEqual(materialized[0][0].shape, key.shape)
        self.assertEqual(materialized[0][1].shape, value.shape)
        self.assertTrue(torch.isfinite(materialized[0][0]).all())
        self.assertTrue(torch.isfinite(materialized[0][1]).all())

    def test_needle_prompt_contains_question(self):
        tokenizer = _FakeTokenizer()
        prompt = build_needle_prompt(
            tokenizer=tokenizer,
            context_tokens=32,
            depth_percent=50.0,
            needle_key="server id",
            needle_value="alpha-7",
            filler_text="filler text",
        )
        self.assertIn("Question: What is the server id?", prompt)
        self.assertIn("Answer:", prompt)

    def test_longbench_prompt_fallback_uses_context_and_input(self):
        prompt = build_longbench_prompt({"context": "ctx", "input": "what?"})
        self.assertIn("Context:", prompt)
        self.assertIn("Question:", prompt)

    def test_longbench_scoring_helpers(self):
        self.assertAlmostEqual(_token_f1("Paris", "Paris"), 1.0)
        self.assertGreater(_rouge_l_f1("a b c", "a c"), 0.0)
        self.assertEqual(_classification_score("The answer is sports", ["sports"], ["sports", "news"]), 1.0)
        self.assertEqual(score_prediction("42", ["42"], "exact_match"), 1.0)


if __name__ == "__main__":
    unittest.main()
