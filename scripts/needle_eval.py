from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_harness.hf_runner import FullPrecisionKVBackend, TurboQuantKVBackend, load_model_and_tokenizer
from turboquant_harness.needle import DEFAULT_FILLER_TEXT, run_needle_suite, write_needle_results_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Needle-In-A-Haystack evaluation with optional TurboQuant KV compression.")
    parser.add_argument("--model", required=True, help="Hugging Face causal LM identifier")
    parser.add_argument("--device", default="cpu", help="Device for inference, e.g. cpu or cuda")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto, float32, float16, bf16")
    parser.add_argument("--backend", choices=["full", "turboquant"], default="full")
    parser.add_argument("--key-quantizer", default="prod:3", help="Quantizer spec for K when backend=turboquant")
    parser.add_argument("--value-quantizer", default="mse:3", help="Quantizer spec for V when backend=turboquant")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-lengths", default="4096,8192,16384", help="Comma-separated token counts for the haystack context")
    parser.add_argument("--depths", default="10,30,50,70,90", help="Comma-separated insertion depths as percentages")
    parser.add_argument("--needle-key", default="secret code")
    parser.add_argument("--needle-value", default="47291")
    parser.add_argument("--filler-file", default="", help="Optional UTF-8 text file used as haystack filler")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="artifacts/needle_results.csv")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, dtype=args.dtype)
    backend = (
        FullPrecisionKVBackend()
        if args.backend == "full"
        else TurboQuantKVBackend(args.key_quantizer, args.value_quantizer, seed=args.seed)
    )

    filler_text = DEFAULT_FILLER_TEXT
    if args.filler_file:
        filler_text = Path(args.filler_file).read_text(encoding="utf-8")

    results = run_needle_suite(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        context_lengths=_parse_ints(args.context_lengths),
        depths=_parse_floats(args.depths),
        needle_key=args.needle_key,
        needle_value=args.needle_value,
        max_new_tokens=args.max_new_tokens,
        filler_text=filler_text,
    )
    write_needle_results_csv(args.output, results)

    accuracy = sum(result.correct for result in results) / len(results) if results else 0.0
    print(f"backend={backend.name}")
    print(f"cases={len(results)}")
    print(f"accuracy={accuracy:.4f}")
    print(f"output={args.output}")


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


if __name__ == "__main__":
    main()
