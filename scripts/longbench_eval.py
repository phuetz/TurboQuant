from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turboquant_harness.hf_runner import FullPrecisionKVBackend, TurboQuantKVBackend, load_model_and_tokenizer
from turboquant_harness.longbench import (
    aggregate_longbench_results,
    evaluate_longbench_subset,
    write_longbench_results_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LongBench subset with optional TurboQuant KV compression.")
    parser.add_argument("--model", required=True, help="Hugging Face causal LM identifier")
    parser.add_argument("--subset", required=True, help="LongBench subset, e.g. hotpotqa_e")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--backend", choices=["full", "turboquant"], default="full")
    parser.add_argument("--key-quantizer", default="prod:3")
    parser.add_argument("--value-quantizer", default="mse:3")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--dataset-name", default="zai-org/LongBench")
    parser.add_argument("--local-path", default="", help="Optional local JSONL file instead of datasets.load_dataset")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", default="artifacts/longbench_results.csv")
    parser.add_argument("--summary-output", default="artifacts/longbench_summary.json")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, device=args.device, dtype=args.dtype)
    backend = (
        FullPrecisionKVBackend()
        if args.backend == "full"
        else TurboQuantKVBackend(args.key_quantizer, args.value_quantizer, seed=args.seed)
    )

    results = evaluate_longbench_subset(
        model=model,
        tokenizer=tokenizer,
        backend=backend,
        subset=args.subset,
        split=args.split,
        max_examples=args.max_examples,
        dataset_name=args.dataset_name,
        local_path=args.local_path or None,
        max_new_tokens=args.max_new_tokens,
    )
    summary = aggregate_longbench_results(results)
    write_longbench_results_csv(args.output, results)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"backend={backend.name}")
    print(f"subset={args.subset}")
    print(f"examples={len(results)}")
    print(f"overall={summary.get('overall', 0.0):.4f}")
    print(f"output={args.output}")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
