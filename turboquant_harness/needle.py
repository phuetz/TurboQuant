from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

from .hf_runner import BaseKVBackend, generate_incremental


DEFAULT_FILLER_TEXT = (
    "This is supporting context for a long document. It contains factual-looking but irrelevant details "
    "about places, projects, timelines, and meeting notes. "
)


@dataclass
class NeedleResult:
    context_tokens: int
    depth_percent: float
    expected_answer: str
    generated_answer: str
    correct: bool
    prompt_tokens: int
    generated_tokens: int
    backend_name: str
    peak_cache_bytes: int


def build_needle_prompt(
    tokenizer,
    context_tokens: int,
    depth_percent: float,
    needle_key: str,
    needle_value: str,
    filler_text: str = DEFAULT_FILLER_TEXT,
) -> str:
    if not (0.0 <= depth_percent <= 100.0):
        raise ValueError("depth_percent must be between 0 and 100")
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")

    needle_sentence = f"The {needle_key} is {needle_value}."
    filler_ids = tokenizer(filler_text, add_special_tokens=False).input_ids
    if not filler_ids:
        raise ValueError("filler_text must tokenize to at least one token")
    needle_ids = tokenizer(needle_sentence, add_special_tokens=False).input_ids

    filler_budget = max(context_tokens - len(needle_ids), len(filler_ids))
    repeated_ids: list[int] = []
    while len(repeated_ids) < filler_budget:
        repeated_ids.extend(filler_ids)
    repeated_ids = repeated_ids[:filler_budget]

    insert_at = int(round(len(repeated_ids) * depth_percent / 100.0))
    context_ids = repeated_ids[:insert_at] + needle_ids + repeated_ids[insert_at:]
    context_ids = context_ids[: max(context_tokens, len(needle_ids))]
    context_text = tokenizer.decode(context_ids, skip_special_tokens=True)

    return (
        "You are given a long context. Find the requested value and answer with only the value.\n\n"
        f"{context_text}\n\n"
        f"Question: What is the {needle_key}?\n"
        "Answer:"
    )


def run_needle_suite(
    model,
    tokenizer,
    backend: BaseKVBackend,
    context_lengths: Iterable[int],
    depths: Iterable[float],
    needle_key: str,
    needle_value: str,
    max_new_tokens: int = 16,
    filler_text: str = DEFAULT_FILLER_TEXT,
) -> list[NeedleResult]:
    results: list[NeedleResult] = []
    for context_tokens in context_lengths:
        for depth_percent in depths:
            prompt = build_needle_prompt(
                tokenizer=tokenizer,
                context_tokens=context_tokens,
                depth_percent=depth_percent,
                needle_key=needle_key,
                needle_value=needle_value,
                filler_text=filler_text,
            )
            generation = generate_incremental(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                backend=backend,
                max_new_tokens=max_new_tokens,
                stop_strings=("\n",),
            )
            generated = _extract_first_line(generation.generated_text)
            results.append(
                NeedleResult(
                    context_tokens=context_tokens,
                    depth_percent=depth_percent,
                    expected_answer=needle_value,
                    generated_answer=generated,
                    correct=_normalize_answer(generated) == _normalize_answer(needle_value),
                    prompt_tokens=generation.prompt_tokens,
                    generated_tokens=len(generation.generated_tokens),
                    backend_name=generation.backend_name,
                    peak_cache_bytes=max(generation.cache_bytes_per_step, default=0),
                )
            )
    return results


def write_needle_results_csv(path: str | Path, results: Sequence[NeedleResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "context_tokens",
                "depth_percent",
                "expected_answer",
                "generated_answer",
                "correct",
                "prompt_tokens",
                "generated_tokens",
                "backend_name",
                "peak_cache_bytes",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.context_tokens,
                    result.depth_percent,
                    result.expected_answer,
                    result.generated_answer,
                    int(result.correct),
                    result.prompt_tokens,
                    result.generated_tokens,
                    result.backend_name,
                    result.peak_cache_bytes,
                ]
            )


def _extract_first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _normalize_answer(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^a-z0-9 ]+", "", lowered)
