from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from datasets import load_dataset

from .hf_runner import BaseKVBackend, generate_incremental


TASK_METRICS = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "multifieldqa_en": "f1",
    "multifieldqa_zh": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "dureader": "rouge_l",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "vcsum": "rouge_l",
    "trec": "classification",
    "triviaqa": "f1",
    "samsum": "rouge_l",
    "lsht": "classification",
    "passage_count": "exact_match",
    "passage_retrieval_en": "exact_match",
    "passage_retrieval_zh": "exact_match",
    "lcc": "code_f1",
    "repobench-p": "code_f1",
}

TASK_CATEGORIES = {
    "narrativeqa": "single_doc_qa",
    "qasper": "single_doc_qa",
    "multifieldqa_en": "single_doc_qa",
    "multifieldqa_zh": "single_doc_qa",
    "hotpotqa": "multi_doc_qa",
    "2wikimqa": "multi_doc_qa",
    "musique": "multi_doc_qa",
    "dureader": "multi_doc_qa",
    "gov_report": "summarization",
    "qmsum": "summarization",
    "multi_news": "summarization",
    "vcsum": "summarization",
    "trec": "classification",
    "triviaqa": "few_shot",
    "samsum": "few_shot",
    "lsht": "classification",
    "passage_count": "synthetic",
    "passage_retrieval_en": "synthetic",
    "passage_retrieval_zh": "synthetic",
    "lcc": "code",
    "repobench-p": "code",
}


@dataclass
class LongBenchResult:
    dataset: str
    sample_id: str
    metric: str
    score: float
    category: str
    prompt_tokens: int
    generated_tokens: int
    backend_name: str
    peak_cache_bytes: int
    prediction: str


def load_longbench_records(
    subset: str,
    split: str = "test",
    max_examples: int | None = None,
    dataset_name: str = "zai-org/LongBench",
    local_path: str | None = None,
    trust_remote_code: bool = True,
) -> list[dict[str, Any]]:
    if local_path is not None:
        path = Path(local_path)
        return _load_jsonl_records(path, max_examples)

    dataset = load_dataset(dataset_name, subset, split=split, trust_remote_code=trust_remote_code)
    rows = list(dataset)
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


def evaluate_longbench_subset(
    model,
    tokenizer,
    backend: BaseKVBackend,
    subset: str,
    split: str = "test",
    max_examples: int | None = None,
    dataset_name: str = "zai-org/LongBench",
    local_path: str | None = None,
    max_new_tokens: int = 64,
) -> list[LongBenchResult]:
    records = load_longbench_records(
        subset=subset,
        split=split,
        max_examples=max_examples,
        dataset_name=dataset_name,
        local_path=local_path,
    )
    results: list[LongBenchResult] = []
    for record in records:
        prompt = build_longbench_prompt(record)
        generation = generate_incremental(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            backend=backend,
            max_new_tokens=max_new_tokens,
            stop_strings=("\n\n",),
        )
        prediction = generation.generated_text.strip()
        task_name = _canonical_task_name(subset)
        metric = TASK_METRICS.get(task_name, "f1")
        score = score_prediction(
            prediction=prediction,
            answers=record.get("answers") or [],
            metric=metric,
            all_classes=record.get("all_classes"),
        )
        results.append(
            LongBenchResult(
                dataset=subset,
                sample_id=str(record.get("_id") or record.get("id") or len(results)),
                metric=metric,
                score=score,
                category=TASK_CATEGORIES.get(task_name, "other"),
                prompt_tokens=generation.prompt_tokens,
                generated_tokens=len(generation.generated_tokens),
                backend_name=generation.backend_name,
                peak_cache_bytes=max(generation.cache_bytes_per_step, default=0),
                prediction=prediction,
            )
        )
    return results


def build_longbench_prompt(record: dict[str, Any]) -> str:
    if record.get("prompt"):
        return str(record["prompt"])

    context = str(record.get("context") or "")
    question = str(record.get("input") or "")
    if context and question:
        return (
            "Read the following context carefully and answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
    if question:
        return f"{question}\n\nAnswer:"
    raise ValueError("record does not contain enough fields to build a prompt")


def aggregate_longbench_results(results: Sequence[LongBenchResult]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for result in results:
        totals.setdefault(result.category, []).append(result.score)
        totals.setdefault(result.dataset, []).append(result.score)
    summary = {key: sum(values) / len(values) for key, values in totals.items() if values}
    if results:
        summary["overall"] = sum(result.score for result in results) / len(results)
    return summary


def write_longbench_results_csv(path: str | Path, results: Sequence[LongBenchResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "sample_id",
                "metric",
                "score",
                "category",
                "prompt_tokens",
                "generated_tokens",
                "backend_name",
                "peak_cache_bytes",
                "prediction",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.dataset,
                    result.sample_id,
                    result.metric,
                    result.score,
                    result.category,
                    result.prompt_tokens,
                    result.generated_tokens,
                    result.backend_name,
                    result.peak_cache_bytes,
                    result.prediction,
                ]
            )


def score_prediction(
    prediction: str,
    answers: Sequence[str],
    metric: str,
    all_classes: Sequence[str] | None = None,
) -> float:
    if not answers:
        return 0.0
    if metric == "exact_match":
        return max(_exact_match(prediction, answer) for answer in answers)
    if metric == "classification":
        return _classification_score(prediction, answers, all_classes)
    if metric == "rouge_l":
        return max(_rouge_l_f1(prediction, answer) for answer in answers)
    if metric == "code_f1":
        return max(_token_f1(prediction, answer, strip_punctuation=False) for answer in answers)
    return max(_token_f1(prediction, answer) for answer in answers)


def _load_jsonl_records(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def _canonical_task_name(subset: str) -> str:
    return subset[:-2] if subset.endswith("_e") else subset


def _normalize_text(text: str, strip_punctuation: bool = True) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    if strip_punctuation:
        lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered


def _exact_match(prediction: str, answer: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(answer) else 0.0


def _tokenize(text: str, strip_punctuation: bool = True) -> list[str]:
    normalized = _normalize_text(text, strip_punctuation=strip_punctuation)
    return [token for token in normalized.split(" ") if token]


def _token_f1(prediction: str, answer: str, strip_punctuation: bool = True) -> float:
    pred_tokens = _tokenize(prediction, strip_punctuation=strip_punctuation)
    answer_tokens = _tokenize(answer, strip_punctuation=strip_punctuation)
    if not pred_tokens or not answer_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    answer_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in answer_tokens:
        answer_counts[token] = answer_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(token, 0), answer_counts.get(token, 0)) for token in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _rouge_l_f1(prediction: str, answer: str) -> float:
    pred_tokens = _tokenize(prediction, strip_punctuation=False)
    answer_tokens = _tokenize(answer, strip_punctuation=False)
    if not pred_tokens or not answer_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, answer_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(answer_tokens)
    if precision == 0 or recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _classification_score(prediction: str, answers: Sequence[str], all_classes: Sequence[str] | None) -> float:
    normalized_prediction = _normalize_text(prediction)
    candidates = all_classes or answers
    chosen = None
    for candidate in candidates:
        normalized = _normalize_text(str(candidate))
        if normalized and normalized in normalized_prediction:
            chosen = normalized
            break
    if chosen is None:
        chosen = normalized_prediction
    answer_set = {_normalize_text(answer) for answer in answers}
    return 1.0 if chosen in answer_set else 0.0
