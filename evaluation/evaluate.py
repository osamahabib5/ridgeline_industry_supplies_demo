#!/usr/bin/env python3
"""Ridgeline Quote Desk evaluation.

Compares ``outputs/*.json`` against the ground truth in ``answer_key/`` (only
the three RFQs that have an answer key) and writes ``evaluation/results.json``.

The pipeline itself never reads ``answer_key/``; only this script does.

Metrics
-------
threshold             : the confidence value below which a line needs review.
extraction_accuracy   : fraction of answer-key lines whose quantity AND unit
                        were extracted correctly.
match_precision       : TP / (TP + FP) on ridgeline part numbers.
match_recall          : TP / (TP + FN) on ridgeline part numbers.

TP: predicted part == expected part (expected is not null).
FP: predicted a part where the answer key has none, or predicted the wrong part.
FN: expected a part but predicted none, or predicted the wrong part.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
ANSWER_KEY_DIR = ROOT / "answer_key"
RESULTS_PATH = ROOT / "evaluation" / "results.json"

NEEDS_REVIEW_THRESHOLD = 0.80

# Only these RFQs have an answer key and are scored.
SCORED_RFQS = ("rfq_01_email", "rfq_02_clearwater", "rfq_03_tricounty")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def align_lines(expected_lines, actual_lines):
    """Return a list of (expected, actual) pairs aligned by ``line`` number."""
    actual_by_line = {item.get("line"): item for item in actual_lines if isinstance(item, dict)}
    pairs = []
    for expected in expected_lines:
        line_number = expected.get("line")
        actual = actual_by_line.get(line_number)
        pairs.append((expected, actual))
    return pairs


def values_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    return a is not None and b is not None and a == b


def main() -> int:
    total_lines = 0
    extraction_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    lines_needing_review = 0

    for stem in SCORED_RFQS:
        answer = load_json(ANSWER_KEY_DIR / f"{stem}.json")
        try:
            actual = load_json(OUTPUTS_DIR / f"{stem}.json")
        except (FileNotFoundError, json.JSONDecodeError):
            actual_lines: list = []
        else:
            actual_lines = actual.get("lines", [])

        for line in actual_lines:
            if line.get("needs_review"):
                lines_needing_review += 1

        for expected, actual in align_lines(answer.get("lines", []), actual_lines):
            total_lines += 1
            if actual is None:
                actual = {}
            if values_equal(expected.get("quantity"), actual.get("quantity")) and values_equal(
                expected.get("unit"), actual.get("unit")
            ):
                extraction_correct += 1

            expected_part = expected.get("expected_part_number")
            predicted_part = actual.get("ridgeline_part_number")
            if predicted_part is not None and predicted_part == expected_part and expected_part is not None:
                true_positive += 1
            else:
                if predicted_part is not None:
                    false_positive += 1
                if expected_part is not None:
                    false_negative += 1

    extraction_accuracy = (extraction_correct / total_lines) if total_lines else 0.0
    match_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 1.0
    )
    match_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 1.0
    )

    results = {
        "threshold": NEEDS_REVIEW_THRESHOLD,
        "extraction_accuracy": round(extraction_accuracy, 4),
        "match_precision": round(match_precision, 4),
        "match_recall": round(match_recall, 4),
        "additional_metrics": {
            "scored_rfqs": list(SCORED_RFQS),
            "total_scored_lines": total_lines,
            "lines_needing_review": lines_needing_review,
            "description_threshold": 0.60,
            "cross_reference_confidence": 0.90,
            "direct_part_confidence": 0.95,
            "llm_confidence": 0.75,
        },
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
