#!/usr/bin/env python3
"""Evaluate the Ridgeline extraction pipeline for rfq_01, rfq_02 and rfq_03.

Reads the answer key (ground truth) and the pipeline outputs, then reports:

  * extraction_accuracy  - field-level correctness (quantity, unit, part number)
  * match_precision      - TP / (TP + FP) over auto-approved lines
  * match_recall         - TP / (TP + FN) over lines the key expects a match for
  * additional_metrics   - flag accuracy, no-match agreement, cross-reference
                           hit rate, and lists of missed / mismatched parts

Results are printed to stdout and written to evaluation/results.json. The
answer key must only ever be read by this script, never by the pipeline.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSWER_KEY_DIR = ROOT / "answer_key"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "evaluation"
RESULTS_PATH = RESULTS_DIR / "results.json"

RFQ_IDS = ("rfq_01_email", "rfq_02_clearwater", "rfq_03_tricounty")

# Operating confidence cutoff the pipeline uses to auto-approve a line.
THRESHOLD = 0.80
# Thresholds shown for comparison; THRESHOLD is the selected operating point.
SWEEP_THRESHOLDS = (0.70, 0.80, 0.90)

# Fields checked for extraction accuracy. The output field
# `ridgeline_part_number` maps to the answer key's `expected_part_number`.
EXTRACTION_FIELDS = ("quantity", "unit", "ridgeline_part_number")


def load_json(path):
    """Load a JSON file that must contain a top-level object."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")
    return data


def values_equal(a, b):
    """Field equality where None == None and numbers compare numerically."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def evaluate_line(rfq_id, key_line, out_line):
    """Compare one answer-key line with its pipeline output (or None)."""
    expected = key_line.get("expected_part_number")
    predicted = out_line.get("ridgeline_part_number") if out_line else None
    needs_review = out_line.get("needs_review") if out_line else None
    confidence = out_line.get("confidence") if out_line else None

    # Primary approval definition: the pipeline auto-approved the line.
    approved = needs_review is False and predicted is not None

    tp = approved and expected is not None and predicted == expected
    fp = approved and not (expected is not None and predicted == expected)
    fn = expected is not None and not tp
    tn = expected is None and predicted is None

    field_correct = 0
    for field in EXTRACTION_FIELDS:
        predicted_value = out_line.get(field) if out_line else None
        expected_value = (
            key_line.get("expected_part_number")
            if field == "ridgeline_part_number"
            else key_line.get(field)
        )
        if values_equal(predicted_value, expected_value):
            field_correct += 1

    key_flags = set(key_line.get("flags") or [])
    predicted_flags = set(out_line.get("flags") or []) if out_line else set()
    flag_match = key_flags == predicted_flags

    both_null = expected is None and predicted is None
    is_cross_reference = key_line.get("match_method") == "cross_reference"
    cross_reference_hit = (
        is_cross_reference and expected is not None and predicted == expected
    )

    # "New flag" describing what went wrong on this line, if anything.
    mismatch_type = None
    if expected is None:
        if approved:
            mismatch_type = "approved_no_match"
    else:
        if predicted is None:
            mismatch_type = "missing_part"
        elif predicted != expected:
            mismatch_type = "wrong_part_approved" if approved else "wrong_part_flagged"
        elif needs_review is not False:
            mismatch_type = "correct_part_flagged_review"

    return {
        "rfq_id": rfq_id,
        "line": key_line.get("line"),
        "as_written": key_line.get("as_written"),
        "expected": expected,
        "predicted": predicted,
        "needs_review": needs_review,
        "confidence": confidence,
        "approved": approved,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "field_correct": field_correct,
        "flag_match": flag_match,
        "both_null": both_null,
        "is_cross_reference": is_cross_reference,
        "cross_reference_hit": cross_reference_hit,
        "mismatch_type": mismatch_type,
    }


def align_lines(key_doc, out_doc, warnings):
    """Join answer-key lines with output lines by 1-based `line` number."""
    key_lines = key_doc.get("lines")
    out_lines = out_doc.get("lines")
    if not isinstance(key_lines, list):
        key_lines = []
    if not isinstance(out_lines, list):
        out_lines = []

    out_by_line = {}
    for item in out_lines:
        if isinstance(item, dict) and "line" in item:
            out_by_line[item["line"]] = item

    if len(out_by_line) != len(key_lines):
        warnings.append(
            f"{key_doc.get('rfq_id')}: line count differs "
            f"(answer key {len(key_lines)}, output {len(out_by_line)})"
        )

    key_line_numbers = {line.get("line") for line in key_lines if isinstance(line, dict)}
    for number in sorted(set(out_by_line) - key_line_numbers):
        warnings.append(
            f"{key_doc.get('rfq_id')}: output has extra line {number} "
            f"not present in the answer key"
        )

    results = []
    for key_line in key_lines:
        if not isinstance(key_line, dict):
            continue
        number = key_line.get("line")
        out_line = out_by_line.get(number)
        if out_line is None:
            warnings.append(
                f"{key_doc.get('rfq_id')}: line {number} missing from output "
                f"(treated as not extracted)"
            )
        results.append(evaluate_line(key_doc.get("rfq_id"), key_line, out_line))
    return results


def aggregate(lines):
    """Combine per-line results into the final metrics."""
    total_lines = len(lines)
    total_fields = total_lines * len(EXTRACTION_FIELDS)
    correct_fields = sum(r["field_correct"] for r in lines)

    tp = sum(r["tp"] for r in lines)
    fp = sum(r["fp"] for r in lines)
    fn = sum(r["fn"] for r in lines)
    tn = sum(r["tn"] for r in lines)

    extraction_accuracy = correct_fields / total_fields if total_fields else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    flag_accuracy = sum(r["flag_match"] for r in lines) / total_lines if total_lines else 0.0
    no_match_agreement = (
        sum(r["both_null"] for r in lines) / total_lines if total_lines else 0.0
    )

    cross_reference_total = sum(r["is_cross_reference"] for r in lines)
    cross_reference_hits = sum(r["cross_reference_hit"] for r in lines)
    cross_reference_hit_rate = (
        cross_reference_hits / cross_reference_total if cross_reference_total else None
    )

    return {
        "total_lines": total_lines,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "extraction_accuracy": extraction_accuracy,
        "match_precision": precision,
        "match_recall": recall,
        "flag_accuracy": flag_accuracy,
        "no_match_agreement": no_match_agreement,
        "cross_reference_hit_rate": cross_reference_hit_rate,
        "cross_reference_total": cross_reference_total,
    }


def sweep_precision_recall(lines):
    """Precision/recall when auto-approval is `confidence >= threshold`."""
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        tp = fp = fn = 0
        for r in lines:
            confidence = r["confidence"]
            approved = (
                confidence is not None
                and confidence >= threshold
                and r["predicted"] is not None
            )
            correct = r["expected"] is not None and r["predicted"] == r["expected"]
            if approved:
                if correct:
                    tp += 1
                else:
                    fp += 1
            if r["expected"] is not None and not (approved and correct):
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def collect_discrepancies(lines):
    """Lines where the pipeline missed a part or matched it wrongly."""
    out = []
    for r in lines:
        if r["mismatch_type"] is None:
            continue
        out.append(
            {
                "rfq_id": r["rfq_id"],
                "line": r["line"],
                "as_written": r["as_written"],
                "expected_part_number": r["expected"],
                "ridgeline_part_number": r["predicted"],
                "needs_review": r["needs_review"],
                "mismatch_type": r["mismatch_type"],
            }
        )
    return out


def dedupe(items):
    """Keep first occurrences, preserving order."""
    seen = set()
    out = []
    for item in items:
        key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def build_results(lines, metrics):
    """Assemble the evaluation/results.json payload."""
    discrepancies = collect_discrepancies(lines)

    missed_parts = dedupe(
        [
            r["expected"]
            for r in lines
            if r["expected"] is not None
            and not (r["approved"] and r["predicted"] == r["expected"])
        ]
    )

    mismatched_parts = dedupe(
        [
            f"{r['expected']} -> {r['predicted']}"
            for r in lines
            if r["mismatch_type"]
            in ("wrong_part_approved", "wrong_part_flagged", "approved_no_match")
        ]
    )

    additional_metrics = {
        "flag_accuracy": round(metrics["flag_accuracy"], 4),
        "no_match_agreement": round(metrics["no_match_agreement"], 4),
        "cross_reference_hit_rate": (
            round(metrics["cross_reference_hit_rate"], 4)
            if metrics["cross_reference_hit_rate"] is not None
            else None
        ),
        "missed_parts": missed_parts,
        "mismatched_parts": mismatched_parts,
        "discrepancies": discrepancies,
    }

    return {
        "threshold": THRESHOLD,
        "extraction_accuracy": round(metrics["extraction_accuracy"], 4),
        "match_precision": round(metrics["match_precision"], 4),
        "match_recall": round(metrics["match_recall"], 4),
        "additional_metrics": additional_metrics,
    }


def print_report(lines, metrics, sweep_rows, warnings):
    """Print the human-readable evaluation report."""
    bar = "=" * 72
    print(bar)
    print("RIS Evaluation Report (rfq_01, rfq_02, rfq_03)")
    print(bar)

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    by_rfq = {}
    for r in lines:
        by_rfq.setdefault(r["rfq_id"], []).append(r)

    print("\nLine counts:")
    for rfq_id in RFQ_IDS:
        print(f"  {rfq_id}: {len(by_rfq.get(rfq_id, []))} answer-key lines")
    print(f"  TOTAL: {metrics['total_lines']} lines")

    print("\nConfusion matrix (combined):")
    print(f"  TP (auto-approved, correct part)        : {metrics['tp']}")
    print(f"  FP (auto-approved, key disagrees)       : {metrics['fp']}  <-- big flag")
    print(f"  FN (key expects a part, not auto-approved correctly): {metrics['fn']}")
    print(f"  TN (both key and pipeline left null)    : {metrics['tn']}")

    print("\nMetrics:")
    print(f"  extraction_accuracy = {metrics['extraction_accuracy']:.4f}")
    print(f"  match_precision     = {metrics['match_precision']:.4f}")
    print(f"  match_recall        = {metrics['match_recall']:.4f}")
    print(f"  flag_accuracy       = {metrics['flag_accuracy']:.4f}")
    print(f"  no_match_agreement  = {metrics['no_match_agreement']:.4f}")
    rate = metrics["cross_reference_hit_rate"]
    print(
        f"  cross_reference_hit_rate = {rate:.4f}"
        if rate is not None
        else "  cross_reference_hit_rate = n/a (no cross-reference lines)"
    )

    print("\nThreshold sweep (confidence >= threshold, auto-approval):")
    print(f"  {'threshold':>9} | {'precision':>9} | {'recall':>9} | {'TP':>3} | {'FP':>3} | {'FN':>3}")
    for row in sweep_rows:
        marker = "  <-- selected" if row["threshold"] == THRESHOLD else ""
        print(
            f"  {row['threshold']:>9.2f} | {row['precision']:>9.4f} | "
            f"{row['recall']:>9.4f} | {row['tp']:>3} | {row['fp']:>3} | {row['fn']:>3}{marker}"
        )

    discrepancies = collect_discrepancies(lines)
    print(f"\nMissed / mismatched parts ({len(discrepancies)} line(s)):")
    if not discrepancies:
        print("  (none)")
    for d in discrepancies:
        print(
            f"  {d['rfq_id']} line {d['line']}: [{d['mismatch_type']}] "
            f"expected={d['expected_part_number']} "
            f"predicted={d['ridgeline_part_number']} "
            f"needs_review={d['needs_review']} "
            f"as_written={d['as_written']!r}"
        )

    print(bar)


def main():
    """Load inputs, evaluate, print the report, and write results.json."""
    key_docs = {}
    out_docs = {}
    missing_keys = []
    missing_outputs = []

    for rfq_id in RFQ_IDS:
        key_path = ANSWER_KEY_DIR / f"{rfq_id}.json"
        out_path = OUTPUTS_DIR / f"{rfq_id}.json"
        if key_path.exists():
            key_docs[rfq_id] = load_json(key_path)
        else:
            missing_keys.append(rfq_id)
        if out_path.exists():
            out_docs[rfq_id] = load_json(out_path)
        else:
            missing_outputs.append(rfq_id)

    if missing_keys:
        names = ", ".join(f"{rfq_id}.json" for rfq_id in missing_keys)
        print(f"ERROR: missing answer key file(s): {names}", file=sys.stderr)
        return 1
    if missing_outputs:
        names = ", ".join(f"{rfq_id}.json" for rfq_id in missing_outputs)
        print(
            f"ERROR: missing pipeline output file(s) in {OUTPUTS_DIR}: {names}",
            file=sys.stderr,
        )
        return 1

    warnings = []
    all_lines = []
    for rfq_id in RFQ_IDS:
        key_doc = key_docs[rfq_id]
        out_doc = out_docs[rfq_id]
        if key_doc.get("rfq_id") != rfq_id:
            warnings.append(
                f"{rfq_id}: answer key rfq_id is {key_doc.get('rfq_id')!r}"
            )
        if out_doc.get("rfq_id") != rfq_id:
            warnings.append(f"{rfq_id}: output rfq_id is {out_doc.get('rfq_id')!r}")
        all_lines.extend(align_lines(key_doc, out_doc, warnings))

    metrics = aggregate(all_lines)
    sweep_rows = sweep_precision_recall(all_lines)
    results = build_results(all_lines, metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    print_report(all_lines, metrics, sweep_rows, warnings)
    print(f"\nWrote results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
