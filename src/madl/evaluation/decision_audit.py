"""State-wise paired audit for predictions before and after Agent C."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from madl.evaluation.statistics import exact_mcnemar_p

STATES = ("agreement", "suppression", "override", "conflict")


def audit(rows: list[dict]) -> dict:
    output = {}
    corrected_total = 0
    worsened_total = 0
    for state in (*STATES, "overall"):
        selected = rows if state == "overall" else [row for row in rows if row["state"] == state]
        if not selected:
            continue
        before_correct = [row["before"] == row["target"] for row in selected]
        after_correct = [row["after"] == row["target"] for row in selected]
        corrected = sum(
            not before and after for before, after in zip(before_correct, after_correct)
        )
        worsened = sum(before and not after for before, after in zip(before_correct, after_correct))
        if state == "overall":
            corrected_total, worsened_total = corrected, worsened
        output[state] = {
            "count": len(selected),
            "accuracy_before": sum(before_correct) / len(selected),
            "accuracy_after": sum(after_correct) / len(selected),
            "corrected_rate": corrected / len(selected),
            "worsened_rate": worsened / len(selected),
        }
    output["mcnemar"] = {
        "corrected": corrected_total,
        "worsened": worsened_total,
        "exact_two_sided_p": exact_mcnemar_p(corrected_total, worsened_total),
    }
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Agent C decisions from a frozen CSV evidence table."
    )
    parser.add_argument(
        "--input", required=True, help="CSV columns: sample_id,target,before,after,state."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    report = audit(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
