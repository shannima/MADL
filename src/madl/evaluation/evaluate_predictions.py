"""Evaluate frozen MADL predictions from a portable CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from madl.evaluation.metrics import classification_metrics, localization_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate MADL JSON/mask predictions.")
    parser.add_argument("--manifest", required=True, help="CSV: sample_id,image,label,mask.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _read_binary_mask(path: Path):
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"mask is unreadable: {path}")
    return mask > 127


def evaluate(manifest: str | Path, prediction_dir: str | Path) -> dict:
    prediction_root = Path(prediction_dir)
    targets = []
    predictions = []
    localization_rows = []
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = str(row["sample_id"]).strip()
            target = str(row["label"]).strip().lower()
            prediction_path = prediction_root / f"{sample_id}_madl.json"
            with prediction_path.open("r", encoding="utf-8") as prediction_stream:
                payload = json.load(prediction_stream)
            predicted = str(payload["label"]).strip().lower()
            targets.append(target)
            predictions.append(predicted)
            if target == "tampered" and row.get("mask"):
                predicted_mask = payload.get("mask_path")
                if not predicted_mask:
                    localization_rows.append({"sample_id": sample_id, "iou": 0.0, "f1": 0.0})
                    continue
                metrics = localization_metrics(
                    _read_binary_mask(Path(predicted_mask)),
                    _read_binary_mask(Path(row["mask"])),
                )
                localization_rows.append({"sample_id": sample_id, **metrics})
    classification = classification_metrics(targets, predictions)
    localization = {
        "count": len(localization_rows),
        "mean_iou": sum(item["iou"] for item in localization_rows) / len(localization_rows)
        if localization_rows
        else 0.0,
        "mean_f1": sum(item["f1"] for item in localization_rows) / len(localization_rows)
        if localization_rows
        else 0.0,
        "samples": localization_rows,
    }
    return {"classification": classification, "localization": localization}


def main(argv=None) -> int:
    args = parse_args(argv)
    report = evaluate(args.manifest, args.prediction_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
