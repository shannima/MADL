"""Five-fold linear-probe protocol for image-token representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate_features(
    features_path: str | Path, labels_path: str | Path, *, seed: int = 42
) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.load(features_path)
    labels = np.load(labels_path)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be NxD and labels must be a matching vector")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.empty_like(labels)
    fold_metrics = []
    for train_indices, test_indices in splitter.split(features, labels):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed, multi_class="auto"),
        )
        estimator.fit(features[train_indices], labels[train_indices])
        fold_prediction = estimator.predict(features[test_indices])
        predictions[test_indices] = fold_prediction
        fold_metrics.append(
            {
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels[test_indices], fold_prediction)
                ),
                "macro_f1": float(f1_score(labels[test_indices], fold_prediction, average="macro")),
            }
        )
    classes = sorted(set(labels.tolist()))
    return {
        "seed": seed,
        "folds": 5,
        "sample_count": int(len(labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=classes, normalize="true"
        ).tolist(),
        "classes": [str(value) for value in classes],
        "fold_metrics": fold_metrics,
        "out_of_fold_predictions": predictions.tolist(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed five-fold MADL semantic linear probe."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = evaluate_features(args.features, args.labels, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
