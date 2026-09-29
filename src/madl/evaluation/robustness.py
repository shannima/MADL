"""Multithreaded construction of the fixed robustness perturbation suite."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CONDITIONS = (
    "jpeg50",
    "jpeg60",
    "jpeg70",
    "jpeg80",
    "resize050",
    "resize075",
    "resize150",
    "gaussian05",
    "gaussian10",
    "webp80",
    "blur3x3",
)


def _write_image(path: Path, image, parameters=None) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, parameters or []):
        raise OSError(f"failed to write perturbed image: {path}")


def perturb(image, condition: str, seed: int):
    import cv2
    import numpy as np

    if condition.startswith("jpeg") or condition.startswith("webp"):
        return image
    if condition.startswith("resize"):
        scale = {"resize050": 0.5, "resize075": 0.75, "resize150": 1.5}[condition]
        return cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
        )
    if condition.startswith("gaussian"):
        sigma = 5.0 if condition == "gaussian05" else 10.0
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if condition == "blur3x3":
        return cv2.GaussianBlur(image, (3, 3), 0)
    raise ValueError(f"unknown robustness condition: {condition}")


def _prepare_one(row: dict, condition: str, output_root: Path, seed: int) -> dict:
    import cv2

    sample_id = row["sample_id"]
    image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"image is unreadable: {row['image']}")
    transformed = perturb(image, condition, seed)
    suffix = (
        ".webp" if condition == "webp80" else ".jpg" if condition.startswith("jpeg") else ".png"
    )
    output_image = output_root / condition / "images" / f"{sample_id}{suffix}"
    parameters = []
    if condition.startswith("jpeg"):
        parameters = [cv2.IMWRITE_JPEG_QUALITY, int(condition[-2:])]
    elif condition == "webp80":
        parameters = [cv2.IMWRITE_WEBP_QUALITY, 80]
    _write_image(output_image, transformed, parameters)

    output_mask = ""
    if row.get("mask"):
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"mask is unreadable: {row['mask']}")
        if transformed.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(
                mask, (transformed.shape[1], transformed.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        mask_path = output_root / condition / "masks" / f"{sample_id}_mask.png"
        _write_image(mask_path, mask)
        output_mask = str(mask_path)
    return {
        "sample_id": sample_id,
        "image": str(output_image),
        "label": row["label"],
        "mask": output_mask,
        "condition": condition,
    }


def prepare_suite(
    manifest: str | Path, output_root: str | Path, *, workers: int = 8, seed: int = 20260708
):
    output_root = Path(output_root)
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for condition_index, condition in enumerate(CONDITIONS):
            for row_index, row in enumerate(rows):
                sample_seed = seed + condition_index * 1_000_003 + row_index
                futures.append(pool.submit(_prepare_one, row, condition, output_root, sample_seed))
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["condition"], row["sample_id"]))
    for condition in CONDITIONS:
        manifest_path = output_root / condition / "manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("sample_id", "image", "label", "mask", "condition")
            )
            writer.writeheader()
            writer.writerows(row for row in results if row["condition"] == condition)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare MADL robustness inputs with multithreaded I/O."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260708)
    args = parser.parse_args(argv)
    rows = prepare_suite(args.manifest, args.output_root, workers=args.workers, seed=args.seed)
    print(f"Prepared {len(rows)} perturbed inputs across {len(CONDITIONS)} conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
