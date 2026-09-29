"""Command-line inference entry point."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from madl.config import MADLConfig
from madl.factory import build_pipeline
from madl.serialization import json_safe


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MADL three-class inference and conditional localization."
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and mask outputs.")
    parser.add_argument("--config", default="", help="Optional YAML configuration.")
    parser.add_argument("--weights", default=os.environ.get("MADL_WEIGHT_ROOT", "models"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = MADLConfig.from_yaml(args.config) if args.config else MADLConfig()
    config = replace(config, weight_root=args.weights)
    pipeline = build_pipeline(config, device=args.device)
    result = pipeline.predict(args.image)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    mask_path = None
    if result.mask is not None:
        import cv2
        import numpy as np

        mask_path = output_dir / f"{stem}_madl_mask.png"
        mask = (np.asarray(result.mask) > 0).astype(np.uint8) * 255
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"failed to save mask: {mask_path}")
    payload = json_safe(result)
    payload["image"] = str(Path(args.image))
    payload["mask_path"] = str(mask_path) if mask_path else None
    output_path = output_dir / f"{stem}_madl.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
