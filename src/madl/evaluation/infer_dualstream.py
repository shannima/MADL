"""Standalone inference for the released Agent B dual-stream checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from madl.runtime import DualStreamRuntime
from madl.serialization import json_safe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the dual-stream pixel model without other MADL agents."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)

    import cv2
    import numpy as np

    runtime = DualStreamRuntime(args.checkpoint, device=args.device)
    result = dict(runtime.predict(args.image))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    heatmap = np.asarray(result.pop("anomaly_heatmap"), dtype=np.float32)
    heatmap_npy = output_dir / f"{stem}_dualstream_heatmap.npy"
    heatmap_png = output_dir / f"{stem}_dualstream_heatmap.png"
    np.save(heatmap_npy, heatmap)
    colored = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if not cv2.imwrite(str(heatmap_png), colored):
        raise OSError(f"failed to save heatmap: {heatmap_png}")
    result.update({"heatmap_npy": str(heatmap_npy), "heatmap_png": str(heatmap_png)})
    output_json = output_dir / f"{stem}_dualstream.json"
    output_json.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
