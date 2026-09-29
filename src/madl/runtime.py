"""Model-backed runtime adapters used by the public inference factory."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

SFT_SYSTEM_PROMPT = (
    "You are an image-forensics proposal-grounding agent. Return strict JSON with keys "
    "label, is_tampered, is_synthetic, explanation, and evidence_regions. The label must be "
    "one of real, synthetic, or tampered. Local tampering requires localized evidence_regions "
    "whose box_2d values use [x_min, y_min, x_max, y_max] in 0-1000 coordinates."
)
SFT_USER_PROMPT = (
    "Classify the image and provide spatially testable forensic evidence as strict JSON."
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QwenForensicsRuntime:
    """Lazy wrapper around the SFT-adapted Qwen2.5-VL Agent A model."""

    def __init__(self, adapter_path: str | Path, *, small_object_second_pass: bool = True) -> None:
        from madl.backbones.qwen import VisualDetectiveAgent

        self.adapter_path = Path(adapter_path)
        if not self.adapter_path.is_dir():
            raise FileNotFoundError(self.adapter_path)
        self.model = VisualDetectiveAgent(
            model_path=str(self.adapter_path),
            small_object_second_pass=small_object_second_pass,
        )

    def analyze(self, image: Any) -> Mapping[str, Any]:
        return self.model.analyze_image(
            str(image),
            user_prompt=SFT_USER_PROMPT,
            system_instruction=SFT_SYSTEM_PROMPT,
        )


class DualStreamRuntime:
    """Load the sanitized Agent B dual-stream checkpoint and emit structured evidence."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | None = None) -> None:
        import torch

        from madl.backbones.dualstream.model import DualStreamPixelAgent

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != "madl.checkpoint.v1":
            raise ValueError("unsupported dual-stream checkpoint format")
        architecture = dict(checkpoint["architecture"])
        self.image_size = int(architecture["image_size"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualStreamPixelAgent(
            backbone_name=str(architecture["backbone_name"]),
            image_size=self.image_size,
            fusion_dim=int(architecture.get("fusion_dim", 256)),
            num_classes=int(architecture.get("num_classes", 3)),
            use_pretrained=False,
            proposal_threshold=float(architecture.get("proposal_threshold", 0.45)),
            proposal_top_k=int(architecture.get("proposal_top_k", 5)),
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()

    def predict(self, image: str | Path) -> Mapping[str, Any]:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as transform

        from madl.backbones.dualstream.dataset import compute_noise_map

        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            original_size = source.size
            resized = transform.resize(
                source.convert("RGB"),
                [self.image_size, self.image_size],
                antialias=True,
            )
        image_array = np.asarray(resized, dtype=np.uint8)
        rgb = transform.to_tensor(resized).unsqueeze(0).to(self.device)
        noise = torch.from_numpy(compute_noise_map(image_array)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = dict(self.model.predict_structured(rgb, noise)[0])
        heatmap = result["anomaly_heatmap"]
        if heatmap.shape[::-1] != original_size:
            heatmap = cv2.resize(heatmap, original_size, interpolation=cv2.INTER_LINEAR)
        result["anomaly_heatmap"] = heatmap.astype(np.float32)
        return result


class SAMCandidateSegmenter:
    """Convert Agent A/B proposal boxes into a common pool of SAM mask candidates."""

    def __init__(self, checkpoint_path: str | Path, *, device: str | None = None) -> None:
        import torch

        from madl.backbones.sam import PrecisionTrackerAgent

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PrecisionTrackerAgent(
            sam_checkpoint=str(self.checkpoint_path),
            device=runtime_device,
            safe_mode=True,
            safe_strategy="multimask_refine_smallboost",
        )

    def segment(
        self,
        image: str | Path,
        regions: Sequence[Mapping[str, Any]],
        heatmap: Any | None = None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            box = region.get("box_2d")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            for candidate in self.model.extract_mask_candidates(
                image_path=str(image),
                box_2d=list(box),
                max_candidates=10,
            ):
                item = dict(candidate)
                item["proposal_source"] = str(region.get("source", f"proposal:{region_index}"))
                candidates.append(item)
        return candidates


class DualStreamCandidateBuilder:
    """Generate the full heatmap-guided proposal pool used by Agent B."""

    def __init__(self, *, max_regions: int = 40) -> None:
        self.max_regions = int(max_regions)

    def build(
        self,
        image: str | Path,
        heatmap: Any,
        *,
        spatial_priors: Sequence[Sequence[float]] = (),
        pixel_report: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from madl.localization.candidate_pool import build_regions_from_model_heatmap

        evidence_regions = []
        for box in spatial_priors:
            values = [float(value) for value in box]
            if len(values) != 4:
                continue
            scale = 1000.0 if max(abs(value) for value in values) > 1.0 else 1.0
            permille = [int(round(min(1.0, max(0.0, value / scale)) * 1000.0)) for value in values]
            evidence_regions.append({"box_2d": permille, "confidence": 0.75})
        semantic_report = {"evidence_regions": evidence_regions} if evidence_regions else None
        regions, _ = build_regions_from_model_heatmap(
            str(image),
            heatmap,
            max_regions=self.max_regions,
            semantic_report=semantic_report,
        )
        return [{"box_2d": list(region.box_2d), "source": region.source} for region in regions]


class VisualRankerAdapter:
    """Adapt in-memory SAM masks to the verified visual candidate-ranker runtime."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_sha256: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        import torch

        from madl.localization.visual_ranker import load_visual_candidate_ranker

        self.checkpoint_path = Path(checkpoint_path).resolve()
        digest = expected_sha256 or sha256_file(self.checkpoint_path)
        runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ranker = load_visual_candidate_ranker(
            self.checkpoint_path,
            expected_sha256=digest.lower(),
            device=runtime_device,
            batch_size=batch_size,
        )

    def select(self, image: str | Path, heatmap: Any, candidates: Sequence[Mapping[str, Any]]):
        import cv2
        import numpy as np

        image_bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"image is unreadable: {image}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        heatmap_array = np.asarray(heatmap, dtype=np.float32)
        if heatmap_array.shape != image_rgb.shape[:2]:
            heatmap_array = cv2.resize(
                heatmap_array,
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        with tempfile.TemporaryDirectory(prefix="madl_masks_") as directory:
            rows = []
            for index, candidate in enumerate(candidates):
                mask = np.asarray(candidate.get("mask"), dtype=np.uint8)
                if mask.shape != image_rgb.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (image_rgb.shape[1], image_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_path = Path(directory) / f"candidate_{index:04d}.png"
                if not cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255):
                    raise OSError(f"failed to write temporary candidate mask: {mask_path}")
                row = dict(candidate)
                row["raw_mask"] = str(mask_path)
                row.setdefault("proposal_source", f"candidate:{index}")
                rows.append(row)
            selected = self.ranker.select(image_rgb, heatmap_array, rows)
            if selected is None:
                return None
            position = int(selected.get("selection", {}).get("selected_position", 0))
            result = dict(candidates[position])
            result["visual_ranker_score"] = float(selected["visual_ranker_score"])
            result["selection"] = dict(selected["selection"])
            return result
