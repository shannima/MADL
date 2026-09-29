import os
from typing import Any, Dict, List

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry

# ==========================================
# Agent B: Precision Tracker (SAM)
# ==========================================


class PrecisionTrackerAgent:
    def __init__(
        self,
        sam_checkpoint,
        model_type="vit_h",
        device="cuda",
        safe_mode=False,
        safe_strategy="ensemble",
    ):
        """
        Initialize the SAM model.
        """
        print(f"Loading SAM ({model_type}) from {sam_checkpoint}...")

        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint not found at: {sam_checkpoint}\nPlease download it first."
            )

        # Load the model
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.device = device
        self.runtime_device = device
        self.safe_mode = safe_mode
        self.safe_strategy = safe_strategy
        self._cached_image_path = None
        self._cached_image_mtime_ns = None
        self._cached_image_rgb = None
        print("SAM loaded successfully.")

    def _clear_image_cache(self):
        self._cached_image_path = None
        self._cached_image_mtime_ns = None
        self._cached_image_rgb = None

    def _load_image_and_prepare_predictor(self, image_path: str):
        image_stat = os.stat(image_path)
        cached_path = getattr(self, "_cached_image_path", None)
        cached_mtime_ns = getattr(self, "_cached_image_mtime_ns", None)
        cached_image_rgb = getattr(self, "_cached_image_rgb", None)
        if (
            cached_path == image_path
            and cached_mtime_ns == image_stat.st_mtime_ns
            and cached_image_rgb is not None
        ):
            return cached_image_rgb

        print(f"Agent B is reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image with cv2.")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.predictor.set_image(image_rgb)
        self._cached_image_path = image_path
        self._cached_image_mtime_ns = image_stat.st_mtime_ns
        self._cached_image_rgb = image_rgb
        return image_rgb

    def _mask_to_box(self, mask: np.ndarray) -> np.ndarray | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x0 = int(xs.min())
        y0 = int(ys.min())
        x1 = int(xs.max()) + 1
        y1 = int(ys.max()) + 1
        return np.array([x0, y0, x1, y1], dtype=np.int32)

    def _mask_centroid(self, mask: np.ndarray) -> np.ndarray | None:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        cx = int(round(xs.mean()))
        cy = int(round(ys.mean()))
        return np.array([[cx, cy]], dtype=np.float32)

    def _box_area_ratio(self, box: np.ndarray, width: int, height: int) -> float:
        box_area = max(1, int(box[2] - box[0]) * int(box[3] - box[1]))
        image_area = max(1, width * height)
        return float(box_area / image_area)

    def move_model_to(self, device: str):
        if self.runtime_device == device:
            return
        print(f"[Agent B] Moving SAM model to {device}...")
        self.sam.to(device=device)
        self.device = device
        self.runtime_device = device
        self._clear_image_cache()
        if device == "cpu":
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def _permille_box_to_pixels(self, box_2d: List[int], width: int, height: int) -> np.ndarray:
        x_min_permille, y_min_permille, x_max_permille, y_max_permille = box_2d
        x_min = int((x_min_permille / 1000.0) * width)
        y_min = int((y_min_permille / 1000.0) * height)
        x_max = int((x_max_permille / 1000.0) * width)
        y_max = int((y_max_permille / 1000.0) * height)
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))
        return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)

    def _expand_box(self, box: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
        x0, y0, x1, y1 = box.astype(np.float32)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        bw = max(2.0, (x1 - x0) * scale)
        bh = max(2.0, (y1 - y0) * scale)
        nx0 = int(max(0, round(cx - bw / 2.0)))
        ny0 = int(max(0, round(cy - bh / 2.0)))
        nx1 = int(min(width, round(cx + bw / 2.0)))
        ny1 = int(min(height, round(cy + bh / 2.0)))
        nx1 = max(nx0 + 1, nx1)
        ny1 = max(ny0 + 1, ny1)
        return np.array([nx0, ny0, nx1, ny1], dtype=np.int32)

    def _shift_box(
        self, box: np.ndarray, width: int, height: int, shift_x: float, shift_y: float
    ) -> np.ndarray:
        x0, y0, x1, y1 = box.astype(np.float32)
        bw = x1 - x0
        bh = y1 - y0
        nx0 = int(round(x0 + shift_x * bw))
        ny0 = int(round(y0 + shift_y * bh))
        nx1 = int(round(x1 + shift_x * bw))
        ny1 = int(round(y1 + shift_y * bh))
        nx0 = max(0, min(nx0, width - 1))
        ny0 = max(0, min(ny0, height - 1))
        nx1 = max(nx0 + 1, min(nx1, width))
        ny1 = max(ny0 + 1, min(ny1, height))
        return np.array([nx0, ny0, nx1, ny1], dtype=np.int32)

    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_uint8 = mask.astype(np.uint8) * 255
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel3)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel5)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
        if num_labels <= 1:
            return (refined > 127).astype(np.uint8)

        areas = stats[:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas[1:]) + 1)
        largest_area = int(areas[largest_label])
        min_keep_area = max(16, int(largest_area * 0.03))
        filtered = np.zeros_like(refined)
        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_keep_area:
                filtered[labels == label_idx] = 255
        return (filtered > 127).astype(np.uint8)

    def _candidate_score(self, mask: np.ndarray, sam_score: float, source_box: np.ndarray) -> float:
        mask_area = float(mask.sum())
        box_area = float(max(1, (source_box[2] - source_box[0]) * (source_box[3] - source_box[1])))
        fill_ratio = min(mask_area / box_area, 1.0)
        return float(sam_score) * 0.75 + fill_ratio * 0.25

    def extract_mask_candidates(
        self,
        image_path: str,
        box_2d: list,
        output_path: str = "sam_output.jpg",
        box_scales: List[float] | None = None,
        multimask_output: bool = True,
        max_candidates: int = 6,
        use_box_center_point: bool = False,
        respect_safe_mode: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run SAM with multi-box ensemble and multi-mask output.
        Returns sorted candidate masks with refined binary masks and scores.
        """
        """
        Takes an image and a bounding box [x_min, y_min, x_max, y_max] in permille (0-1000)
        from Agent A, converts it to absolute pixel coordinates, and predicts candidate masks.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if self.safe_mode and respect_safe_mode:
            if self.safe_strategy == "multimask":
                box_scales = [1.00]
                multimask_output = True
                max_candidates = 3
            elif self.safe_strategy == "multimask_refine":
                box_scales = [1.00]
                multimask_output = True
                max_candidates = 6
            elif self.safe_strategy == "multimask_point_refine":
                box_scales = [1.00]
                multimask_output = True
                max_candidates = 8
            elif self.safe_strategy == "multimask_refine_smallboost":
                box_scales = [1.00]
                multimask_output = True
                max_candidates = 10
            else:
                box_scales = [0.95, 1.00, 1.05]
                multimask_output = False
                max_candidates = 3
        elif box_scales is None:
            box_scales = [0.90, 1.00, 1.10]

        image_rgb = self._load_image_and_prepare_predictor(image_path)
        h, w, _ = image_rgb.shape
        base_box = self._permille_box_to_pixels(box_2d, w, h)
        print(f"Received Agent A's permille box: {box_2d}")
        print(f"Mapped to base pixel coordinates: {base_box}")

        all_candidates: List[Dict[str, Any]] = []
        for scale in box_scales:
            input_box = self._expand_box(base_box, w, h, scale)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=multimask_output,
            )
            for idx, (mask, score, logit_map) in enumerate(zip(masks, scores, logits)):
                refined_mask = self._post_process_mask(mask)
                candidate = {
                    "mask": refined_mask,
                    "sam_score": float(score),
                    "score": self._candidate_score(refined_mask, float(score), input_box),
                    "box_pixels": input_box.copy(),
                    "box_scale": float(scale),
                    "candidate_index": int(idx),
                    "score_map": np.array(logit_map, dtype=np.float32),
                }
                all_candidates.append(candidate)

        if use_box_center_point:
            center_x = float((base_box[0] + base_box[2]) / 2.0)
            center_y = float((base_box[1] + base_box[3]) / 2.0)
            point_coords = np.array([[center_x, center_y]], dtype=np.float32)
            for scale in box_scales:
                point_box = self._expand_box(base_box, w, h, scale)
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=np.array([1], dtype=np.int32),
                    box=point_box[None, :],
                    multimask_output=True,
                )
                for idx, (mask, score, logit_map) in enumerate(zip(masks, scores, logits)):
                    refined_mask = self._post_process_mask(mask)
                    candidate = {
                        "mask": refined_mask,
                        "sam_score": float(score),
                        "score": self._candidate_score(refined_mask, float(score), point_box),
                        "box_pixels": point_box.copy(),
                        "box_scale": float(scale),
                        "candidate_index": int(idx),
                        "used_box_center_point": True,
                        "score_map": np.array(logit_map, dtype=np.float32),
                    }
                    all_candidates.append(candidate)

        if (
            self.safe_mode
            and self.safe_strategy
            in {"multimask_refine", "multimask_point_refine", "multimask_refine_smallboost"}
            and all_candidates
        ):
            seed_candidates = sorted(all_candidates, key=lambda item: item["score"], reverse=True)[
                :2
            ]
            for seed_idx, seed in enumerate(seed_candidates):
                refined_box = self._mask_to_box(seed["mask"])
                if refined_box is None:
                    continue
                refined_box = self._expand_box(refined_box, w, h, 1.08)
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=refined_box[None, :],
                    multimask_output=True,
                )
                for idx, (mask, score, logit_map) in enumerate(zip(masks, scores, logits)):
                    refined_mask = self._post_process_mask(mask)
                    candidate = {
                        "mask": refined_mask,
                        "sam_score": float(score),
                        "score": self._candidate_score(refined_mask, float(score), refined_box),
                        "box_pixels": refined_box.copy(),
                        "box_scale": float(1.08),
                        "candidate_index": int(idx),
                        "refined_from_seed": int(seed_idx),
                        "score_map": np.array(logit_map, dtype=np.float32),
                    }
                    all_candidates.append(candidate)

        if self.safe_mode and self.safe_strategy == "multimask_point_refine" and all_candidates:
            point_seed_candidates = sorted(
                all_candidates, key=lambda item: item["score"], reverse=True
            )[:2]
            for seed_idx, seed in enumerate(point_seed_candidates):
                point_coords = self._mask_centroid(seed["mask"])
                if point_coords is None:
                    continue
                point_box = self._mask_to_box(seed["mask"])
                if point_box is None:
                    continue
                point_box = self._expand_box(point_box, w, h, 1.12)
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=np.array([1], dtype=np.int32),
                    box=point_box[None, :],
                    multimask_output=True,
                )
                for idx, (mask, score, logit_map) in enumerate(zip(masks, scores, logits)):
                    refined_mask = self._post_process_mask(mask)
                    candidate = {
                        "mask": refined_mask,
                        "sam_score": float(score),
                        "score": self._candidate_score(refined_mask, float(score), point_box),
                        "box_pixels": point_box.copy(),
                        "box_scale": float(1.12),
                        "candidate_index": int(idx),
                        "refined_from_seed": int(seed_idx),
                        "used_point_refine": True,
                        "score_map": np.array(logit_map, dtype=np.float32),
                    }
                    all_candidates.append(candidate)

        if (
            self.safe_mode
            and self.safe_strategy == "multimask_refine_smallboost"
            and all_candidates
        ):
            boosted_candidates = sorted(
                all_candidates, key=lambda item: item["score"], reverse=True
            )[:3]
            for seed_idx, seed in enumerate(boosted_candidates):
                refined_box = self._mask_to_box(seed["mask"])
                if refined_box is None:
                    continue
                mask_area_ratio = float(seed["mask"].sum() / max(1, h * w))
                box_area_ratio = self._box_area_ratio(refined_box, w, h)
                if mask_area_ratio > 0.02 and box_area_ratio > 0.05:
                    continue
                for boost_scale in (0.85, 1.00, 1.20):
                    boosted_box = self._expand_box(refined_box, w, h, boost_scale)
                    masks, scores, logits = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boosted_box[None, :],
                        multimask_output=True,
                    )
                    for idx, (mask, score, logit_map) in enumerate(zip(masks, scores, logits)):
                        refined_mask = self._post_process_mask(mask)
                        candidate = {
                            "mask": refined_mask,
                            "sam_score": float(score),
                            "score": self._candidate_score(refined_mask, float(score), boosted_box),
                            "box_pixels": boosted_box.copy(),
                            "box_scale": float(boost_scale),
                            "candidate_index": int(idx),
                            "refined_from_seed": int(seed_idx),
                            "used_smallboost": True,
                            "score_map": np.array(logit_map, dtype=np.float32),
                        }
                        all_candidates.append(candidate)

        all_candidates.sort(key=lambda item: item["score"], reverse=True)
        return all_candidates[:max_candidates]

    def extract_mask(self, image_path: str, box_2d: list, output_path: str = "sam_output.jpg"):
        """
        Backward-compatible single-best extraction API.
        """
        candidates = self.extract_mask_candidates(image_path, box_2d, output_path=output_path)
        if not candidates:
            raise RuntimeError("SAM did not return any candidate masks.")

        best_candidate = candidates[0]
        best_mask = best_candidate["mask"]
        confidence = best_candidate["score"]
        box_pixels = best_candidate["box_pixels"]

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"SAM prediction complete. Ensemble confidence score: {confidence:.4f}")
        self._visualize_and_save(image_rgb, box_pixels, best_mask, confidence, output_path)

        raw_mask_path = output_path.replace(".jpg", "_raw_mask.png")
        raw_score_map_path = output_path.replace(".jpg", "_raw_score.npy")
        raw_mask_uint8 = (best_mask.astype(np.uint8)) * 255
        cv2.imwrite(raw_mask_path, raw_mask_uint8)
        np.save(
            raw_score_map_path,
            best_candidate.get("score_map", np.zeros((256, 256), dtype=np.float32)),
        )
        return best_mask, confidence, raw_mask_path, raw_score_map_path

    def _visualize_and_save(self, image, box, mask, score, output_path):
        """
        Helper method to overlay the mask and box on the image and save it.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # Show mask (apply a colored semi-transparent overlay)
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])  # DodgerBlue with 0.6 alpha
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image)

        # Show box
        x0, y0, x1, y1 = box
        plt.gca().add_patch(
            plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", facecolor="none", lw=2)
        )

        plt.title(f"SAM Precision Tracker (Confidence: {score:.3f})")
        plt.axis("off")

        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        print(f"Visualization saved to: {output_path}")
        plt.close()
