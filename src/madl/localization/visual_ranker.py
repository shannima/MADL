from __future__ import annotations

import hashlib
import io
import json
import numbers
import os
import re
import stat
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models as tvm

from madl.localization.candidate_pool import build_local_anomaly_heatmap

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
GEOMETRY_FEATURE_DIM = 24
SELECTION_TAXONOMY = "learned visual mask ranker"
LEGACY_CHECKPOINT_KEYS = {"epoch", "model", "args", "val"}
LEGACY_MODEL_ARG_KEYS = frozenset(
    {
        "image_size",
        "use_geometry_features",
        "use_source_embedding",
        "source_vocab_size",
        "source_embedding_dim",
    }
)
PRODUCTION_LEGACY_ARG_KEYS = frozenset(
    {
        "augment_train",
        "batch_size",
        "device",
        "epochs",
        "focus_hard_only",
        "focus_pairs_json",
        "focus_repeats",
        "freeze_backbone",
        "freeze_visual_branch",
        "grouped_batches",
        "hard_negative_weight",
        "image_size",
        "init_checkpoint",
        "listwise_weight",
        "lr",
        "max_candidates_per_image",
        "no_pretrained",
        "num_workers",
        "output_dir",
        "pairwise_weight",
        "persistent_workers",
        "pin_memory",
        "pointwise_weight",
        "prefetch_factor",
        "ranking_margin",
        "seed",
        "source_embedding_dim",
        "source_vocab_size",
        "target_gap",
        "train_image_dir",
        "train_json",
        "trust_replay_image_paths",
        "trust_replay_mask_paths",
        "trusted_image_ext",
        "use_geometry_features",
        "use_source_embedding",
        "val_image_dir",
        "val_json",
        "weight_decay",
    }
)
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_MODEL_STATE_BYTES = 256 * 1024 * 1024
FIXED_BACKBONE_STATE_BUDGET_BYTES = 192 * 1024 * 1024
MAX_CHECKPOINT_CONTROLLED_MODEL_BYTES = 64 * 1024 * 1024
MAX_CANDIDATE_TENSOR_BYTES = 256 * 1024 * 1024
MAX_IMAGE_SIZE = 4096
MAX_SOURCE_VOCAB_SIZE = 1_000_000
MAX_SOURCE_EMBEDDING_DIM = 4096
FORBIDDEN_EVAL_KEYS = {"mask_iou", "target", "gt", "gt_mask", "ground_truth"}


def _positive_int(value: Any, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < minimum:
        raise ValueError(f"{field} must be a non-boolean integer >= {minimum}.")
    return int(value)


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean.")
    return value


def _bounded_product(factors: Sequence[int], *, limit: int, field: str) -> int:
    result = 1
    for factor in factors:
        if factor < 0 or (factor and result > limit // factor):
            raise ValueError(f"{field} exceeds its safe resource budget.")
        result *= factor
    return result


def _validate_model_config_budget(config: Mapping[str, Any]) -> None:
    image_size = _positive_int(config["image_size"], field="image_size")
    vocab_size = _positive_int(config["source_vocab_size"], field="source_vocab_size", minimum=2)
    embedding_dim = _positive_int(config["source_embedding_dim"], field="source_embedding_dim")
    if image_size > MAX_IMAGE_SIZE:
        raise ValueError("image_size exceeds the checkpoint-controlled dimension limit.")
    if vocab_size > MAX_SOURCE_VOCAB_SIZE:
        raise ValueError("source_vocab_size exceeds the checkpoint-controlled dimension limit.")
    if embedding_dim > MAX_SOURCE_EMBEDDING_DIM:
        raise ValueError("source_embedding_dim exceeds the checkpoint-controlled dimension limit.")
    _bounded_product(
        (5, image_size, image_size, 4),
        limit=MAX_CANDIDATE_TENSOR_BYTES,
        field="candidate tensor bytes",
    )
    use_geometry = _strict_bool(config["use_geometry_features"], field="use_geometry_features")
    use_source = _strict_bool(config["use_source_embedding"], field="use_source_embedding")
    parameter_limit = MAX_CHECKPOINT_CONTROLLED_MODEL_BYTES // 4
    embedding_parameters = (
        _bounded_product(
            (vocab_size, embedding_dim),
            limit=parameter_limit,
            field="source embedding parameters",
        )
        if use_source
        else 0
    )
    metadata_dim = (GEOMETRY_FEATURE_DIM if use_geometry else 0) + (
        embedding_dim if use_source else 0
    )
    metadata_parameters = 0
    if metadata_dim:
        metadata_parameters = 2 * metadata_dim + 64 * metadata_dim + 64 + 64 + 1
    if embedding_parameters + metadata_parameters > parameter_limit:
        raise ValueError(
            "checkpoint-controlled model parameters exceed the incremental byte budget."
        )


def _finite_real(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{field} must be a non-boolean real number.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be <= {maximum}.")
    return number


def _numeric_array(value: Any, *, field: str, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != ndim or value.size == 0:
        raise ValueError(f"{field} must be a non-empty {ndim}D numpy array.")
    if value.dtype == np.bool_ or not (
        np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)
    ):
        raise ValueError(f"{field} must have a real numeric non-boolean dtype.")
    if not np.isfinite(value).all():
        raise ValueError(f"{field} must contain only finite values.")
    return value


def _is_forbidden_eval_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    compact = normalized.replace("_", "")
    tokens = tuple(token for token in normalized.split("_") if token)
    if normalized in FORBIDDEN_EVAL_KEYS or normalized == "target":
        return True
    if (
        tokens
        and tokens[0] == "target"
        and len(tokens) > 1
        and tokens[1]
        in {
            "label",
            "iou",
            "mask",
            "score",
            "eval",
            "metric",
            "value",
        }
    ):
        return True
    if compact.startswith(
        (
            "targetlabel",
            "targetiou",
            "targetmask",
            "targetscore",
            "targeteval",
            "targetmetric",
            "targetvalue",
        )
    ):
        return True
    if "groundtruth" in compact or "maskiou" in compact or "selectediou" in compact:
        return True
    if "oracle" in tokens or "oracle" in compact:
        return True
    if normalized == "iou" or "iou" in tokens or compact.endswith("iou"):
        return True
    if any(
        fragment in compact
        for fragment in (
            "maskiou",
            "selectediou",
            "boxiou",
            "candidateiou",
            "meaniou",
            "bestiou",
            "referencemask",
            "goldmask",
            "truthmask",
        )
    ):
        return True
    if normalized == "gt" or normalized.startswith("gt_"):
        return True
    return compact.startswith(
        ("gtlabel", "gtmask", "gtiou", "gtscore", "gteval", "gtmetric", "gtvalue", "gttruth")
    )


def _load_json_rows(input_json: str) -> List[Dict[str, Any]]:
    if not isinstance(input_json, str) or not input_json:
        raise ValueError("input_json must be a non-empty string path.")
    path = Path(input_json)
    if not path.is_file():
        raise FileNotFoundError(input_json)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        if "per_image" not in data or not isinstance(data["per_image"], list):
            raise ValueError("Visual candidate-ranker JSON must contain a per_image list.")
        rows = data["per_image"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported visual candidate-ranker JSON: {input_json}")
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Visual candidate-ranker rows must be a non-empty list of mappings.")
    return [dict(row) for row in rows]


def _stem_variants(stem: str) -> List[str]:
    variants = [str(stem)]
    match = re.match(r"^(.*?)(\d+)$", str(stem))
    if match:
        prefix, digits = match.groups()
        number = int(digits)
        for width in (6, 5):
            variants.append(f"{prefix}{number:0{width}d}")
        variants.append(f"{prefix}{number}")
    deduped: List[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _image_path_candidates(image_dir: str, stem: str) -> List[str]:
    candidates: List[str] = []
    for variant in _stem_variants(stem):
        for suffix in IMAGE_EXTS:
            path = os.path.join(image_dir, f"{variant}{suffix}")
            if os.path.exists(path):
                candidates.append(path)
    return candidates


def _image_path_for(image_dir: str, stem: str) -> str:
    candidates = _image_path_candidates(image_dir, stem)
    if candidates:
        return candidates[0]
    return ""


def _trusted_image_path_for(image_dir: str, stem: str, image_ext: str = ".png") -> str:
    extension = str(image_ext or ".png")
    if not extension.startswith("."):
        extension = f".{extension}"
    trusted_stem = str(stem)
    match = re.match(r"^(.*?)(\d+)$", trusted_stem)
    if match:
        prefix, digits = match.groups()
        number = int(digits)
        trusted_stem = f"{prefix}{number:06d}" if len(digits) < 6 else trusted_stem
    return os.path.join(image_dir, f"{trusted_stem}{extension}")


def _build_image_path_index(image_dir: str) -> Dict[str, str]:
    exact: Dict[str, str] = {}
    variants: Dict[str, str] = {}
    try:
        entries = list(os.scandir(image_dir))
    except OSError:
        return {}
    for entry in entries:
        name = entry.name
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        stem = Path(name).stem
        path = entry.path
        exact[stem] = path
        for variant in _stem_variants(stem):
            variants.setdefault(variant, path)
    variants.update(exact)
    return variants


def _read_image_with_fallback(
    image_path: str,
    *,
    retries: int = 3,
    delay_seconds: float = 0.05,
    allow_fallback: bool = True,
) -> Tuple[np.ndarray, str]:
    candidates = (
        _image_path_candidates(str(Path(image_path).parent), str(Path(image_path).stem))
        if allow_fallback
        else []
    )
    if image_path and image_path not in candidates:
        candidates.insert(0, image_path)
    for path in candidates:
        for attempt in range(max(1, int(retries))):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is not None:
                return image, path
            if attempt + 1 < max(1, int(retries)):
                time.sleep(float(delay_seconds))
    raise FileNotFoundError(image_path)


def _mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        height, width = mask.shape[:2]
        return 0, 0, width, height
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _pad_box(
    box: Tuple[int, int, int, int], width: int, height: int, pad_ratio: float = 0.20
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    pad_x = int(round(max(1, (x1 - x0) * pad_ratio)))
    pad_y = int(round(max(1, (y1 - y0) * pad_ratio)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def _candidate_number(candidate: Mapping[str, Any], field: str, default: float) -> float:
    if field not in candidate:
        return float(default)
    return _finite_real(candidate[field], field=field)


def _clip01(value: float) -> float:
    return float(np.clip(_finite_real(value, field="geometry feature"), 0.0, 1.0))


def source_bucket_id(source: str, vocab_size: int = 128) -> int:
    """Map sparse proposal-source strings to a stable embedding bucket."""
    if not isinstance(source, str):
        raise ValueError("source must be a string.")
    vocab_size = _positive_int(vocab_size, field="vocab_size", minimum=2)
    if not source:
        return 0
    digest = hashlib.md5(str(source).encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % (vocab_size - 1))


def build_candidate_geometry_features(
    candidate: Mapping[str, Any],
    raw_mask: np.ndarray,
    image_shape: Tuple[int, int] | Tuple[int, int, int],
) -> np.ndarray:
    """Encode absolute position, mask geometry, and numeric forensic priors."""
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be a mapping.")
    raw_mask = _numeric_array(raw_mask, field="raw_mask", ndim=2)
    if not isinstance(image_shape, tuple) or len(image_shape) not in (2, 3):
        raise ValueError("image_shape must be a 2D or 3D shape tuple.")
    height = _positive_int(image_shape[0], field="image_shape height")
    width = _positive_int(image_shape[1], field="image_shape width")
    if raw_mask.shape[:2] != (height, width):
        raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (raw_mask > 0).astype(np.uint8)
    x0, y0, x1, y1 = _mask_bbox(binary)
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    bbox_area = float(bbox_w * bbox_h) / float(width * height)
    mask_area = float(np.count_nonzero(binary)) / float(width * height)
    fill = float(mask_area / max(1e-6, bbox_area))
    features = np.asarray(
        [
            _clip01(x0 / width),
            _clip01(y0 / height),
            _clip01(x1 / width),
            _clip01(y1 / height),
            _clip01((x0 + x1) / (2.0 * width)),
            _clip01((y0 + y1) / (2.0 * height)),
            _clip01(bbox_w / width),
            _clip01(bbox_h / height),
            _clip01(bbox_area),
            _clip01(mask_area),
            _clip01(fill),
            _clip01(_candidate_number(candidate, "mask_area_ratio", mask_area)),
            _clip01(_candidate_number(candidate, "box_fill_ratio", fill)),
            _clip01(_candidate_number(candidate, "edge_contact_ratio", 0.0)),
            _clip01(_candidate_number(candidate, "tight_bbox_ratio", 0.0)),
            _clip01(_candidate_number(candidate, "box_scale", 0.0) / 3.0),
            _clip01(_candidate_number(candidate, "candidate_index", 0.0) / 128.0),
            _clip01(_candidate_number(candidate, "region_rank", 0.0) / 64.0),
            _clip01(_candidate_number(candidate, "sam_score", 0.0)),
            _clip01(_candidate_number(candidate, "visual_mask_score", 0.0) / 2.0),
            _clip01(_candidate_number(candidate, "pixel_pool_score", 0.0) / 2.0),
            _clip01(_candidate_number(candidate, "proposal_confidence", 0.0)),
            _clip01(_candidate_number(candidate, "heatmap_contrast", 0.0)),
            _clip01(_candidate_number(candidate, "heatmap_mass_coverage", 0.0)),
        ],
        dtype=np.float32,
    )
    if features.shape != (GEOMETRY_FEATURE_DIM,) or not np.isfinite(features).all():
        raise ValueError("geometry features must have the frozen finite shape.")
    return features


def _to_tensor(
    crop_rgb: np.ndarray, crop_mask: np.ndarray, crop_heatmap: np.ndarray, image_size: int
) -> torch.Tensor:
    rgb = (
        cv2.resize(crop_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        / 255.0
    )
    mask = cv2.resize(crop_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST).astype(
        np.float32
    )
    heatmap = cv2.resize(
        crop_heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    stacked = np.concatenate([rgb.transpose(2, 0, 1), mask[None, ...], heatmap[None, ...]], axis=0)
    tensor = torch.from_numpy(stacked.astype(np.float32))
    if (
        tensor.shape != (5, image_size, image_size)
        or tensor.dtype != torch.float32
        or not torch.isfinite(tensor).all()
    ):
        raise ValueError("visual candidate tensor must have frozen finite 5xSxS float32 shape.")
    return tensor


@dataclass
class VisualCandidateSample:
    stem: str
    image_path: str
    raw_mask: str
    target: float
    sam_score: float
    proposal_source: str
    region_id: str
    candidate: Dict[str, Any] | None = None


class VisualCandidateDataset(Dataset):
    def __init__(
        self,
        input_json: str,
        image_dir: str,
        image_size: int = 160,
        max_candidates_per_image: int = 0,
        augment: bool = False,
        source_vocab_size: int = 128,
        image_cache_limit: int = 16,
        verify_raw_masks: bool = True,
        trust_image_paths: bool = False,
        trusted_image_ext: str = ".png",
    ) -> None:
        if not isinstance(image_dir, str) or not image_dir:
            raise ValueError("image_dir must be a non-empty string path.")
        if not Path(image_dir).is_dir():
            raise FileNotFoundError(image_dir)
        self.input_json = input_json
        self.image_dir = image_dir
        self.image_size = _positive_int(image_size, field="image_size")
        max_candidates_per_image = _positive_int(
            max_candidates_per_image, field="max_candidates_per_image", minimum=0
        )
        self.augment = _strict_bool(augment, field="augment")
        self.source_vocab_size = _positive_int(
            source_vocab_size, field="source_vocab_size", minimum=2
        )
        self.image_cache_limit = _positive_int(image_cache_limit, field="image_cache_limit")
        self.verify_raw_masks = _strict_bool(verify_raw_masks, field="verify_raw_masks")
        self.trust_image_paths = _strict_bool(trust_image_paths, field="trust_image_paths")
        if not isinstance(trusted_image_ext, str) or not trusted_image_ext:
            raise ValueError("trusted_image_ext must be a non-empty string.")
        self.trusted_image_ext = trusted_image_ext
        self._image_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.samples: List[VisualCandidateSample] = []
        image_index = _build_image_path_index(image_dir) if self.trust_image_paths else {}
        for row in _load_json_rows(input_json):
            stem = row.get("stem")
            if not isinstance(stem, str) or not stem:
                raise ValueError("Each training row must contain a non-empty string stem.")
            image_path = (
                image_index.get(stem)
                or _trusted_image_path_for(image_dir, stem, self.trusted_image_ext)
                if self.trust_image_paths
                else _image_path_for(image_dir, stem)
            )
            if not image_path or not Path(image_path).is_file():
                raise FileNotFoundError(f"No image found for training stem: {stem}")
            if cv2.imread(image_path, cv2.IMREAD_COLOR) is None:
                # Replay manifests may point at a stale exact stem while a
                # canonical zero-padded sibling is available.  Resolve that
                # fallback during normal training; an explicitly trusted
                # metadata-only manifest may defer image readability until
                # __getitem__.
                if self.trust_image_paths and not self.verify_raw_masks:
                    pass
                else:
                    _read_image_with_fallback(
                        image_path,
                        allow_fallback=not self.trust_image_paths,
                    )
            if "extraction_diagnostics" not in row or not isinstance(
                row["extraction_diagnostics"], list
            ):
                raise ValueError("Each training row must contain an extraction_diagnostics list.")
            candidates = row["extraction_diagnostics"]
            if not candidates or any(
                not isinstance(candidate, Mapping) for candidate in candidates
            ):
                raise ValueError("Training diagnostics must be a non-empty list of mappings.")
            if max_candidates_per_image > 0:
                candidates = candidates[:max_candidates_per_image]
            for candidate in candidates:
                raw_mask = candidate.get("raw_mask")
                if not isinstance(raw_mask, str) or not raw_mask:
                    raise ValueError("Each training diagnostic must contain a raw_mask path.")
                if self.verify_raw_masks:
                    if not Path(raw_mask).is_file():
                        raise ValueError(f"Training mask is missing: {raw_mask}")
                    if cv2.imread(raw_mask, cv2.IMREAD_GRAYSCALE) is None:
                        raise ValueError(f"Training mask is unreadable: {raw_mask}")
                if "mask_iou" not in candidate:
                    raise ValueError("Each training diagnostic must contain mask_iou.")
                target = _finite_real(
                    candidate["mask_iou"], field="mask_iou", minimum=0.0, maximum=1.0
                )
                sam_score = _finite_real(
                    candidate.get("sam_score", 0.0), field="sam_score", minimum=0.0, maximum=1.0
                )
                proposal_source = candidate.get("proposal_source", "")
                region_id = candidate.get("region_id", "")
                if not isinstance(proposal_source, str) or not isinstance(region_id, str):
                    raise ValueError("proposal_source and region_id must be strings.")
                self.samples.append(
                    VisualCandidateSample(
                        stem=stem,
                        image_path=image_path,
                        raw_mask=raw_mask,
                        target=target,
                        sam_score=sam_score,
                        proposal_source=proposal_source,
                        region_id=region_id,
                        candidate=dict(candidate),
                    )
                )
        if not self.samples:
            raise ValueError(f"No visual candidate samples found in {input_json}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_and_heatmap(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._image_cache.get(image_path)
        if cached is not None:
            self._image_cache.move_to_end(image_path)
            return cached
        image_bgr, resolved_path = _read_image_with_fallback(
            image_path, allow_fallback=not self.trust_image_paths
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        heatmap = build_local_anomaly_heatmap(image_bgr)
        self._image_cache[image_path] = (image_rgb, heatmap)
        self._image_cache.move_to_end(image_path)
        if resolved_path != image_path:
            self._image_cache[resolved_path] = (image_rgb, heatmap)
            self._image_cache.move_to_end(resolved_path)
        while len(self._image_cache) > self.image_cache_limit:
            self._image_cache.popitem(last=False)
        return image_rgb, heatmap

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image_rgb, heatmap = self._load_image_and_heatmap(sample.image_path)
        raw_mask = cv2.imread(sample.raw_mask, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(sample.raw_mask)
        tensor = build_visual_candidate_tensor_from_arrays(
            image_rgb=image_rgb,
            heatmap=heatmap,
            raw_mask=raw_mask,
            image_size=self.image_size,
        )
        candidate = sample.candidate or {}
        geometry = build_candidate_geometry_features(candidate, raw_mask, image_rgb.shape)
        if self.augment:
            tensor = augment_visual_candidate_tensor(tensor)
        return {
            "input": tensor,
            "target": torch.tensor(sample.target, dtype=torch.float32),
            "sam_score": torch.tensor(sample.sam_score, dtype=torch.float32),
            "geometry": torch.from_numpy(geometry.astype(np.float32)),
            "source_id": torch.tensor(
                source_bucket_id(sample.proposal_source, self.source_vocab_size), dtype=torch.long
            ),
            "stem": sample.stem,
            "raw_mask": sample.raw_mask,
            "proposal_source": sample.proposal_source,
            "region_id": sample.region_id,
        }


def augment_visual_candidate_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clone()
    if torch.rand(()) < 0.50:
        tensor = torch.flip(tensor, dims=(2,))
    if torch.rand(()) < 0.20:
        tensor = torch.flip(tensor, dims=(1,))
    rgb = tensor[:3]
    contrast = 0.88 + 0.24 * torch.rand(())
    brightness = -0.06 + 0.12 * torch.rand(())
    tensor[:3] = torch.clamp((rgb - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    if torch.rand(()) < 0.25:
        tensor[:3] = torch.clamp(tensor[:3] + 0.015 * torch.randn_like(tensor[:3]), 0.0, 1.0)
    return tensor


def build_visual_candidate_tensor_from_arrays(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    raw_mask: np.ndarray,
    image_size: int = 160,
) -> torch.Tensor:
    image_size = _positive_int(image_size, field="image_size")
    image_rgb = _numeric_array(image_rgb, field="image_rgb", ndim=3)
    heatmap = _numeric_array(heatmap, field="heatmap", ndim=2)
    raw_mask = _numeric_array(raw_mask, field="raw_mask", ndim=2)
    if image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    height, width = image_rgb.shape[:2]
    if raw_mask.shape[:2] != (height, width):
        raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if heatmap.shape[:2] != (height, width):
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    x0, y0, x1, y1 = _pad_box(_mask_bbox(raw_mask), width, height)
    return _to_tensor(
        image_rgb[y0:y1, x0:x1],
        raw_mask[y0:y1, x0:x1],
        heatmap[y0:y1, x0:x1],
        image_size,
    )


def build_visual_candidate_tensor(
    image_path: str, raw_mask_path: str, image_size: int = 160
) -> torch.Tensor:
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string.")
    if not isinstance(raw_mask_path, str) or not raw_mask_path:
        raise ValueError("raw_mask_path must be a non-empty string.")
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw_mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        raise FileNotFoundError(raw_mask_path)
    heatmap = build_local_anomaly_heatmap(image_bgr)
    return build_visual_candidate_tensor_from_arrays(
        image_rgb=image_rgb,
        heatmap=heatmap,
        raw_mask=raw_mask,
        image_size=image_size,
    )


class ConvNeXtCandidateRanker(nn.Module):
    def __init__(
        self,
        image_size: int = 160,
        use_pretrained: bool = False,
        dropout: float = 0.10,
        use_geometry_features: bool = False,
        use_source_embedding: bool = False,
        source_vocab_size: int = 128,
        source_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.image_size = _positive_int(image_size, field="image_size")
        use_pretrained = _strict_bool(use_pretrained, field="use_pretrained")
        self.use_geometry_features = _strict_bool(
            use_geometry_features, field="use_geometry_features"
        )
        self.use_source_embedding = _strict_bool(use_source_embedding, field="use_source_embedding")
        self.source_vocab_size = _positive_int(
            source_vocab_size, field="source_vocab_size", minimum=2
        )
        self.source_embedding_dim = _positive_int(
            source_embedding_dim, field="source_embedding_dim"
        )
        dropout = _finite_real(dropout, field="dropout", minimum=0.0)
        if dropout >= 1.0:
            raise ValueError("dropout must be < 1.0.")
        _validate_model_config_budget(
            {
                "image_size": self.image_size,
                "use_geometry_features": self.use_geometry_features,
                "use_source_embedding": self.use_source_embedding,
                "source_vocab_size": self.source_vocab_size,
                "source_embedding_dim": self.source_embedding_dim,
            }
        )
        try:
            weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if use_pretrained else None
            backbone = tvm.convnext_tiny(weights=weights)
        except TypeError:
            backbone = tvm.convnext_tiny(pretrained=use_pretrained)
        self.input_adapter = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1),
            nn.GELU(),
        )
        self.features = backbone.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Dropout(float(dropout)),
            nn.Linear(768, 192),
            nn.GELU(),
            nn.Linear(192, 1),
        )
        metadata_dim = 0
        if self.use_geometry_features:
            metadata_dim += GEOMETRY_FEATURE_DIM
        if self.use_source_embedding:
            self.source_embedding = nn.Embedding(self.source_vocab_size, self.source_embedding_dim)
            metadata_dim += self.source_embedding_dim
        else:
            self.source_embedding = None
        self.metadata_head = (
            nn.Sequential(
                nn.LayerNorm(metadata_dim),
                nn.Linear(metadata_dim, 64),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(64, 1),
            )
            if metadata_dim > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        geometry: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 4
            or x.shape[0] <= 0
            or x.shape[1] != 5
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
        ):
            raise ValueError("x must be a finite floating tensor with shape Nx5xHxW.")
        features = self.features(self.input_adapter(x))
        visual_logit = self.head(features).squeeze(-1)
        if self.metadata_head is None:
            if visual_logit.shape != (x.shape[0],) or not torch.isfinite(visual_logit).all():
                raise ValueError("model output must contain one finite logit per candidate.")
            return visual_logit
        metadata_parts: List[torch.Tensor] = []
        if self.use_geometry_features:
            if (
                not isinstance(geometry, torch.Tensor)
                or geometry.shape != (x.shape[0], GEOMETRY_FEATURE_DIM)
                or not geometry.is_floating_point()
                or not torch.isfinite(geometry).all()
            ):
                raise ValueError(
                    "geometry must be a finite floating NxGEOMETRY_FEATURE_DIM tensor."
                )
            metadata_parts.append(geometry.to(device=x.device, dtype=x.dtype))
        if self.use_source_embedding:
            if (
                not isinstance(source_ids, torch.Tensor)
                or source_ids.shape != (x.shape[0],)
                or source_ids.dtype == torch.bool
                or source_ids.is_floating_point()
            ):
                raise ValueError(
                    "source_ids must be an integer vector with one entry per candidate."
                )
            source_ids = source_ids.to(device=x.device, dtype=torch.long)
            if torch.any(source_ids < 0) or torch.any(source_ids >= self.source_vocab_size):
                raise ValueError("source_ids are outside the configured vocabulary.")
            metadata_parts.append(self.source_embedding(source_ids).to(dtype=x.dtype))
        metadata = torch.cat(metadata_parts, dim=1)
        output = visual_logit + self.metadata_head(metadata).squeeze(-1)
        if output.shape != (x.shape[0],) or not torch.isfinite(output).all():
            raise ValueError("model output must contain one finite logit per candidate.")
        return output


def _group_indices(group_ids: Sequence[Any]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(str(group_id), []).append(index)
    return grouped


def visual_candidate_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_ids: Sequence[Any],
    *,
    sam_scores: torch.Tensor | None = None,
    pointwise_weight: float = 1.0,
    pairwise_weight: float = 0.35,
    listwise_weight: float = 0.15,
    hard_negative_weight: float = 1.0,
    margin: float = 0.20,
    target_gap: float = 0.20,
) -> Dict[str, torch.Tensor]:
    """Combine pointwise IoU supervision with per-image ranking objectives."""
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 1
        or logits.numel() == 0
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("logits must be a non-empty finite floating vector.")
    if (
        not isinstance(targets, torch.Tensor)
        or targets.shape != logits.shape
        or not targets.is_floating_point()
        or not torch.isfinite(targets).all()
        or torch.any(targets < 0.0)
        or torch.any(targets > 1.0)
    ):
        raise ValueError("targets must be a finite [0, 1] floating vector matching logits.")
    if isinstance(group_ids, (str, bytes, Mapping)) or not isinstance(group_ids, Sequence):
        raise ValueError("group_ids must be a sequence with one group per logit.")
    if len(group_ids) != logits.numel() or any(
        not isinstance(group_id, str) or not group_id for group_id in group_ids
    ):
        raise ValueError("group_ids must contain one non-empty string per logit.")
    weights = {
        "pointwise_weight": pointwise_weight,
        "pairwise_weight": pairwise_weight,
        "listwise_weight": listwise_weight,
        "hard_negative_weight": hard_negative_weight,
        "margin": margin,
        "target_gap": target_gap,
    }
    validated_weights = {
        name: _finite_real(value, field=name, minimum=0.0) for name, value in weights.items()
    }
    logits = logits.float()
    targets = targets.float()
    pointwise_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    pairwise_terms: List[torch.Tensor] = []
    listwise_terms: List[torch.Tensor] = []
    if sam_scores is None:
        sam_scores = torch.zeros_like(targets)
    else:
        if (
            not isinstance(sam_scores, torch.Tensor)
            or sam_scores.shape != targets.shape
            or not sam_scores.is_floating_point()
            or not torch.isfinite(sam_scores).all()
            or torch.any(sam_scores < 0.0)
            or torch.any(sam_scores > 1.0)
        ):
            raise ValueError("sam_scores must be a finite [0, 1] floating vector matching targets.")
        sam_scores = sam_scores.float().to(targets.device)

    for indices in _group_indices(group_ids).values():
        if len(indices) < 2:
            continue
        idx = torch.as_tensor(indices, dtype=torch.long, device=targets.device)
        group_logits = logits.index_select(0, idx)
        group_targets = targets.index_select(0, idx)
        group_sam = sam_scores.index_select(0, idx)
        best_local = int(torch.argmax(group_targets).item())
        best_logit = group_logits[best_local]
        best_target = group_targets[best_local]
        for neg_index in range(len(indices)):
            if neg_index == best_local:
                continue
            neg_target = group_targets[neg_index]
            if best_target - neg_target < validated_weights["target_gap"]:
                continue
            hard_weight = 1.0 + validated_weights["hard_negative_weight"] * float(
                group_sam[neg_index] >= 0.90 and neg_target <= 0.30
            )
            pairwise_terms.append(
                torch.relu(
                    torch.as_tensor(validated_weights["margin"], device=targets.device)
                    - (best_logit - group_logits[neg_index])
                )
                * float(hard_weight)
            )
        if torch.max(group_targets) > 0.0:
            soft_targets = torch.softmax(group_targets / 0.20, dim=0)
            log_probs = torch.log_softmax(group_logits, dim=0)
            listwise_terms.append(-(soft_targets * log_probs).sum())

    pairwise_loss = torch.stack(pairwise_terms).mean() if pairwise_terms else logits.new_tensor(0.0)
    listwise_loss = torch.stack(listwise_terms).mean() if listwise_terms else logits.new_tensor(0.0)
    total_loss = (
        validated_weights["pointwise_weight"] * pointwise_loss
        + validated_weights["pairwise_weight"] * pairwise_loss
        + validated_weights["listwise_weight"] * listwise_loss
    )
    result = {
        "total_loss": total_loss,
        "pointwise_loss": pointwise_loss.detach(),
        "pairwise_loss": pairwise_loss.detach(),
        "listwise_loss": listwise_loss.detach(),
    }
    if any(value.ndim != 0 or not torch.isfinite(value) for value in result.values()):
        raise ValueError("ranking loss produced a non-finite or non-scalar result.")
    return result


@dataclass
class VisualCandidateRanker:
    """Runtime for the learned visual mask ranker."""

    model: nn.Module
    image_size: int = 160
    device: str = "cuda"
    batch_size: int = 64
    checkpoint_path: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise ValueError("model must be a torch module.")
        self.image_size = _positive_int(self.image_size, field="image_size")
        self.batch_size = _positive_int(self.batch_size, field="batch_size")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string.")
        try:
            parsed_device = torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device is invalid.") from error
        if parsed_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if parsed_device.type not in {"cpu", "cuda"}:
            raise ValueError("device must select cpu or cuda.")
        if not isinstance(self.checkpoint_path, str):
            raise ValueError("checkpoint_path must be a string.")
        self.device = str(parsed_device)
        self.model.to(self.device)
        self.model.eval()

    def _score_model(
        self, batch: torch.Tensor, geometry: torch.Tensor, source_ids: torch.Tensor
    ) -> torch.Tensor:
        try:
            return self.model(batch, geometry=geometry, source_ids=source_ids)
        except TypeError:
            return self.model(batch)

    @torch.no_grad()
    def score_candidates_from_arrays(
        self,
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        candidates: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[int, float]]:
        image_rgb = _numeric_array(image_rgb, field="image_rgb", ndim=3)
        heatmap = _numeric_array(heatmap, field="heatmap", ndim=2)
        if image_rgb.shape[2] != 3 or heatmap.shape != image_rgb.shape[:2]:
            raise ValueError("image_rgb and heatmap shapes must be HxWx3 and HxW.")
        if isinstance(candidates, (str, bytes, Mapping)) or not isinstance(candidates, Sequence):
            raise ValueError("candidates must be a sequence of mappings.")
        tensors: List[torch.Tensor] = []
        geometries: List[torch.Tensor] = []
        source_ids: List[int] = []
        candidate_indices: List[int] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"candidate {index} must be a mapping.")
            raw_mask_path = candidate.get("raw_mask")
            if not isinstance(raw_mask_path, str) or not raw_mask_path:
                raise ValueError(f"candidate {index} must contain a raw_mask path.")
            if not Path(raw_mask_path).is_file():
                raise FileNotFoundError(raw_mask_path)
            raw_mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                raise ValueError(f"candidate mask is unreadable: {raw_mask_path}")
            tensors.append(
                build_visual_candidate_tensor_from_arrays(
                    image_rgb=image_rgb,
                    heatmap=heatmap,
                    raw_mask=raw_mask,
                    image_size=self.image_size,
                )
            )
            # Metadata is built from full-image coordinates so crop resizing does not erase location.
            geometries.append(
                torch.from_numpy(
                    build_candidate_geometry_features(candidate, raw_mask, image_rgb.shape)
                )
            )
            proposal_source = candidate.get("proposal_source", "")
            if not isinstance(proposal_source, str):
                raise ValueError("proposal_source must be a string.")
            source_ids.append(
                source_bucket_id(
                    proposal_source,
                    getattr(self.model, "source_vocab_size", 128),
                )
            )
            candidate_indices.append(index)
        if not tensors:
            return []

        scores: List[float] = []
        amp_enabled = str(self.device).startswith("cuda")
        for start in range(0, len(tensors), int(self.batch_size)):
            batch = torch.stack(tensors[start : start + int(self.batch_size)], dim=0).to(
                self.device
            )
            geometry = torch.stack(geometries[start : start + int(self.batch_size)], dim=0).to(
                self.device
            )
            source_batch = torch.tensor(
                source_ids[start : start + int(self.batch_size)],
                dtype=torch.long,
                device=self.device,
            )
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = self._score_model(batch, geometry, source_batch)
            expected = batch.shape[0]
            if not isinstance(logits, torch.Tensor) or logits.shape != (expected,):
                raise ValueError("model must return exactly one logit per candidate.")
            if not logits.is_floating_point() or not torch.isfinite(logits).all():
                raise ValueError("model logits must be finite floating values.")
            score_batch = torch.sigmoid(logits.detach().float()).cpu()
            if (
                score_batch.shape != (expected,)
                or not torch.isfinite(score_batch).all()
                or torch.any(score_batch < 0.0)
                or torch.any(score_batch > 1.0)
            ):
                raise ValueError("model scores must be finite values within [0, 1].")
            scores.extend(float(value) for value in score_batch.tolist())
        return list(zip(candidate_indices, scores))

    def select(
        self,
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        candidates: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any] | None:
        scored = self.score_candidates_from_arrays(
            image_rgb=image_rgb, heatmap=heatmap, candidates=candidates
        )
        if not scored:
            return None
        selected_index, selected_score = max(scored, key=lambda item: (item[1], -item[0]))
        candidate = candidates[selected_index]
        selected: Dict[str, Any] = {}
        for key in candidate.keys():
            if not isinstance(key, str):
                raise ValueError("candidate keys must be strings.")
            if _is_forbidden_eval_key(key):
                continue
            selected[key] = candidate[key]
        previous_value = selected.get("selection", {})
        if previous_value is None:
            previous_value = {}
        if not isinstance(previous_value, Mapping):
            raise ValueError("candidate selection provenance must be a mapping.")
        previous_selection = dict(previous_value)
        selected["visual_ranker_score"] = float(selected_score)
        selected["selection"] = {
            "selected_by": SELECTION_TAXONOMY,
            "selected_position": int(selected_index),
            "selection_score": round(float(selected_score), 6),
            "previous_selected_by": str(previous_selection.get("selected_by", "")),
            "previous_score": _finite_real(
                previous_selection.get("selection_score", selected.get("visual_mask_score", 0.0)),
                field="previous selection score",
            ),
            "checkpoint": self.checkpoint_path,
        }
        return selected


def _path_identity(path: Path) -> tuple[int, int, int, int]:
    try:
        record = path.stat(follow_symlinks=False)
    except OSError as error:
        raise FileNotFoundError(path) from error
    return (int(record.st_dev), int(record.st_ino), int(record.st_size), int(record.st_mtime_ns))


def _has_reparse_component(path: Path) -> bool:
    current = path
    while True:
        try:
            record = current.lstat()
        except OSError:
            return True
        attributes = int(getattr(record, "st_file_attributes", 0))
        if current.is_symlink() or attributes & int(
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            return True
        if current.parent == current:
            return False
        current = current.parent


def _assert_no_reparse(path: Path, *, boundary: str) -> None:
    if _has_reparse_component(path):
        raise ValueError(f"checkpoint path crossed a reparse point at {boundary}.")


def _read_checkpoint_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as file:
            before = os.fstat(file.fileno())
            if before.st_size <= 0 or before.st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint snapshot exceeds the allowed resource bound.")
            snapshot = file.read(MAX_CHECKPOINT_BYTES + 1)
            if len(snapshot) > MAX_CHECKPOINT_BYTES:
                raise ValueError("checkpoint snapshot exceeds the allowed resource bound.")
            after = os.fstat(file.fileno())
    except OSError as error:
        raise FileNotFoundError(path) from error
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(snapshot) != before.st_size:
        raise ValueError("checkpoint changed while its snapshot was being read.")
    if len(snapshot) <= 0 or len(snapshot) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint snapshot exceeds the allowed resource bound.")
    return snapshot


def _stream_checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_CHECKPOINT_BYTES:
                    raise ValueError("checkpoint stream exceeds the allowed resource bound.")
                digest.update(chunk)
    except OSError as error:
        raise FileNotFoundError(path) from error
    return digest.hexdigest()


def _validate_checkpoint_config(args: Any) -> dict[str, Any]:
    if not isinstance(args, dict) or frozenset(args) not in {
        LEGACY_MODEL_ARG_KEYS,
        PRODUCTION_LEGACY_ARG_KEYS,
    }:
        raise ValueError("checkpoint args do not match canonical5 or production legacy39 schema.")
    config = {
        "image_size": _positive_int(args["image_size"], field="checkpoint image_size"),
        "use_geometry_features": _strict_bool(
            args["use_geometry_features"], field="checkpoint use_geometry_features"
        ),
        "use_source_embedding": _strict_bool(
            args["use_source_embedding"], field="checkpoint use_source_embedding"
        ),
        "source_vocab_size": _positive_int(
            args["source_vocab_size"], field="checkpoint source_vocab_size", minimum=2
        ),
        "source_embedding_dim": _positive_int(
            args["source_embedding_dim"], field="checkpoint source_embedding_dim"
        ),
    }
    _validate_model_config_budget(config)
    return config


def load_visual_candidate_ranker(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    device: str = "cpu",
    image_size: int | None = None,
    batch_size: int = 64,
) -> VisualCandidateRanker:
    if isinstance(checkpoint_path, bool) or not isinstance(checkpoint_path, (str, Path)):
        raise ValueError("checkpoint_path must be a literal path.")
    path = Path(checkpoint_path)
    if not path.is_absolute():
        raise ValueError("checkpoint_path must be absolute.")
    _assert_no_reparse(path, boundary="entry")
    if not path.is_file():
        raise FileNotFoundError(path)
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be an exact lowercase SHA-256 digest.")
    batch_size = _positive_int(batch_size, field="batch_size")
    if image_size is not None:
        image_size = _positive_int(image_size, field="image_size")

    _assert_no_reparse(path, boundary="initial identity")
    identity_before = _path_identity(path)
    if identity_before[2] <= 0 or identity_before[2] > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint file exceeds the allowed resource bound.")
    _assert_no_reparse(path, boundary="snapshot read")
    snapshot = _read_checkpoint_bytes(path)
    _assert_no_reparse(path, boundary="snapshot completion")
    _assert_no_reparse(path, boundary="post-snapshot identity")
    identity_after_snapshot = _path_identity(path)
    if identity_after_snapshot != identity_before:
        raise ValueError("checkpoint identity changed during snapshot acquisition.")
    digest = hashlib.sha256(snapshot).hexdigest()
    if digest != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match expected_sha256.")
    try:
        checkpoint = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint snapshot could not be loaded safely.") from error
    if not isinstance(checkpoint, dict) or set(checkpoint) != LEGACY_CHECKPOINT_KEYS:
        raise ValueError("checkpoint does not match the explicit legacy schema.")
    _positive_int(checkpoint["epoch"], field="checkpoint epoch", minimum=0)
    if not isinstance(checkpoint["val"], dict):
        raise ValueError("checkpoint val must be a mapping.")
    config = _validate_checkpoint_config(checkpoint["args"])
    if image_size is not None and image_size != config["image_size"]:
        raise ValueError("requested image_size does not match the checkpoint config.")
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint model state must be a non-empty mapping.")
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint model state must contain tensors under string keys.")
        if value.layout != torch.strided or value.is_quantized:
            raise ValueError(
                "checkpoint model state tensors must use non-quantized strided layout."
            )
        if value.device.type != "cpu":
            raise ValueError("checkpoint model state tensors must be on CPU.")

    model = ConvNeXtCandidateRanker(
        image_size=config["image_size"],
        use_pretrained=False,
        use_geometry_features=config["use_geometry_features"],
        use_source_embedding=config["use_source_embedding"],
        source_vocab_size=config["source_vocab_size"],
        source_embedding_dim=config["source_embedding_dim"],
    )
    expected_state = model.state_dict()
    if set(state_dict) != set(expected_state):
        raise ValueError("checkpoint state keys do not exactly match the configured model.")
    total_state_bytes = 0
    for key, expected_tensor in expected_state.items():
        checkpoint_tensor = state_dict[key]
        if tuple(checkpoint_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(f"checkpoint tensor shape drifted for {key}.")
        if checkpoint_tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"checkpoint tensor dtype drifted for {key}.")
        if checkpoint_tensor.layout != expected_tensor.layout:
            raise ValueError(f"checkpoint tensor layout drifted for {key}.")
        if checkpoint_tensor.is_quantized != expected_tensor.is_quantized:
            raise ValueError(f"checkpoint tensor quantization drifted for {key}.")
        if expected_tensor.layout != torch.strided or expected_tensor.is_quantized:
            raise ValueError(f"configured model state is not tightly strided for {key}.")
        dense_bytes = int(expected_tensor.numel() * expected_tensor.element_size())
        if (
            expected_tensor.storage_offset() != 0
            or int(expected_tensor.untyped_storage().nbytes()) != dense_bytes
        ):
            raise ValueError(f"configured model state storage is not tight for {key}.")
        if (
            checkpoint_tensor.stride() != expected_tensor.stride()
            or checkpoint_tensor.storage_offset() != 0
            or int(checkpoint_tensor.untyped_storage().nbytes()) != dense_bytes
        ):
            raise ValueError(f"checkpoint tensor storage layout drifted for {key}.")
        total_state_bytes += dense_bytes
        if total_state_bytes > MAX_MODEL_STATE_BYTES:
            raise ValueError("checkpoint model state exceeds the allowed resource bound.")
        if (
            checkpoint_tensor.requires_grad != expected_tensor.requires_grad
            or checkpoint_tensor.is_conj() != expected_tensor.is_conj()
            or checkpoint_tensor.is_neg() != expected_tensor.is_neg()
        ):
            raise ValueError(f"checkpoint tensor properties drifted for {key}.")
        try:
            finite = bool(torch.isfinite(checkpoint_tensor).all())
        except (NotImplementedError, RuntimeError, TypeError) as error:
            raise ValueError(
                f"checkpoint tensor finiteness could not be validated for {key}."
            ) from error
        if not finite:
            raise ValueError(f"checkpoint tensor must be finite for {key}.")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint state does not exactly match the configured model.") from error
    ranker = VisualCandidateRanker(
        model=model,
        image_size=config["image_size"],
        device=device,
        batch_size=batch_size,
        checkpoint_path=str(path),
    )
    _assert_no_reparse(path, boundary="final streaming hash")
    return_digest = _stream_checkpoint_sha256(path)
    _assert_no_reparse(path, boundary="final hash completion")
    _assert_no_reparse(path, boundary="return identity")
    identity_before_return = _path_identity(path)
    if identity_before_return != identity_before or return_digest != expected_sha256:
        raise ValueError("checkpoint changed before the verified ranker could be returned.")
    _assert_no_reparse(path, boundary="return")
    return ranker
