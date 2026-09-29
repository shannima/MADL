from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


class PixelJudge:
    """
    Pixel Forensics Agent.

    It keeps the old ranking behaviour for compatibility, but now also exposes
    structured proposals, anomaly heatmap evidence, and a communication packet
    that can be consumed by semantic / decision agents.
    """

    def __init__(
        self,
        confirm_score_threshold: float = 0.72,
        weak_score_threshold: float = 0.58,
        min_mask_area_ratio: float = 0.0001,
        max_mask_area_ratio: float = 0.75,
        anomaly_activation_threshold: float = 0.42,
        proposal_top_k: int = 3,
    ) -> None:
        self.confirm_score_threshold = confirm_score_threshold
        self.weak_score_threshold = weak_score_threshold
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_mask_area_ratio = max_mask_area_ratio
        self.anomaly_activation_threshold = anomaly_activation_threshold
        self.proposal_top_k = proposal_top_k

    def build_empty_summary(
        self,
        image_shape: tuple[int, int],
        report: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        height, width = image_shape
        anomaly_heatmap = np.zeros((height, width), dtype=np.float32)
        summary = {
            "is_tampered": False,
            "confirmed_masks": [],
            "selected_masks": [],
            "localization_masks": [],
            "candidate_groups": [],
            "reasoning": "Pixel stage skipped because no localized proposal was available.",
            "anomaly_heatmap": anomaly_heatmap,
            "anomaly_heatmap_stats": self._heatmap_stats(anomaly_heatmap),
            "suspicious_region_proposals": [],
            "low_level_confidence": 0.0,
            "consensus_state": "no_local_evidence",
            "agent_packet": {
                "agent_role": "PixelForensicsAgent",
                "protocol_version": "pixel_v2",
                "bottom_up_score": 0.0,
                "proposal_count": 0,
                "consensus_state": "no_local_evidence",
                "message": "No localized pixel proposal reached the analysis stage.",
            },
        }
        if report is not None:
            summary["agent_packet"]["agent_a_vote"] = bool(report.get("is_tampered", False))
        return summary

    def _score_candidate(
        self, candidate: Dict[str, Any], image_shape: tuple[int, int]
    ) -> Dict[str, Any]:
        height, width = image_shape
        image_area = max(1, height * width)
        mask = candidate["mask"]
        mask_area = int(mask.sum())
        mask_area_ratio = mask_area / image_area
        width_box = max(1, int(candidate["box_pixels"][2] - candidate["box_pixels"][0]))
        height_box = max(1, int(candidate["box_pixels"][3] - candidate["box_pixels"][1]))
        box_area = width_box * height_box
        box_fill_ratio = min(mask_area / max(1, box_area), 1.0)
        box_area_ratio = min(box_area / image_area, 1.0)

        edge_pixels = int(
            mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
        )
        edge_contact_ratio = min(edge_pixels / max(1, mask_area), 1.0) if mask_area > 0 else 0.0

        adjusted_score = float(candidate["score"])
        if mask_area_ratio < self.min_mask_area_ratio:
            adjusted_score *= 0.75
        if mask_area_ratio > self.max_mask_area_ratio:
            adjusted_score *= 0.80
        if box_fill_ratio < 0.01:
            adjusted_score *= 0.85
        # A mask that almost exactly fills the prompt box is often a SAM box
        # fallback rather than evidence of the manipulated boundary itself.
        if box_fill_ratio >= 0.98:
            adjusted_score *= 0.72
        elif box_fill_ratio >= 0.92:
            adjusted_score *= 0.84
        if edge_contact_ratio > 0.50:
            adjusted_score *= 0.88
        if 0.001 <= mask_area_ratio <= 0.08:
            adjusted_score *= 1.03

        scored = dict(candidate)
        scored["mask_area"] = mask_area
        scored["mask_area_ratio"] = mask_area_ratio
        scored["box_fill_ratio"] = box_fill_ratio
        scored["edge_contact_ratio"] = edge_contact_ratio
        scored["tight_bbox_ratio"] = box_area_ratio
        scored["adjusted_score"] = adjusted_score
        return scored

    def _normalize_score_map(
        self, candidate: Dict[str, Any], image_shape: tuple[int, int]
    ) -> np.ndarray:
        height, width = image_shape
        score_map = candidate.get("score_map")
        if score_map is None:
            return candidate["mask"].astype(np.float32)

        arr = np.asarray(score_map, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return candidate["mask"].astype(np.float32)
        if arr.shape != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        if float(arr.max()) > 1.0 or float(arr.min()) < 0.0:
            arr = np.clip(arr, -12.0, 12.0)
            arr = 1.0 / (1.0 + np.exp(-arr))
        return np.clip(arr.astype(np.float32), 0.0, 1.0)

    def _heatmap_stats(self, anomaly_heatmap: np.ndarray) -> Dict[str, float]:
        active = anomaly_heatmap >= self.anomaly_activation_threshold
        return {
            "max_value": float(np.max(anomaly_heatmap)) if anomaly_heatmap.size else 0.0,
            "mean_value": float(np.mean(anomaly_heatmap)) if anomaly_heatmap.size else 0.0,
            "active_ratio": float(active.mean()) if anomaly_heatmap.size else 0.0,
        }

    def _build_anomaly_heatmap(
        self,
        image_shape: tuple[int, int],
        scored_groups: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, Dict[str, float]]:
        height, width = image_shape
        anomaly_heatmap = np.zeros((height, width), dtype=np.float32)
        for group in scored_groups:
            for candidate in group.get("candidates", [])[:2]:
                norm_map = self._normalize_score_map(candidate, image_shape)
                weighted = norm_map * max(0.15, float(candidate["adjusted_score"]))
                anomaly_heatmap = np.maximum(anomaly_heatmap, weighted)
        anomaly_heatmap = np.clip(anomaly_heatmap, 0.0, 1.0)
        return anomaly_heatmap, self._heatmap_stats(anomaly_heatmap)

    def _box_pixels_to_permille(self, box_pixels: Any, image_shape: tuple[int, int]) -> List[int]:
        height, width = image_shape
        x0, y0, x1, y1 = [int(v) for v in box_pixels]
        return [
            max(0, min(1000, int(round((x0 / max(1, width)) * 1000.0)))),
            max(0, min(1000, int(round((y0 / max(1, height)) * 1000.0)))),
            max(0, min(1000, int(round((x1 / max(1, width)) * 1000.0)))),
            max(0, min(1000, int(round((y1 / max(1, height)) * 1000.0)))),
        ]

    def _build_region_proposals(
        self,
        image_shape: tuple[int, int],
        scored_groups: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        for group_idx, group in enumerate(scored_groups, start=1):
            candidate = group.get("top_candidate")
            if not candidate:
                continue
            proposals.append(
                {
                    "proposal_id": f"pixel_region_{group_idx}",
                    "description": group.get("description", "Unknown Region"),
                    "source_box": group.get("source_box"),
                    "proposal_box_pixels": [int(v) for v in candidate["box_pixels"]],
                    "proposal_box_permille": self._box_pixels_to_permille(
                        candidate["box_pixels"], image_shape
                    ),
                    "confidence": round(float(candidate["adjusted_score"]), 4),
                    "sam_score": round(float(candidate["sam_score"]), 4),
                    "mask_area_ratio": round(float(candidate["mask_area_ratio"]), 6),
                    "box_fill_ratio": round(float(candidate["box_fill_ratio"]), 6),
                    "edge_contact_ratio": round(float(candidate.get("edge_contact_ratio", 0.0)), 6),
                    "tight_bbox_ratio": round(float(candidate.get("tight_bbox_ratio", 0.0)), 6),
                    "evidence_type": "localized_edit",
                }
            )
        return proposals[: self.proposal_top_k]

    def _compute_low_level_confidence(
        self,
        scored_groups: List[Dict[str, Any]],
        confirmed_masks: List[Dict[str, Any]],
        selected_masks: List[Dict[str, Any]],
    ) -> float:
        if not scored_groups:
            return 0.0
        top_scores = [
            float(group["top_candidate"]["adjusted_score"]) for group in scored_groups[:3]
        ]
        top_mean = sum(top_scores) / max(1, len(top_scores))
        if confirmed_masks:
            return min(1.0, top_mean + 0.08)
        if selected_masks:
            return min(1.0, top_mean)
        return min(1.0, top_mean * 0.65)

    def judge(
        self,
        image_shape: tuple[int, int],
        regions: List[Dict[str, Any]],
        candidate_groups: List[Dict[str, Any]],
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not candidate_groups:
            return self.build_empty_summary(image_shape, report)

        scored_groups: List[Dict[str, Any]] = []
        for group in candidate_groups:
            scored_candidates = [
                self._score_candidate(candidate, image_shape)
                for candidate in group.get("candidates", [])
            ]
            scored_candidates.sort(key=lambda item: item["adjusted_score"], reverse=True)
            if not scored_candidates:
                continue
            top_candidate = scored_candidates[0]
            scored_groups.append(
                {
                    "description": group.get("description", "Unknown Region"),
                    "source_box": group.get("source_box"),
                    "region_id": group.get("region_id", ""),
                    "proposal_source": group.get("proposal_source", ""),
                    "proposal_confidence": group.get("proposal_confidence", 0.0),
                    "evidence_type": group.get("evidence_type", ""),
                    "evidence_scope": group.get("evidence_scope", ""),
                    "top_candidate": top_candidate,
                    "candidates": scored_candidates,
                }
            )

        if not scored_groups:
            return self.build_empty_summary(image_shape, report)

        scored_groups.sort(key=lambda item: item["top_candidate"]["adjusted_score"], reverse=True)
        selected_masks = []
        localization_masks = []
        for group in scored_groups:
            candidate = group["top_candidate"]
            if candidate["adjusted_score"] >= self.weak_score_threshold:
                selected_masks.append(
                    {
                        "description": group["description"],
                        "region_id": group.get("region_id", ""),
                        "source_box": group.get("source_box"),
                        "proposal_source": group.get("proposal_source", ""),
                        "proposal_confidence": group.get("proposal_confidence", 0.0),
                        "evidence_type": group.get("evidence_type", ""),
                        "evidence_scope": group.get("evidence_scope", ""),
                        "confidence": candidate["adjusted_score"],
                        "sam_score": candidate["sam_score"],
                        "mask": candidate["mask"],
                        "score_map": candidate.get("score_map"),
                        "mask_area": candidate["mask_area"],
                        "mask_area_ratio": candidate["mask_area_ratio"],
                        "box_fill_ratio": candidate["box_fill_ratio"],
                        "edge_contact_ratio": candidate.get("edge_contact_ratio", 0.0),
                        "tight_bbox_ratio": candidate.get("tight_bbox_ratio", 0.0),
                        "box_pixels": candidate["box_pixels"],
                        "box_scale": candidate["box_scale"],
                        "candidate_index": candidate["candidate_index"],
                        "source": "pixel_judge",
                    }
                )
            for loc_candidate in group["candidates"][:2]:
                if loc_candidate["adjusted_score"] < self.weak_score_threshold:
                    continue
                localization_masks.append(
                    {
                        "description": group["description"],
                        "region_id": group.get("region_id", ""),
                        "source_box": group.get("source_box"),
                        "proposal_source": group.get("proposal_source", ""),
                        "proposal_confidence": group.get("proposal_confidence", 0.0),
                        "evidence_type": group.get("evidence_type", ""),
                        "evidence_scope": group.get("evidence_scope", ""),
                        "confidence": loc_candidate["adjusted_score"],
                        "sam_score": loc_candidate["sam_score"],
                        "mask": loc_candidate["mask"],
                        "score_map": loc_candidate.get("score_map"),
                        "mask_area": loc_candidate["mask_area"],
                        "mask_area_ratio": loc_candidate["mask_area_ratio"],
                        "box_fill_ratio": loc_candidate["box_fill_ratio"],
                        "edge_contact_ratio": loc_candidate.get("edge_contact_ratio", 0.0),
                        "tight_bbox_ratio": loc_candidate.get("tight_bbox_ratio", 0.0),
                        "box_pixels": loc_candidate["box_pixels"],
                        "box_scale": loc_candidate["box_scale"],
                        "candidate_index": loc_candidate["candidate_index"],
                        "source": "pixel_judge",
                    }
                )

        confirmed_masks = [
            item
            for item in selected_masks
            if float(item["confidence"]) >= self.confirm_score_threshold
        ]
        semantic_vote = bool(report.get("is_tampered", False))
        final_vote = bool(confirmed_masks) or (semantic_vote and bool(selected_masks))

        anomaly_heatmap, anomaly_heatmap_stats = self._build_anomaly_heatmap(
            image_shape, scored_groups
        )
        proposals = self._build_region_proposals(image_shape, scored_groups)
        low_level_confidence = self._compute_low_level_confidence(
            scored_groups, confirmed_masks, selected_masks
        )

        consensus_state = "confirmed_local_tampering" if confirmed_masks else "weak_local_evidence"
        if not selected_masks:
            consensus_state = "no_pixel_confirmation"

        reasoning = []
        if confirmed_masks:
            reasoning.append(
                f"{len(confirmed_masks)} candidate mask(s) passed the confirm threshold."
            )
        elif selected_masks:
            reasoning.append("Only weak-but-plausible masks were found; kept as soft evidence.")
        else:
            reasoning.append("No mask candidate passed the minimum pixel evidence threshold.")
        if proposals:
            reasoning.append(
                f"Structured pixel proposals prepared for {len(proposals)} suspicious region(s)."
            )

        return {
            "is_tampered": final_vote,
            "confirmed_masks": confirmed_masks,
            "selected_masks": selected_masks,
            "localization_masks": localization_masks,
            "candidate_groups": scored_groups,
            "reasoning": " ".join(reasoning),
            "anomaly_heatmap": anomaly_heatmap,
            "anomaly_heatmap_stats": anomaly_heatmap_stats,
            "suspicious_region_proposals": proposals,
            "low_level_confidence": round(float(low_level_confidence), 6),
            "consensus_state": consensus_state,
            "agent_packet": {
                "agent_role": "PixelForensicsAgent",
                "protocol_version": "pixel_v2",
                "bottom_up_score": round(float(low_level_confidence), 6),
                "proposal_count": len(proposals),
                "consensus_state": consensus_state,
                "message": " ".join(reasoning),
                "anomaly_heatmap_stats": anomaly_heatmap_stats,
            },
        }
