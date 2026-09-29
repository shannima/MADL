from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def json_safe(value: Any) -> Any:
    """Convert legacy module outputs into compact JSON-safe objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric evidence is not JSON-safe")
        return value
    if isinstance(value, complex):
        raise TypeError("complex evidence is not JSON-safe")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    array_protocol = hasattr(value, "__array_interface__") or callable(
        getattr(value, "__dlpack__", None)
    )
    if array_protocol and hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = [int(dim) for dim in getattr(value, "shape", [])]
        if shape:
            return {"array_shape": shape, "dtype": str(getattr(value, "dtype", ""))}
        item = getattr(value, "item", None)
        if callable(item):
            return json_safe(item())
        raise TypeError("zero-dimensional array evidence has no scalar item")
    raise TypeError(f"{type(value).__name__} is not JSON-safe evidence")


@dataclass
class CandidateRegion:
    region_id: str
    box_2d: List[int]
    confidence: float = 0.0
    description: str = "suspicious_region"
    evidence_type: str = "localized_edit"
    evidence_scope: str = "local"
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class PixelProposalPacket:
    agent_role: str = "PixelProposalAgent"
    protocol_version: str = "pixel_proposal_v1"
    source_mode: str = "cached_proposal"
    candidate_regions: List[CandidateRegion] = field(default_factory=list)
    bottom_up_score: float = 0.0
    anomaly_heatmap: Any = ""
    class_probs: Dict[str, float] = field(default_factory=dict)
    cached_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        heatmap = json_safe(self.anomaly_heatmap)
        return {
            "agent_role": json_safe(self.agent_role),
            "protocol_version": json_safe(self.protocol_version),
            "source_mode": json_safe(self.source_mode),
            "candidate_regions": [region.to_dict() for region in self.candidate_regions],
            "bottom_up_score": json_safe(self.bottom_up_score),
            "anomaly_heatmap": heatmap,
            "class_probs": json_safe(self.class_probs),
            "cached_report": json_safe(self.cached_report),
        }


@dataclass
class LMMGroundingPacket:
    agent_role: str = "LMMGroundingAgent"
    protocol_version: str = "lmm_grounding_v1"
    backbone: str = "sida"
    source_mode: str = "lmm_hidden_grounding"
    predicted_label: str = "real"
    confidence: float = 0.0
    class_probs: Dict[str, float] = field(default_factory=dict)
    raw_mask: str = ""
    raw_score_map: str = ""
    mask_overlay: str = ""
    grounding_source: str = "seg_token_mask"
    textual_trace: str = ""
    raw_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class MaskCandidatePacket:
    agent_role: str = "MaskAgent"
    protocol_version: str = "mask_v1"
    region_id: str = ""
    description: str = ""
    source_box: List[int] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_extractions: List[Dict[str, Any]] = field(default_factory=list)
    final_extraction: Optional[Dict[str, Any]] = None
    rerank_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class SemanticPacket:
    agent_role: str = "SemanticConsistencyAgent"
    protocol_version: str = "semantic_v1"
    backbone: str = "qwen"
    enabled: bool = True
    region_traces: List[Dict[str, Any]] = field(default_factory=list)
    top_down_score: float = 0.0
    local_semantic_score: float = 0.0
    textual_rationale: str = ""
    recommended_label: str = "real"
    raw_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class SyntheticPacket:
    agent_role: str = "SyntheticRecognitionAgent"
    protocol_version: str = "synthetic_v1"
    backbone: str = "qwen"
    enabled: bool = True
    global_synthetic_score: float = 0.0
    recommended_label: str = "real"
    generation_cues: List[str] = field(default_factory=list)
    textual_rationale: str = ""
    raw_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class DecisionPacket:
    agent_role: str = "DecisionAgent"
    protocol_version: str = "decision_v1"
    final_label: str = "real"
    is_tampered: bool = False
    is_synthetic: bool = False
    abstain: bool = False
    confidence: float = 0.0
    selected_region_id: str = ""
    selected_mask: str = ""
    state_path: List[str] = field(default_factory=list)
    conflict_detected: bool = False
    raw_decision: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class MADLReport:
    image_path: str
    final_label: str
    pixel_packet: PixelProposalPacket
    mask_packet: MaskCandidatePacket
    semantic_packet: SemanticPacket
    synthetic_packet: SyntheticPacket
    decision_packet: DecisionPacket
    final_extraction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "final_label": self.final_label,
            "pixel_packet": self.pixel_packet.to_dict(),
            "mask_packet": self.mask_packet.to_dict(),
            "semantic_packet": self.semantic_packet.to_dict(),
            "synthetic_packet": self.synthetic_packet.to_dict(),
            "decision_packet": self.decision_packet.to_dict(),
            "final_extraction": json_safe(self.final_extraction),
        }
