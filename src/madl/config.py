"""Portable runtime configuration for the public MADL release."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class MADLConfig:
    """References weights by repository-relative names, never machine-local paths."""

    weight_root: str = "models"
    agent_a_adapter: str = "agent_a_qwen_lora"
    agent_b_checkpoint: str = "agent_b_dualstream/madl_agent_b_dualstream_v1.pt"
    visual_ranker_checkpoint: str = "agent_b_visual_ranker/madl_agent_b_visual_ranker_v1.pt"
    sam_checkpoint: str = "external/sam_vit_h_4b8939.pth"
    extra: Mapping[str, object] = field(default_factory=dict)

    def weight_references(self) -> dict[str, str]:
        return {
            "agent_a": self.agent_a_adapter,
            "agent_b": self.agent_b_checkpoint,
            "visual_ranker": self.visual_ranker_checkpoint,
            "sam": self.sam_checkpoint,
        }

    def validate_portable(self) -> None:
        prohibited = ("/mnt/", "/root/", "\\\\", ":\\")
        for name, value in self.weight_references().items():
            normalized = str(value).replace("\\", "/")
            if Path(value).is_absolute() or any(token in str(value) for token in prohibited):
                raise ValueError(f"{name} must be a portable relative path: {value}")
            if ".." in PurePosixPath(normalized).parts:
                raise ValueError(f"{name} cannot escape the configured root: {value}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MADLConfig":
        import yaml

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        weights = dict(payload.pop("weights", {}) or {})
        mapping = {
            "agent_a": "agent_a_adapter",
            "agent_b": "agent_b_checkpoint",
            "visual_ranker": "visual_ranker_checkpoint",
            "sam": "sam_checkpoint",
        }
        for key, value in weights.items():
            if key not in mapping:
                raise ValueError(f"unknown weight key in {config_path}: {key}")
            payload[mapping[key]] = value
        if "weight_root" not in payload and os.environ.get("MADL_WEIGHT_ROOT"):
            payload["weight_root"] = os.environ["MADL_WEIGHT_ROOT"]
        config = cls(**payload)
        config.validate_portable()
        return config

    def resolve_weight(self, name: str) -> Path:
        self.validate_portable()
        try:
            relative = self.weight_references()[name]
        except KeyError as exc:
            raise KeyError(f"unknown MADL weight reference: {name}") from exc
        return Path(self.weight_root) / Path(relative)
