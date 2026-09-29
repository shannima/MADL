"""Create a sanitized Hugging Face weight package from trusted training outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(args) -> None:
    import torch

    output = Path(args.output)
    adapter_output = output / "agent_a_qwen_lora"
    dual_output = output / "agent_b_dualstream"
    ranker_output = output / "agent_b_visual_ranker"
    for directory in (adapter_output, dual_output, ranker_output):
        directory.mkdir(parents=True, exist_ok=True)

    adapter_source = Path(args.agent_a_adapter)
    for name in ADAPTER_FILES:
        source = adapter_source / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, adapter_output / name)
    adapter_config_path = adapter_output / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = args.base_model
    adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")

    dual = torch.load(args.agent_b_dualstream, map_location="cpu", weights_only=False)
    dual_args = dict(dual.get("args", {}))
    dual_release = {
        "format_version": "madl.checkpoint.v1",
        "component": "agent_b_dualstream",
        "architecture": {
            "backbone_name": str(dual_args.get("backbone", "convnext_tiny")),
            "image_size": int(dual_args.get("image_size", 512)),
            "fusion_dim": 256,
            "num_classes": 3,
            "proposal_threshold": 0.45,
            "proposal_top_k": 5,
        },
        "class_names": list(dual.get("class_names", ["real", "synthetic", "tampered"])),
        "state_dict": dual["model"],
        "training_summary": {
            "epoch": int(dual.get("epoch", 0)),
            "best_score": float(dual.get("best_score", 0.0)),
            "seed": int(dual_args.get("seed", 42)),
            "loss_weights": {
                "classification": float(dual_args.get("cls_loss_weight", 0.2)),
                "bce": float(dual_args.get("bce_loss_weight", 2.0)),
                "dice": float(dual_args.get("dice_loss_weight", 4.0)),
            },
        },
    }
    torch.save(dual_release, dual_output / "madl_agent_b_dualstream_v1.pt")

    ranker = torch.load(args.agent_b_visual_ranker, map_location="cpu", weights_only=False)
    ranker_args = dict(ranker["args"])
    ranker_release = {
        "epoch": int(ranker.get("epoch", 0)),
        "model": ranker["model"],
        "args": {
            "image_size": int(ranker_args.get("image_size", 160)),
            "use_geometry_features": bool(ranker_args.get("use_geometry_features", True)),
            "use_source_embedding": bool(ranker_args.get("use_source_embedding", False)),
            "source_vocab_size": int(ranker_args.get("source_vocab_size", 128)),
            "source_embedding_dim": int(ranker_args.get("source_embedding_dim", 8)),
        },
        "val": dict(ranker.get("val", {})),
    }
    torch.save(ranker_release, ranker_output / "madl_agent_b_visual_ranker_v1.pt")

    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            records.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "format_version": "madl.weights.v1",
        "base_model": args.base_model,
        "external_weights_not_included": ["SAM ViT-H (sam_vit_h_4b8939.pth)"],
        "files": records,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export trusted MADL training checkpoints without optimizer state or local paths."
    )
    parser.add_argument("--agent-a-adapter", required=True)
    parser.add_argument("--agent-b-dualstream", required=True)
    parser.add_argument("--agent-b-visual-ranker", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
