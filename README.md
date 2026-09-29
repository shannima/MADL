# MADL

[GitHub code](https://github.com/shannima/MADL) | [Hugging Face weights](https://huggingface.co/moindy/MADL)

Official release package for **MADL: Towards Dependable Image Forgery
Detection via Multi-Agent Forensic Reasoning**.

MADL treats image forensics as evidence-driven collaboration rather than a
single prediction head. It returns one of three labels—`real`, `synthetic`, or
`tampered`—and emits a localization mask only when the final result is locally
tampered.

```text
image
  ├─ Agent A: SFT-adapted Qwen semantic verification + three-class evidence
  ├─ Agent B: dual-stream heatmap + candidate pool + SAM + learned mask ranker
  └─ Agent C: agreement / suppression / override / conflict adjudication
                         ↓
      label + conditional mask + structured evidence trace
```

The model weights are intentionally stored outside Git in the companion
[Hugging Face release](https://huggingface.co/moindy/MADL). See the
[weight layout](#weight-layout) below and the model card for download details.

## Installation

Python 3.10 and an NVIDIA CUDA environment are recommended.

```bash
conda create -n madl python=3.10 -y
conda activate madl
pip install -e .[inference,evaluation]
```

Alternatively, install `requirements.txt` in an existing PyTorch environment.
The Qwen and SAM base weights follow their upstream download instructions.

## Weight layout

Download the MADL Hugging Face weight package into `models/`, then place the
official SAM ViT-H checkpoint under `models/external/`:

```text
models/
├── agent_a_qwen_lora/
├── agent_b_dualstream/madl_agent_b_dualstream_v1.pt
├── agent_b_visual_ranker/madl_agent_b_visual_ranker_v1.pt
└── external/sam_vit_h_4b8939.pth
```

Verify all released files:

```bash
python scripts/verify_weights.py models
```

## Inference

```bash
madl-infer \
  --image /path/to/image.png \
  --weights /path/to/models \
  --output-dir outputs/example \
  --device cuda
```

Outputs:

- `<stem>_madl.json`: final label, confidence, Agent C state, and evidence trace;
- `<stem>_madl_mask.png`: emitted only for a locally tampered decision.

The model-backed pipeline can also be constructed in Python:

```python
from dataclasses import replace
from madl.config import MADLConfig
from madl.factory import build_pipeline

config = replace(MADLConfig(), weight_root="/path/to/models")
pipeline = build_pipeline(config, device="cuda")
result = pipeline.predict("image.png")
print(result.label, result.decision_state)
```

## Training and evaluation

| Task | Entry point |
|---|---|
| Build Agent A SFT records | `python -m madl.training.prepare_qwen_sft_data` |
| Agent A LoRA recipe | `configs/agent_a_lora_sft.yaml` |
| Train dual-stream Agent B | `python -m madl.training.train_dualstream` |
| Train visual mask ranker | `python -m madl.training.train_visual_ranker` |
| Evaluate saved predictions | `python scripts/evaluate_predictions.py` |
| Prepare robustness inputs (multithreaded) | `python scripts/prepare_robustness.py` |
| Run semantic linear probe | `python scripts/run_semantic_probe.py` |
| Audit Agent C paths | `python scripts/audit_decision_paths.py` |

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

The lightweight test suite does not download model weights. Model-release
verification additionally checks the sanitized dual-stream checkpoint, the
strict visual-ranker schema, and SHA-256 integrity.

## Repository policy

- No `.env`, API key, machine-local absolute path, model binary, dataset, cache,
  or failed experiment is tracked in the GitHub repository.
- The public core uses the paper's Agent A/B/C taxonomy. Obsolete SIDACls and
  five-module draft terminology are not part of the runtime.
- SIDA is retained only as an external comparison baseline; its source and
  weights are not copied into this repository.
- Natural-language output is described as a structured evidence trace, not as
  an independently validated explanation-quality score.

## License

Original MADL release code is provided under Apache-2.0. Third-party models,
datasets, and libraries retain their own licenses and usage terms.

### Third-party components

MADL composes original release code with separately distributed upstream
models and datasets. This repository does not relicense those projects.

| Component | Use in MADL | Upstream | License / terms |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | Agent A base VLM | `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 |
| PEFT | LoRA adapter loading | `huggingface/peft` | Apache-2.0 |
| Segment Anything (SAM) | Agent B candidate masks | `facebookresearch/segment-anything` | Apache-2.0 |
| Torch / TorchVision | Training and inference | `pytorch/pytorch`, `pytorch/vision` | BSD-style upstream licenses |
| Transformers | Qwen loading and generation | `huggingface/transformers` | Apache-2.0 |
| OpenCV | Image processing | `opencv/opencv` | Apache-2.0 |
| SID-Set | Training/evaluation data | `Huang-yating/SID_Set` | Dataset page lists CC BY 4.0 |
| SIDA-7B | Comparison baseline only | `Huang-yating/SIDA-7B` | Model page lists Llama 2 terms |

SIDA source code and SIDA model weights are **not** copied into the MADL code
repository or the MADL weight package. Results for external comparison methods
remain attributed to their respective papers and releases.
