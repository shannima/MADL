import json
import os
import re
import tempfile

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.utils import logging as hf_logging

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")


DEFAULT_FORENSIC_SYSTEM_PROMPT = (
    "You are a forensic image analyst. Inspect the image for signs of tampering, splicing, or AI generation. "
    "Explain suspicious evidence in chain_of_thought and output all suspicious regions in evidence_regions. "
    "Use evidence_regions.box_2d in permille coordinates formatted as [x_min, y_min, x_max, y_max]. "
    "Return strict JSON."
)

DEFAULT_FORENSIC_USER_PROMPT = "Please inspect the image and return a strict JSON report."
HIGH_RECALL_REGION_SYSTEM_PROMPT = (
    "You are a high-recall forensic scout for localized image tampering. "
    "Your goal is to propose plausible suspicious regions for downstream verification. "
    "Be more sensitive to subtle local edits, splicing boundaries, pasted objects, erased details, or texture inconsistencies. "
    "Return strict JSON with keys is_tampered, chain_of_thought, and evidence_regions. "
    "Use evidence_regions.box_2d in permille coordinates formatted as [x_min, y_min, x_max, y_max]. "
    "Prefer up to 3 localized candidate regions. If the image looks genuinely clean, return is_tampered=false and an empty evidence_regions list."
)
HIGH_RECALL_REGION_USER_PROMPT = (
    "Identify localized suspicious regions that may contain tampering, even if the cues are subtle. "
    "Return strict JSON only."
)
DEFAULT_IMAGE_MAX_PIXELS = 262144
DEFAULT_MAX_NEW_TOKENS = 384
LOW_VRAM_RETRY_PROFILES = [
    (262144, 384),
    (196608, 256),
    (131072, 192),
]


def resolve_model_and_adapter_paths(model_path: str) -> tuple[str, str | None]:
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        return model_path, None
    with open(adapter_config, "r", encoding="utf-8") as file:
        config = json.load(file)
    base_model_path = str(config.get("base_model_name_or_path", "")).strip()
    if not base_model_path:
        raise ValueError(
            f"Adapter config does not define base_model_name_or_path: {adapter_config}"
        )
    return base_model_path, model_path


def _best_effort_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


class VisualDetectiveAgent:
    def __init__(
        self,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        small_object_second_pass: bool = True,
    ):
        base_model_path, adapter_path = resolve_model_and_adapter_paths(model_path)
        if adapter_path:
            print(
                f"[Agent A] Loading Forensic Base Model from {base_model_path} with adapter {adapter_path}..."
            )
        else:
            print(f"[Agent A] Loading Forensic Model from {model_path}...")
        hf_logging.set_verbosity_error()
        try:
            hf_logging.disable_progress_bar()
        except AttributeError:
            pass

        processor_path = (
            adapter_path
            if adapter_path and os.path.exists(os.path.join(adapter_path, "processor_config.json"))
            else base_model_path
        )
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = None
        self.runtime_device = "cpu"
        self.small_object_second_pass = small_object_second_pass
        self._load_model(base_model_path, adapter_path=adapter_path)
        self.model.eval()

        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("[Agent A] Model loaded successfully.")

    def _attach_adapter_if_needed(self, adapter_path: str | None):
        if adapter_path is None:
            return
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def _load_model(self, model_path: str, adapter_path: str | None = None):
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self._attach_adapter_if_needed(adapter_path)
            self.runtime_device = "cpu"
            self.model.to("cuda")
            self.runtime_device = "cuda"
        except torch.OutOfMemoryError:
            print(
                "[Agent A] CPU->CUDA load hit OOM, retrying with device_map='cuda'...", flush=True
            )
            try:
                if self.model is not None:
                    del self.model
            except Exception:
                pass
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self._attach_adapter_if_needed(adapter_path)
            self.runtime_device = "cuda"

    def move_model_to(self, device: str):
        if self.runtime_device == device:
            return
        print(f"[Agent A] Moving Qwen model to {device}...")
        self.model.to(device)
        self.runtime_device = device
        if device == "cpu":
            import gc

            gc.collect()
            torch.cuda.empty_cache()

    def _run_inference(
        self,
        image_source: str,
        user_prompt: str = DEFAULT_FORENSIC_USER_PROMPT,
        system_instruction: str | None = None,
    ) -> dict:
        if system_instruction is None:
            system_instruction = DEFAULT_FORENSIC_SYSTEM_PROMPT

        last_oom_error = None
        for max_pixels, max_new_tokens in LOW_VRAM_RETRY_PROFILES:
            text = None
            image_inputs = None
            video_inputs = None
            inputs = None
            generated_ids = None
            generated_ids_trimmed = None
            messages = [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_source}",
                            "max_pixels": max_pixels,
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            if self.runtime_device == "cuda":
                torch.cuda.empty_cache()

            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.runtime_device)

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                return self._parse_json_output(output_text)
            except torch.OutOfMemoryError as exc:
                last_oom_error = exc
                if self.runtime_device == "cuda":
                    torch.cuda.empty_cache()
                continue
            finally:
                del generated_ids, generated_ids_trimmed, inputs, image_inputs, video_inputs, text
                if self.runtime_device == "cuda":
                    torch.cuda.empty_cache()

        if last_oom_error is not None:
            raise last_oom_error
        raise RuntimeError("Inference failed without a captured CUDA OOM.")

    def analyze_image(
        self,
        image_path: str,
        user_prompt: str = DEFAULT_FORENSIC_USER_PROMPT,
        system_instruction: str | None = None,
    ) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        print(f"\nAnalyzing image {image_path}...")
        return self._run_inference(image_path, user_prompt, system_instruction=system_instruction)

    def _run_crop_pass(
        self, img: Image.Image, crop_specs: list[dict], stage_name: str
    ) -> dict | None:
        positive_reports = []
        for crop_idx, crop_spec in enumerate(crop_specs):
            x1, y1, x2, y2 = crop_spec["bounds"]
            crop = img.crop((x1, y1, x2, y2))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            crop.save(tmp_path)

            crop_prompt = (
                f"This is a crop from the original image: {crop_spec['label']}. "
                "Inspect this local region carefully for subtle manipulation, compositing, or abnormal boundaries "
                "and return a strict JSON report."
            )
            try:
                crop_report = self._run_inference(tmp_path, user_prompt=crop_prompt)
                if crop_report.get("is_tampered", False):
                    crop_report["_multiscale_source"] = (
                        f"{stage_name}_{crop_idx}_{crop_spec['label']}"
                    )
                    crop_report["_multiscale_crop_bounds"] = [int(x1), int(y1), int(x2), int(y2)]
                    crop_report["_multiscale_stage"] = stage_name
                    crop_report["chain_of_thought"] = (
                        f"[Multi-scale analysis] The full image looked clean, but tampering cues appeared in the "
                        f"{crop_spec['label']} crop. " + crop_report.get("chain_of_thought", "")
                    )
                    positive_reports.append((self._score_crop_report(crop_report), crop_report))
            except Exception:
                continue
            finally:
                _best_effort_remove(tmp_path)
        if positive_reports:
            positive_reports.sort(key=lambda item: item[0], reverse=True)
            return positive_reports[0][1]
        return None

    def _build_quadrant_specs(self, width: int, height: int) -> list[dict]:
        overlap_x = int(width * 0.1)
        overlap_y = int(height * 0.1)
        mid_x = width // 2
        mid_y = height // 2
        quadrants = [
            (0, 0, mid_x + overlap_x, mid_y + overlap_y),
            (mid_x - overlap_x, 0, width, mid_y + overlap_y),
            (0, mid_y - overlap_y, mid_x + overlap_x, height),
            (mid_x - overlap_x, mid_y - overlap_y, width, height),
        ]
        names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        return [{"label": names[idx], "bounds": bounds} for idx, bounds in enumerate(quadrants)]

    def _build_small_object_specs(self, width: int, height: int) -> list[dict]:
        tile_w = max(64, int(width * 0.55))
        tile_h = max(64, int(height * 0.55))
        x_positions = [0, max(0, (width - tile_w) // 2), max(0, width - tile_w)]
        y_positions = [0, max(0, (height - tile_h) // 2), max(0, height - tile_h)]

        specs = []
        seen = set()
        for row_idx, y1 in enumerate(y_positions):
            for col_idx, x1 in enumerate(x_positions):
                x2 = min(width, x1 + tile_w)
                y2 = min(height, y1 + tile_h)
                bounds = (x1, y1, x2, y2)
                if bounds in seen:
                    continue
                seen.add(bounds)
                specs.append(
                    {
                        "label": f"finecrop_r{row_idx + 1}c{col_idx + 1}",
                        "bounds": bounds,
                    }
                )
        return specs

    def _score_crop_report(self, report: dict) -> float:
        regions = self._extract_regions_from_report(report)
        if not regions:
            return 0.0

        best_score = 0.0
        for region in regions:
            box = region.get("box_2d", [])
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            area_ratio = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / 1_000_000.0))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            dist = ((cx - 500.0) ** 2 + (cy - 500.0) ** 2) ** 0.5
            centrality = max(0.0, 1.0 - dist / 707.11)
            score = centrality + 0.5 * min(area_ratio / 0.10, 1.0)
            best_score = max(best_score, score)
        return best_score

    def _extract_regions_from_report(self, report: dict) -> list:
        regions = report.get("evidence_regions", [])
        if regions:
            return regions
        for key, value in report.items():
            if ("evidence" in key or "region" in key) and isinstance(value, list):
                if any(isinstance(item, dict) and "box_2d" in item for item in value):
                    return value
        return []

    def _report_has_oversized_regions(self, report: dict) -> bool:
        regions = self._extract_regions_from_report(report)
        if not regions:
            return False
        for region in regions:
            box = region.get("box_2d", [])
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            width_ratio = max(0.0, (x2 - x1) / 1000.0)
            height_ratio = max(0.0, (y2 - y1) / 1000.0)
            area_ratio = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / 1_000_000.0))
            if area_ratio >= 0.18 or width_ratio >= 0.80 or height_ratio >= 0.80:
                return True
        return False

    def _largest_region_area_ratio(self, report: dict) -> float:
        regions = self._extract_regions_from_report(report)
        largest = 0.0
        for region in regions:
            box = region.get("box_2d", [])
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            area_ratio = max(0.0, min(1.0, ((x2 - x1) * (y2 - y1)) / 1_000_000.0))
            largest = max(largest, area_ratio)
        return largest

    def _run_region_proposal_pass(
        self, img: Image.Image, crop_specs: list[dict], stage_name: str
    ) -> dict | None:
        positive_reports = []
        for crop_idx, crop_spec in enumerate(crop_specs):
            x1, y1, x2, y2 = crop_spec["bounds"]
            crop = img.crop((x1, y1, x2, y2))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            crop.save(tmp_path)

            crop_prompt = (
                f"This is a crop from the original image: {crop_spec['label']}. "
                "Identify any localized suspicious regions that could indicate tampering, compositing, or object-level editing. "
                "Return strict JSON only."
            )
            try:
                crop_report = self._run_inference(
                    tmp_path,
                    user_prompt=crop_prompt,
                    system_instruction=HIGH_RECALL_REGION_SYSTEM_PROMPT,
                )
                crop_regions = self._extract_regions_from_report(crop_report)
                if crop_regions:
                    crop_report["_multiscale_source"] = (
                        f"{stage_name}_{crop_idx}_{crop_spec['label']}"
                    )
                    crop_report["_multiscale_crop_bounds"] = [int(x1), int(y1), int(x2), int(y2)]
                    crop_report["_multiscale_stage"] = stage_name
                    crop_report["_proposal_mode"] = "high_recall_region_fallback"
                    positive_reports.append((self._score_crop_report(crop_report), crop_report))
            except Exception:
                continue
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        if positive_reports:
            positive_reports.sort(key=lambda item: item[0], reverse=True)
            return positive_reports[0][1]
        return None

    def propose_suspicious_regions(self, image_path: str) -> dict | None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        full_report = self._run_inference(
            image_path,
            user_prompt=HIGH_RECALL_REGION_USER_PROMPT,
            system_instruction=HIGH_RECALL_REGION_SYSTEM_PROMPT,
        )
        full_regions = self._extract_regions_from_report(full_report)
        if full_regions:
            full_report["_proposal_mode"] = "high_recall_region_fallback"
            if not self._report_has_oversized_regions(full_report):
                return full_report
            if self._largest_region_area_ratio(full_report) >= 0.18:
                full_report["_proposal_mode"] = "high_recall_region_large_box"
                return full_report

        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        try:
            quadrant_report = self._run_region_proposal_pass(
                img, self._build_quadrant_specs(width, height), "quadrant"
            )
            if quadrant_report is not None:
                return quadrant_report

            if self.small_object_second_pass:
                fine_report = self._run_region_proposal_pass(
                    img, self._build_small_object_specs(width, height), "finecrop"
                )
                if fine_report is not None:
                    return fine_report
            return None
        finally:
            img.close()

    def analyze_image_multiscale(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        full_report = self._run_inference(image_path)
        if full_report.get("is_tampered", False) and not self._report_has_oversized_regions(
            full_report
        ):
            return full_report

        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        quadrant_report = self._run_crop_pass(
            img, self._build_quadrant_specs(width, height), "quadrant"
        )
        if quadrant_report is not None:
            img.close()
            return quadrant_report

        if self.small_object_second_pass:
            fine_report = self._run_crop_pass(
                img, self._build_small_object_specs(width, height), "finecrop"
            )
            img.close()
            if fine_report is not None:
                return fine_report
        else:
            img.close()

        return full_report

    def _parse_json_output(self, raw_text: str) -> dict:
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        try:
            parsed_json = json.loads(cleaned_text)
            if "evidence_regions" not in parsed_json and "eevidence_regions" in parsed_json:
                parsed_json["evidence_regions"] = parsed_json["eevidence_regions"]
            return parsed_json
        except json.JSONDecodeError:
            fallback_dict = {
                "is_tampered": False,
                "chain_of_thought": raw_text,
                "evidence_regions": [],
            }

            lowered = raw_text.lower()
            if '"is_tampered": true' in lowered:
                fallback_dict["is_tampered"] = True
            elif '"is_tampered": false' in lowered:
                fallback_dict["is_tampered"] = False
            elif "true" in lowered and "false" not in lowered:
                fallback_dict["is_tampered"] = True

            box_match = re.search(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]", raw_text)
            if box_match:
                try:
                    box = json.loads(box_match.group(0))
                    fallback_dict["evidence_regions"] = [
                        {"description": "Regex extracted region", "box_2d": box}
                    ]
                except Exception:
                    pass

            return fallback_dict
