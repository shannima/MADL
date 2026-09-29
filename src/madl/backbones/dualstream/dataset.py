from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
CLASS_TO_INDEX = {"real": 0, "synthetic": 1, "tampered": 2}
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}


def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTS)


def list_images(directory: str) -> List[str]:
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name) for name in os.listdir(directory) if is_image_file(name)
    )


def find_mask_path(mask_dir: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(mask_dir, f"{stem}_mask.png"),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}_mask.jpg"),
        os.path.join(mask_dir, f"{stem}.jpg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _srm_kernels() -> np.ndarray:
    q = 2.0
    kernels = [
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1],
            ],
            dtype=np.float32,
        )
        / q,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, -2, 4, -2, 0],
                [0, 1, -2, 1, 0],
            ],
            dtype=np.float32,
        ),
    ]
    return np.stack(kernels, axis=0).astype(np.float32)


SRM_KERNELS = _srm_kernels()


def compute_noise_map(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    outputs = []
    for kernel in SRM_KERNELS:
        filtered = cv2.filter2D(gray, -1, kernel)
        filtered = np.clip(filtered, -3.0, 3.0)
        filtered = (filtered + 3.0) / 6.0
        outputs.append(filtered)
    return np.stack(outputs, axis=0).astype(np.float32)


@dataclass
class PixelSample:
    image_path: str
    label_name: str
    label_index: int
    mask_path: str


class PixelForgeryDataset(Dataset):
    def __init__(
        self,
        real_dir: str,
        tampered_dir: str,
        mask_dir: str,
        synthetic_dirs: Sequence[str],
        image_size: int = 224,
        split: str = "train",
        max_samples_per_class: int = 0,
        seed: int = 42,
    ) -> None:
        self.image_size = image_size
        self.split = split
        self.seed = seed
        self.rng = random.Random(seed)

        real_paths = list_images(real_dir)
        tampered_paths = list_images(tampered_dir)
        synthetic_paths: List[str] = []
        for directory in synthetic_dirs:
            synthetic_paths.extend(list_images(directory))

        if max_samples_per_class > 0:
            real_paths = self._sample(real_paths, max_samples_per_class, seed + 1)
            tampered_paths = self._sample(tampered_paths, max_samples_per_class, seed + 2)
            synthetic_paths = self._sample(synthetic_paths, max_samples_per_class, seed + 3)

        self.samples: List[PixelSample] = []
        for path in real_paths:
            self.samples.append(PixelSample(path, "real", CLASS_TO_INDEX["real"], ""))
        for path in synthetic_paths:
            self.samples.append(PixelSample(path, "synthetic", CLASS_TO_INDEX["synthetic"], ""))
        for path in tampered_paths:
            self.samples.append(
                PixelSample(
                    path, "tampered", CLASS_TO_INDEX["tampered"], find_mask_path(mask_dir, path)
                )
            )

        self.samples.sort(key=lambda row: (row.label_index, row.image_path))
        self.class_counts = {
            "real": len(real_paths),
            "synthetic": len(synthetic_paths),
            "tampered": len(tampered_paths),
        }

    def _sample(self, paths: List[str], limit: int, seed: int) -> List[str]:
        if limit <= 0 or len(paths) <= limit:
            return paths
        picked = list(paths)
        random.Random(seed).shuffle(picked)
        return sorted(picked[:limit])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_and_mask(self, sample: PixelSample) -> tuple[Image.Image, Image.Image]:
        image = Image.open(sample.image_path).convert("RGB")
        if sample.mask_path and os.path.exists(sample.mask_path):
            mask = Image.open(sample.mask_path).convert("L")
        else:
            mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8), mode="L")
        return image, mask

    def _resize_pair(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        mask = TF.resize(
            mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST
        )
        return image, mask

    def _augment_pair(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if self.split != "train":
            return image, mask
        if self.rng.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if self.rng.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        image, mask = self._load_image_and_mask(sample)
        image, mask = self._resize_pair(image, mask)
        image, mask = self._augment_pair(image, mask)

        image_np = np.asarray(image, dtype=np.uint8)
        rgb_tensor = TF.to_tensor(image)
        noise_tensor = torch.from_numpy(compute_noise_map(image_np))

        mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        mask_valid = 1.0 if sample.label_name == "tampered" and sample.mask_path else 0.0

        return {
            "rgb": rgb_tensor,
            "noise": noise_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(sample.label_index, dtype=torch.long),
            "mask_valid": torch.tensor(mask_valid, dtype=torch.float32),
            "image_path": sample.image_path,
            "label_name": sample.label_name,
        }
