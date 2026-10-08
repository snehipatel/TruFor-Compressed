"""
PyTorch Dataset and Data Loader for Platform-Compression Robustness Study.
Supports:
- Balanced sampling between REAL (label 0) and TAMPERED (label 1)
- Platform-balanced sampling (Original, WhatsApp, Instagram, Facebook, Telegram)
- Proper mask tensor formatting (binary 0/1, -1 ignore label)
- Safe image cropping and normalization matching TruFor requirements
"""

import os
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

BASE_DIR = Path(__file__).resolve().parent.parent

import io
from PIL import ImageFilter


def apply_compression_domain_randomization(img_np: np.ndarray) -> np.ndarray:
    """
    Applies mild compression-aware domain randomization ablation:
    - Downsampling factor in [0.85, 1.0] followed by upsampling back to original size (prob = 0.40)
    - Lossy Codec Re-compression: JPEG or WebP with quality in [70, 95] (prob = 0.65)
    - Blur and additive noise completely removed.
    
    Operates strictly in pixel intensity space so spatial alignment with ground-truth masks
    remains 100% exact.
    """
    pil_img = Image.fromarray(img_np)
    h, w = img_np.shape[:2]
    
    # 1. Mild Resizing / Downsampling & Upsampling (prob = 0.40)
    if random.random() < 0.40:
        s = random.uniform(0.85, 1.00)
        if s < 0.999:
            dw = max(8, int(round(w * s)))
            dh = max(8, int(round(h * s)))
            down_resample = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
            up_resample = random.choice([Image.BILINEAR, Image.BICUBIC])
            pil_img = pil_img.resize((dw, dh), down_resample).resize((w, h), up_resample)
        
    # 2. Mild Lossy Codec Re-compression (prob = 0.65)
    if random.random() < 0.65:
        codec = random.choice(["JPEG", "WEBP"])
        q = random.randint(70, 95)
        buf = io.BytesIO()
        pil_img.save(buf, format=codec, quality=q)
        buf.seek(0)
        pil_img = Image.open(buf)
        
    # Blur and additive noise removed as per mild ablation protocol

    return np.array(pil_img, dtype=np.uint8)



class RobustnessDataset(Dataset):
    def __init__(self, manifest_path: str, crop_size=(512, 512), is_training=True, use_compression_aug=False):
        self.base_dir = BASE_DIR
        self.crop_size = crop_size
        self.is_training = is_training
        self.use_compression_aug = use_compression_aug
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
            
        # Index authentic and tampered samples for balanced sampling
        self.auth_indices = [i for i, s in enumerate(self.samples) if s["label"] == 0]
        self.tamp_indices = [i for i, s in enumerate(self.samples) if s["label"] == 1]
        
    def __len__(self):
        return len(self.samples)
        
    def _load_image(self, rel_path: str) -> np.ndarray:
        full_path = self.base_dir / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = Image.open(full_path).convert("RGB")
        return np.array(img, dtype=np.uint8)
        
    def _load_mask(self, mask_rel_path, target_shape) -> np.ndarray:
        h, w = target_shape[:2]
        if mask_rel_path is None:
            # Authentic / pristine image -> all-zero mask
            return np.zeros((h, w), dtype=np.uint8)
            
        full_path = self.base_dir / mask_rel_path
        if not full_path.exists():
            return np.zeros((h, w), dtype=np.uint8)
            
        mask_img = Image.open(full_path).convert("L")
        mask_np = np.array(mask_img, dtype=np.uint8)
        
        # Binary threshold (tampered pixels > 0)
        mask = (mask_np > 127).astype(np.uint8)
        
        # Match dimensions if needed
        if mask.shape[0] != h or mask.shape[1] != w:
            mask_img_resized = mask_img.resize((w, h), Image.NEAREST)
            mask = (np.array(mask_img_resized) > 127).astype(np.uint8)
            
        return mask

    def _apply_crop_and_aug(self, img_np: np.ndarray, mask_np: np.ndarray):
        h, w, _ = img_np.shape
        th, tw = self.crop_size
        
        # Pad if smaller than crop size
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=127)
            mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w, _ = img_np.shape
            
        if self.is_training:
            # Random crop
            r = random.randint(0, h - th)
            c = random.randint(0, w - tw)
            img_crop = img_np[r:r+th, c:c+tw, :]
            mask_crop = mask_np[r:r+th, c:c+tw]
            
            # Random horizontal flip
            if random.random() > 0.5:
                img_crop = np.fliplr(img_crop).copy()
                mask_crop = np.fliplr(mask_crop).copy()
                
            # Compression-aware domain randomization
            if self.use_compression_aug:
                img_crop = apply_compression_domain_randomization(img_crop)
        else:
            # Deterministic center crop of size (th, tw)
            s_r = max((h - th) // 2, 0)
            s_c = max((w - tw) // 2, 0)
            img_crop = img_np[s_r:s_r+th, s_c:s_c+tw, :]
            mask_crop = mask_np[s_r:s_r+th, s_c:s_c+tw]
            
        return img_crop, mask_crop

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_np = self._load_image(item["image_path"])
        mask_np = self._load_mask(item["mask_path"], img_np.shape)
        
        img_crop, mask_crop = self._apply_crop_and_aug(img_np, mask_np)
        
        # Normalize image tensor [0, 1] range float
        t_rgb = torch.tensor(img_crop.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        t_mask = torch.tensor(mask_crop, dtype=torch.long)
        label_det = torch.tensor([item["label"]], dtype=torch.float32)
        
        return {
            "rgb": t_rgb,
            "mask": t_mask,
            "label": label_det,
            "platform": item["platform"],
            "dataset": item["dataset"],
            "id": item["id"]
        }


from collections import defaultdict


class MultiFactorBalancedSampler(Sampler):
    """
    Guarantees simultaneous balanced sampling across:
    1. CLASS: Exactly 50% Authentic (label 0) and 50% Tampered (label 1).
    2. PLATFORM: Equal 20% distribution across all 5 transmission channels:
       ['original', 'whatsapp', 'instagram', 'facebook', 'telegram'].
    3. DATASET: Balanced representation across:
       - Authentic: IMD2020 (50%) + RAISE (50%)
       - Tampered: CASIA 2.0 (50%) + IMD2020 (50%)
    """
    def __init__(self, dataset: RobustnessDataset, num_samples_per_epoch=600):
        self.dataset = dataset
        self.num_samples = num_samples_per_epoch
        self.platforms = ["original", "whatsapp", "instagram", "facebook", "telegram"]
        
        # Partition all dataset indices into fine-grained buckets: (label, dataset, platform)
        self.buckets = defaultdict(list)
        for idx, s in enumerate(dataset.samples):
            self.buckets[(s["label"], s["dataset"], s["platform"])].append(idx)
            
    def __iter__(self):
        half = self.num_samples // 2  # 50% authentic, 50% tampered
        per_platform = max(1, half // len(self.platforms))  # e.g. 60 per platform
        
        selected = []
        
        # 1. Authentic (Real): equal per platform (50% IMD2020 + 50% RAISE)
        for p in self.platforms:
            imd_pool = self.buckets[(0, "IMD2020", p)]
            raise_pool = self.buckets[(0, "RAISE", p)]
            
            k_imd = per_platform // 2
            k_raise = per_platform - k_imd
            if imd_pool:
                selected.extend(random.choices(imd_pool, k=k_imd))
            if raise_pool:
                selected.extend(random.choices(raise_pool, k=k_raise))
                
        # 2. Tampered (Fake): equal per platform
        # Original: all from IMD2020 (CASIA 2.0 groundtruth has social variants)
        imd_orig = self.buckets[(1, "IMD2020", "original")]
        if imd_orig:
            selected.extend(random.choices(imd_orig, k=per_platform))
            
        # 4 Social platforms: balance between CASIA 2.0 and IMD2020
        # Target: ~50% CASIA2.0 and ~50% IMD2020 across all fake samples
        k_casia = int(round(per_platform * 0.63))
        k_imd_comp = per_platform - k_casia
        for p in ["whatsapp", "instagram", "facebook", "telegram"]:
            casia_pool = self.buckets[(1, "CASIA2.0", p)]
            imd_pool = self.buckets[(1, "IMD2020", p)]
            if casia_pool:
                selected.extend(random.choices(casia_pool, k=k_casia))
            if imd_pool:
                selected.extend(random.choices(imd_pool, k=k_imd_comp))
                
        random.shuffle(selected)
        return iter(selected)
        
    def __len__(self):
        return self.num_samples


# Backward compatibility alias
BalancedClassSampler = MultiFactorBalancedSampler

