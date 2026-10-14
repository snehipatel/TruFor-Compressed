"""
Feature Caching Script for Stage 1 TruFor Detection Head Fine-Tuning.
Extracts and persists the deterministic 8-dimensional detection features
from the frozen TruFor backbone and decoders for Stage 1 manifests.
"""

import os
import sys
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from study.model_manager import get_trufor_model
from test_docker.src.models.cmx.layer_utils import weighted_statistics_pooling
from study.dataset_builder_stage1 import is_compressed_path

CACHE_DIR = BASE_DIR / "study" / "cache_features"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage1"


def extract_features_from_manifest(manifest_path: Path, output_file: Path, model, device, max_size=512):
    if output_file.exists():
        print(f"[*] Cache file already exists: {output_file.name} - loading existing cache.")
        return torch.load(output_file, weights_only=False)

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"\n[*] Extracting features for {len(samples)} samples from {manifest_path.name}...")
    t0 = time.time()
    
    feats = []
    labels = []
    ids = []
    datasets = []
    
    for i, s in enumerate(samples):
        p = s["image_path"]
        if is_compressed_path(p):
            raise ValueError(f"[PROTOCOL VIOLATION] Compressed path in dataset: {p}")
            
        full_p = BASE_DIR / p
        pil_img = Image.open(str(full_p)).convert("RGB")
        w, h = pil_img.size
        
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w = max(8, (int(round(w * scale)) // 8) * 8)
            new_h = max(8, (int(round(h * scale)) // 8) * 8)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w = max(8, (w // 8) * 8)
            new_h = max(8, (h // 8) * 8)
            if (new_w, new_h) != (w, h):
                pil_img = pil_img.crop((0, 0, new_w, new_h))
                
        img_np = np.array(pil_img, dtype=np.float32) / 255.0
        t_rgb = torch.tensor(img_np.transpose(2, 0, 1), device=device).unsqueeze(0)
        
        with torch.no_grad():
            out, conf, _, _ = model(t_rgb)
            f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
            f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(out.shape[0], -1)
            f_vec = torch.cat((f1, f2), -1).squeeze(0).cpu()
            
        feats.append(f_vec)
        labels.append(s["label"])
        ids.append(s["id"])
        datasets.append(s["dataset"])
        
        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            avg_sec = elapsed / (i + 1)
            remain = avg_sec * (len(samples) - (i + 1))
            print(f"    [{i+1:3d}/{len(samples)}] Extracted | Avg: {avg_sec:.2f}s/img | Est Remaining: {remain/60:.1f}m", flush=True)
            
    payload = {
        "features": torch.stack(feats),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "ids": ids,
        "datasets": datasets
    }
    torch.save(payload, output_file)
    print(f"[+] Saved {len(feats)} cached features to {output_file.name} in {(time.time()-t0)/60:.2f}m")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Extract pilot features only")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print(f"Loading TruFor feature extractor on {device}...")
    model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    model.eval()
    
    if args.pilot:
        extract_features_from_manifest(
            SPLITS_DIR / "train_pilot_manifest.json",
            CACHE_DIR / "train_pilot_features.pt",
            model, device
        )
        extract_features_from_manifest(
            SPLITS_DIR / "val_pilot_manifest.json",
            CACHE_DIR / "val_pilot_features.pt",
            model, device
        )
    else:
        extract_features_from_manifest(
            SPLITS_DIR / "train_manifest.json",
            CACHE_DIR / "train_features.pt",
            model, device
        )
        extract_features_from_manifest(
            SPLITS_DIR / "val_manifest.json",
            CACHE_DIR / "val_features.pt",
            model, device
        )
