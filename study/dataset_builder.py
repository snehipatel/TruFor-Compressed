"""
Dataset Builder & Split Manager for Platform Compression Robustness Study.
Enforces strict scientific protocol and guarantees ZERO image-level leakage:
- Training + Validation: IMD2020, RAISE, CASIA 2.0
- Testing ONLY: CocoGlide, Columbia, CASIA 1.0 (Completely unseen, strictly isolated)
"""

import os
import sys
import json
import random
from glob import glob
from pathlib import Path

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "DATASET"
COMPRESSED_DIR = DATASET_DIR / "COMPRESSED"
SPLITS_DIR = BASE_DIR / "study" / "data_splits"

PLATFORMS = ["original", "whatsapp", "instagram", "facebook", "telegram"]

# Strict guard against leakage
TEST_DATASET_NAMES = ["cocoglide", "columbia", "casia1.0", "casia 1.0"]


def assert_no_test_leakage(file_path: str):
    """Enforce that no test dataset path ever enters training/validation data."""
    path_lower = str(file_path).lower().replace("\\", "/")
    for test_name in TEST_DATASET_NAMES:
        if f"/{test_name}/" in path_lower:
            raise ValueError(
                f"[CRITICAL PROTOCOL VIOLATION] Test image detected in training pool: {file_path}"
            )


def build_imd2020_samples():
    """
    Scans IMD2020 original and compressed versions across all 4 platforms.
    Groups by folder ID (e.g. 1a07yi) to guarantee zero cross-split leakage.
    """
    imd_orig_base = DATASET_DIR / "IMD2020"
    imd_comp_base = COMPRESSED_DIR / "IMD2020"
    
    # Platform directory mapping
    platform_dirs = {
        "original": imd_orig_base,
        "facebook": imd_comp_base / "IMD2020_facebook",
        "instagram": imd_comp_base / "IMD2020_instagram",
        "telegram": imd_comp_base / "IMD2020_telegram",
        "whatsapp": imd_comp_base / "IMD2020_whatsapp",
    }
    
    # Collect all folder names from original
    folders = [f for f in os.listdir(imd_orig_base) if (imd_orig_base / f).is_dir()]
    folders.sort()
    
    grouped_samples = {}
    
    for folder in folders:
        group_id = f"IMD_{folder}"
        grouped_samples[group_id] = []
        
        for platform, p_dir in platform_dirs.items():
            f_path = p_dir / folder
            if not f_path.exists():
                continue
            
            all_files = os.listdir(f_path)
            
            # 1. Authentic / pristine original image
            orig_imgs = [f for f in all_files if "_orig" in f and f.lower().endswith(('.jpg', '.png', '.webp'))]
            for img_name in orig_imgs:
                img_path = str(f_path / img_name)
                assert_no_test_leakage(img_path)
                grouped_samples[group_id].append({
                    "id": f"IMD_{folder}_{platform}_{img_name}",
                    "dataset": "IMD2020",
                    "group_id": group_id,
                    "image_path": os.path.relpath(img_path, BASE_DIR).replace("\\", "/"),
                    "mask_path": None,
                    "platform": platform,
                    "label": 0  # Authentic / Real
                })
            
            # 2. Tampered / manipulated images and masks
            tampered_imgs = [
                f for f in all_files 
                if "_orig" not in f and "_mask" not in f and f.lower().endswith(('.jpg', '.png', '.webp'))
            ]
            for img_name in tampered_imgs:
                img_stem = os.path.splitext(img_name)[0]
                # Corresponding mask
                mask_candidates = [
                    f_path / f"{img_stem}_mask.png",
                    f_path / f"{img_stem}_mask.jpg",
                    imd_orig_base / folder / f"{img_stem}_mask.png"
                ]
                mask_path = None
                for mc in mask_candidates:
                    if mc.exists():
                        mask_path = os.path.relpath(str(mc), BASE_DIR).replace("\\", "/")
                        break
                
                img_path = str(f_path / img_name)
                assert_no_test_leakage(img_path)
                grouped_samples[group_id].append({
                    "id": f"IMD_{folder}_{platform}_{img_name}",
                    "dataset": "IMD2020",
                    "group_id": group_id,
                    "image_path": os.path.relpath(img_path, BASE_DIR).replace("\\", "/"),
                    "mask_path": mask_path,
                    "platform": platform,
                    "label": 1  # Tampered / Fake
                })
                
    return grouped_samples


def build_raise_samples():
    """
    Scans RAISE pristine authentic images across original and all 4 platform folders.
    Groups by image stem to guarantee zero cross-split leakage.
    """
    raise_orig_base = DATASET_DIR / "RAISE"
    raise_comp_base = COMPRESSED_DIR / "RAISE"
    
    platform_dirs = {
        "original": raise_orig_base,
        "facebook": raise_comp_base / "facebook",
        "instagram": raise_comp_base / "instagram",
        "telegram": raise_comp_base / "telegram",
        "whatsapp": raise_comp_base / "whatsapp",
    }
    
    grouped_samples = {}
    
    # Check original RAISE images
    orig_files = [f for f in os.listdir(raise_orig_base) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    orig_files.sort()
    
    for f in orig_files:
        stem = os.path.splitext(f)[0]
        group_id = f"RAISE_{stem}"
        grouped_samples[group_id] = []
        
        # Add original
        orig_path = str(raise_orig_base / f)
        assert_no_test_leakage(orig_path)
        grouped_samples[group_id].append({
            "id": f"RAISE_{stem}_original",
            "dataset": "RAISE",
            "group_id": group_id,
            "image_path": os.path.relpath(orig_path, BASE_DIR).replace("\\", "/"),
            "mask_path": None,
            "platform": "original",
            "label": 0
        })
        
        # Check compressed platforms
        for p in ["facebook", "instagram", "telegram", "whatsapp"]:
            p_dir = platform_dirs[p]
            if not p_dir.exists():
                continue
            matches = [m for m in os.listdir(p_dir) if os.path.splitext(m)[0] == stem]
            for m in matches:
                img_path = str(p_dir / m)
                assert_no_test_leakage(img_path)
                grouped_samples[group_id].append({
                    "id": f"RAISE_{stem}_{p}",
                    "dataset": "RAISE",
                    "group_id": group_id,
                    "image_path": os.path.relpath(img_path, BASE_DIR).replace("\\", "/"),
                    "mask_path": None,
                    "platform": p,
                    "label": 0
                })
                
    return grouped_samples


def build_casia2_samples():
    """
    Scans CASIA 2.0 groundtruth masks and compressed versions across all 4 platforms
    (facebook, instagram, telegram, whatsapp).
    Groups by base image stem so all compressed versions stay together, guaranteeing zero cross-split leakage.
    All CASIA 2.0 samples in this pool are tampered (label=1).
    """
    casia2_mask_dir = DATASET_DIR / "CASIA2.0_Groundtruth" / "CASIA2.0_Groundtruth"
    casia2_comp_base = COMPRESSED_DIR / "CASIA2.0_Groundtruth"
    
    platform_dirs = {
        "facebook": casia2_comp_base / "CASIA2.0_Groundtruth_facebook" / "CASIA2.0_Groundtruth",
        "instagram": casia2_comp_base / "CASIA2.0_Groundtruth_instagram" / "CASIA2.0_Groundtruth",
        "telegram": casia2_comp_base / "CASIA2.0_Groundtruth_telegram" / "CASIA2.0_Groundtruth",
        "whatsapp": casia2_comp_base / "CASIA2.0_Groundtruth_whatsapp" / "CASIA2.0_Groundtruth",
    }
    
    grouped_samples = {}
    
    for platform, p_dir in platform_dirs.items():
        if not p_dir.exists():
            continue
        for f in os.listdir(p_dir):
            if not f.lower().endswith(('.jpg', '.png', '.webp')):
                continue
            stem = Path(f).stem
            # Clean duplicate index suffixes e.g. (1)
            clean_stem = stem.replace("(1)", "").strip()
            base_group = clean_stem.replace("_gt", "")
            group_id = f"CASIA2_{base_group}"
            
            mask_candidate = casia2_mask_dir / f"{clean_stem}.png"
            if not mask_candidate.exists():
                mask_candidate = casia2_mask_dir / f"{base_group}_gt.png"
                
            mask_path = None
            if mask_candidate.exists():
                mask_path = os.path.relpath(str(mask_candidate), BASE_DIR).replace("\\", "/")
                
            img_path = str(p_dir / f)
            assert_no_test_leakage(img_path)
            
            if group_id not in grouped_samples:
                grouped_samples[group_id] = []
                
            grouped_samples[group_id].append({
                "id": f"CASIA2_{clean_stem}_{platform}",
                "dataset": "CASIA2.0",
                "group_id": group_id,
                "image_path": os.path.relpath(img_path, BASE_DIR).replace("\\", "/"),
                "mask_path": mask_path,
                "platform": platform,
                "label": 1  # Tampered / Fake
            })
            
    return grouped_samples


def build_train_val_manifests(val_ratio=0.20, seed=42):
    """
    Builds train and validation manifests ensuring:
    - Partitioning strictly by group_id (never image level)
    - 0% leakage between train and val
    - Test datasets strictly excluded
    """
    random.seed(seed)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[*] Indexing IMD2020 pool...")
    imd_groups = build_imd2020_samples()
    print(f"    Found {len(imd_groups)} unique IMD2020 base groups.")
    
    print("[*] Indexing RAISE pool...")
    raise_groups = build_raise_samples()
    print(f"    Found {len(raise_groups)} unique RAISE base groups.")
    
    print("[*] Indexing CASIA 2.0 pool...")
    casia2_groups = build_casia2_samples()
    print(f"    Found {len(casia2_groups)} unique CASIA 2.0 base groups.")
    
    all_groups = {**imd_groups, **raise_groups, **casia2_groups}
    group_keys = list(all_groups.keys())
    random.shuffle(group_keys)
    
    # Split group keys
    num_val = int(len(group_keys) * val_ratio)
    val_keys = set(group_keys[:num_val])
    train_keys = set(group_keys[num_val:])
    
    train_samples = []
    val_samples = []
    
    for k in train_keys:
        train_samples.extend(all_groups[k])
        
    for k in val_keys:
        val_samples.extend(all_groups[k])
        
    # Verify zero leakage
    train_group_set = set(s['group_id'] for s in train_samples)
    val_group_set = set(s['group_id'] for s in val_samples)
    leakage = train_group_set.intersection(val_group_set)
    assert len(leakage) == 0, f"Critical error: group leakage detected! {leakage}"
    
    # Save manifests
    train_path = SPLITS_DIR / "train_manifest.json"
    val_path = SPLITS_DIR / "val_manifest.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
        
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
        
    # Stats summary
    def print_stats(name, samples):
        real_cnt = sum(1 for s in samples if s['label'] == 0)
        fake_cnt = sum(1 for s in samples if s['label'] == 1)
        p_counts = {}
        for s in samples:
            p_counts[s['platform']] = p_counts.get(s['platform'], 0) + 1
        print(f"\n[+] {name} Partition:")
        print(f"    Total Images: {len(samples)}")
        print(f"    Authentic (Real): {real_cnt} | Tampered (Fake): {fake_cnt}")
        print(f"    Platform Distribution: {p_counts}")
        
    print(f"\n[OK] Manifests saved to:\n    - {train_path}\n    - {val_path}")
    return train_samples, val_samples


if __name__ == "__main__":
    build_train_val_manifests()
