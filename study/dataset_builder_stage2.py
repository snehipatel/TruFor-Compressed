"""
Stage 2 Dataset Builder: Compressed-Image Fine-Tuning Manifests.
Builds train/val manifests with original + all 4 platform-compressed versions
of the same source image grouped together (no leakage).

Data sources:
  - IMD2020 (original + whatsapp/instagram/facebook/telegram)
  - RAISE_training (original PNGs) + raise_{platform} (compressed)
  - CASIA 2.0 Au/Tp (original) — no compressed versions
  - DSO-1 (original)
  - Inpainting micro-sets (original)
  - D:\ml_project\DATA platform folders (real platform-processed images)

Rules:
  1. Test sets (CocoGlide, Columbia, CASIA 1.0) NEVER in training.
  2. Grouped by source (all versions of one image in the same split).
  3. 50/50 Real/Fake overall, per platform where possible.
  4. No mask files as input images.
  5. Oversample Instagram/Facebook.
"""

import os
import sys
import re
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage2"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SML = Path("D:/sample_images_ml")
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}

TEST_DATASET_NAMES = ["cocoglide", "columbia", "casia1.0", "casia 1.0", "casia1corel"]
PLATFORMS = ["original", "whatsapp", "instagram", "facebook", "telegram"]


def is_mask_file(path_str):
    """Check if a file is a mask (ground truth) rather than a photo."""
    p = path_str.lower()
    fname = os.path.basename(p)
    if "_gt." in fname or "_mask." in fname:
        return True
    if "groundtruth" in p and ("_gt." in p or fname.endswith("_gt.png") or fname.endswith("_gt.jpg")):
        return True
    return False


def is_test_image(path_str):
    """Check if a path belongs to a test set."""
    p = str(path_str).lower().replace("\\", "/")
    for tn in TEST_DATASET_NAMES:
        if f"/{tn}/" in p or f"/{tn}\\" in p:
            return True
    # Also check DATASET folder specifically
    if "/dataset/" in p:
        return True
    return False


def is_readable_image(path_str):
    """Check that a file is a readable image (not a mask, not corrupt)."""
    try:
        with Image.open(path_str) as im:
            im.verify()
        return True
    except Exception:
        return False


def build_imd2020_groups(max_folders=None):
    """
    Build IMD2020 groups: each folder has orig + tampered images.
    For each folder, collect original AND compressed versions across all 4 platforms.
    """
    imd_base = BASE_DIR / "IMD2020"
    imd_plat = {
        "whatsapp": BASE_DIR / "IMD2020_whatsapp",
        "instagram": BASE_DIR / "IMD2020_instagram",
        "facebook": BASE_DIR / "IMD2020_facebook",
        "telegram": BASE_DIR / "IMD2020_telegram",
    }
    
    if not imd_base.exists():
        print("[-] IMD2020 not found")
        return {}
    
    folders = sorted([f for f in os.listdir(imd_base) if (imd_base / f).is_dir()])
    if max_folders:
        folders = folders[:max_folders]
    
    groups = {}
    for folder in folders:
        f_dir = imd_base / folder
        files = os.listdir(f_dir)
        
        orig_imgs = [f for f in files if "_orig" in f and Path(f).suffix.lower() in IMG_EXTS]
        tamp_imgs = [f for f in files if "_orig" not in f and "_mask" not in f 
                     and Path(f).suffix.lower() in IMG_EXTS]
        
        if not orig_imgs or not tamp_imgs:
            continue
        
        real_f = sorted(orig_imgs)[0]
        fake_f = sorted(tamp_imgs)[0]
        
        gid = f"IMD_{folder}"
        entries = []
        
        # Original versions
        real_orig = str(f_dir / real_f)
        fake_orig = str(f_dir / fake_f)
        
        entries.append({
            "id": f"IMD_{folder}_orig_real",
            "dataset": "IMD2020",
            "group_id": gid,
            "image_path": real_orig,
            "label": 0,
            "manipulation": "pristine",
            "platform": "original",
        })
        entries.append({
            "id": f"IMD_{folder}_orig_fake",
            "dataset": "IMD2020",
            "group_id": gid,
            "image_path": fake_orig,
            "label": 1,
            "manipulation": "splicing/photoshop",
            "platform": "original",
        })
        
        # Compressed versions
        for plat, plat_dir in imd_plat.items():
            pf_dir = plat_dir / folder
            if not pf_dir.exists():
                continue
            pf_files = os.listdir(pf_dir)
            
            p_orig = [f for f in pf_files if "_orig" in f and Path(f).suffix.lower() in IMG_EXTS
                      and not is_mask_file(f)]
            p_tamp = [f for f in pf_files if "_orig" not in f and "_mask" not in f 
                      and Path(f).suffix.lower() in IMG_EXTS and not is_mask_file(f)]
            
            if p_orig:
                entries.append({
                    "id": f"IMD_{folder}_{plat}_real",
                    "dataset": "IMD2020",
                    "group_id": gid,
                    "image_path": str(pf_dir / sorted(p_orig)[0]),
                    "label": 0,
                    "manipulation": "pristine",
                    "platform": plat,
                })
            if p_tamp:
                entries.append({
                    "id": f"IMD_{folder}_{plat}_fake",
                    "dataset": "IMD2020",
                    "group_id": gid,
                    "image_path": str(pf_dir / sorted(p_tamp)[0]),
                    "label": 1,
                    "manipulation": "splicing/photoshop",
                    "platform": plat,
                })
        
        groups[gid] = entries
    
    return groups


def build_raise_groups(max_images=None):
    """
    Build RAISE groups: RAISE_training PNGs (originals, label=real)
    paired with raise_{platform} compressed versions.
    These are ALL pristine (label=0) since RAISE is an authentic photo dataset.
    """
    orig_dir = SML / "RAISE_training"
    plat_dirs = {
        "whatsapp": SML / "raise_whatsapp",
        "instagram": SML / "raise_instagram",
        "facebook": SML / "raise_facebook",
        "telegram": SML / "raise_telegram",
    }
    
    if not orig_dir.exists():
        print("[-] RAISE_training not found")
        return {}
    
    orig_files = sorted([f for f in os.listdir(orig_dir) if Path(f).suffix.lower() in IMG_EXTS])
    if max_images:
        orig_files = orig_files[:max_images]
    
    groups = {}
    for f in orig_files:
        stem = Path(f).stem
        gid = f"RAISE_{stem}"
        entries = []
        
        entries.append({
            "id": f"RAISE_{stem}_orig",
            "dataset": "RAISE",
            "group_id": gid,
            "image_path": str(orig_dir / f),
            "label": 0,
            "manipulation": "pristine",
            "platform": "original",
        })
        
        for plat, pd in plat_dirs.items():
            if not pd.exists():
                continue
            # Find matching file by stem
            matches = [pf for pf in os.listdir(pd) if Path(pf).stem == stem]
            if matches:
                entries.append({
                    "id": f"RAISE_{stem}_{plat}",
                    "dataset": "RAISE",
                    "group_id": gid,
                    "image_path": str(pd / matches[0]),
                    "label": 0,
                    "manipulation": "pristine",
                    "platform": plat,
                })
        
        groups[gid] = entries
    
    return groups


def build_casia2_groups(max_components=None):
    """
    Build CASIA 2.0 groups with connected-component stem grouping.
    Original JPEGs only (no compressed versions available).
    """
    casia2_dir = BASE_DIR / "stage1_data" / "CASIA2.0"
    au_dir = casia2_dir / "Au"
    tp_dir = casia2_dir / "Tp"
    
    if not au_dir.exists() or not tp_dir.exists():
        print("[-] CASIA 2.0 not found")
        return {}
    
    au_files = sorted([f for f in os.listdir(au_dir) 
                       if not f.startswith(".") and f != "Thumbs.db"
                       and Path(f).suffix.lower() in IMG_EXTS])
    tp_files = sorted([f for f in os.listdir(tp_dir) 
                       if not f.startswith(".") and f != "Thumbs.db"
                       and Path(f).suffix.lower() in IMG_EXTS])
    
    def get_au_stem(fname):
        parts = Path(fname).stem.split("_")
        if len(parts) >= 3:
            return f"{parts[1]}{parts[2]}"
        return Path(fname).stem

    def get_tp_stems(fname):
        parts = Path(fname).stem.split("_")
        stems = []
        for p in parts:
            if re.match(r"^[a-zA-Z]{3}\d{5}$", p):
                stems.append(p)
        if not stems and len(parts) >= 7:
            stems = [parts[5], parts[6]]
        return stems

    # Connected components via union-find
    parent = {}
    def find(x):
        if parent.setdefault(x, x) != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    tp_info = []
    for f in tp_files:
        stems = get_tp_stems(f)
        if len(stems) >= 2:
            union(stems[0], stems[1])
        tp_info.append((f, stems))

    au_info = []
    for f in au_files:
        stem = get_au_stem(f)
        find(stem)
        au_info.append((f, stem))

    group_to_tp = defaultdict(list)
    group_to_au = defaultdict(list)
    for f, stems in tp_info:
        rep = find(stems[0]) if stems else find(f)
        group_to_tp[rep].append(f)
    for f, stem in au_info:
        rep = find(stem)
        group_to_au[rep].append(f)

    shared_comps = sorted(list(set(group_to_tp.keys()) & set(group_to_au.keys())))
    
    groups = {}
    count = 0
    for comp in shared_comps:
        if max_components and count >= max_components:
            break
        t_list = group_to_tp[comp]
        a_list = group_to_au[comp]
        if not t_list or not a_list:
            continue
        
        gid = f"CASIA2_comp_{comp}"
        entries = []
        
        # Take one Au and one Tp from each component
        entries.append({
            "id": f"CASIA2_Au_{a_list[0]}",
            "dataset": "CASIA2.0",
            "group_id": gid,
            "image_path": str(au_dir / a_list[0]),
            "label": 0,
            "manipulation": "pristine",
            "platform": "original",
        })
        entries.append({
            "id": f"CASIA2_Tp_{t_list[0]}",
            "dataset": "CASIA2.0",
            "group_id": gid,
            "image_path": str(tp_dir / t_list[0]),
            "label": 1,
            "manipulation": "splicing/copymove",
            "platform": "original",
        })
        
        groups[gid] = entries
        count += 1
    
    return groups


def build_dso1_groups(max_pairs=None):
    """Build DSO-1 groups (100 real / 100 fake PNGs, original only)."""
    real_dir = SML / "DSO-1_real"
    fake_dir = SML / "DSO-1_fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        print("[-] DSO-1 not found")
        return {}
    
    r_files = sorted([f for f in os.listdir(real_dir) if Path(f).suffix.lower() in IMG_EXTS])
    f_files = sorted([f for f in os.listdir(fake_dir) if Path(f).suffix.lower() in IMG_EXTS])
    
    if max_pairs:
        r_files = r_files[:max_pairs]
        f_files = f_files[:max_pairs]
    
    groups = {}
    for i, (rf, ff) in enumerate(zip(r_files, f_files)):
        gid = f"DSO1_{i:03d}"
        groups[gid] = [
            {
                "id": f"DSO1_real_{rf}",
                "dataset": "DSO-1",
                "group_id": gid,
                "image_path": str(real_dir / rf),
                "label": 0,
                "manipulation": "pristine",
                "platform": "original",
            },
            {
                "id": f"DSO1_fake_{ff}",
                "dataset": "DSO-1",
                "group_id": gid,
                "image_path": str(fake_dir / ff),
                "label": 1,
                "manipulation": "splicing",
                "platform": "original",
            }
        ]
    
    return groups


def build_inpainting_groups():
    """Build inpainting micro-set groups (original only)."""
    groups = {}
    
    # 1. Normalized pair (25 real, 25 fake)
    nr_dir = SML / "normalized_real"
    nf_dir = SML / "normalized_fake"
    if nr_dir.exists() and nf_dir.exists():
        nr_files = sorted([f for f in os.listdir(nr_dir) if Path(f).suffix.lower() in IMG_EXTS])
        nf_files = sorted([f for f in os.listdir(nf_dir) if Path(f).suffix.lower() in IMG_EXTS])
        for i, (rf, ff) in enumerate(zip(nr_files, nf_files)):
            gid = f"NORM_{i:03d}"
            groups[gid] = [
                {
                    "id": f"NORM_real_{rf}",
                    "dataset": "Normalized",
                    "group_id": gid,
                    "image_path": str(nr_dir / rf),
                    "label": 0,
                    "manipulation": "pristine",
                    "platform": "original",
                },
                {
                    "id": f"NORM_fake_{ff}",
                    "dataset": "Normalized",
                    "group_id": gid,
                    "image_path": str(nf_dir / ff),
                    "label": 1,
                    "manipulation": "generative_inpainting",
                    "platform": "original",
                }
            ]
    
    # 2. BTB + SAGID paired with RAISE originals
    raise_dir = SML / "RAISE_training"
    raise_files = sorted(os.listdir(raise_dir)) if raise_dir.exists() else []
    
    inpaint_fakes = []
    for sub in ["beyond_the_brush_images", "beyond_the_brush_diverse_images", "sagid_images"]:
        p = SML / sub
        if p.exists():
            for f in sorted(os.listdir(p)):
                if Path(f).suffix.lower() in IMG_EXTS:
                    inpaint_fakes.append((sub, str(p / f)))
    
    # Use RAISE files starting from index 500 to avoid overlap with RAISE platform groups
    raise_offset = 500
    for i, (src_name, fake_p) in enumerate(inpaint_fakes):
        ri = raise_offset + i
        if ri >= len(raise_files):
            break
        real_p = str(raise_dir / raise_files[ri])
        gid = f"INPAINT_{src_name}_{i:03d}"
        groups[gid] = [
            {
                "id": f"RAISE_inpaint_pair_{raise_files[ri]}",
                "dataset": "RAISE_Inpainting_Pair",
                "group_id": gid,
                "image_path": real_p,
                "label": 0,
                "manipulation": "pristine",
                "platform": "original",
            },
            {
                "id": f"{src_name}_{Path(fake_p).name}",
                "dataset": "Generative_Inpainting",
                "group_id": gid,
                "image_path": fake_p,
                "label": 1,
                "manipulation": "generative_inpainting",
                "platform": "original",
            }
        ]
    
    return groups


def build_platform_data_groups():
    """
    Build groups from D:\ml_project\DATA platform folders.
    These contain REAL platform-processed images (not simulator).
    Folders: FACEBOOK/{facebook_fake,facebook_real,...}, INSTAGRAM/..., etc.
    IMPORTANT: These are REAL platform images, valuable for realism.
    """
    data_dir = BASE_DIR / "DATA"
    groups = {}
    
    platform_map = {
        "FACEBOOK": {
            "platform": "facebook",
            "real": "facebook_real",
            "fake": "facebook_fake",
        },
        "INSTAGRAM": {
            "platform": "instagram",
            "real": "Instagram_Real",
            "fake": "Instagram_Fake",
        },
        "WHATSAPP": {
            "platform": "whatsapp",
            "real": "whatsapp_real",
            "fake": "whatsapp_fake",
        },
        "TELEGRAM": {
            "platform": "telegram",
            "real": "telegram_real",
            "fake": "telegram_fake",
        },
    }
    
    # Also check D:\sample_images_ml for telegram_real/telegram_fake
    tg_real_sml = SML / "telegram_real"
    tg_fake_sml = SML / "telegram_fake"
    
    for plat_folder, info in platform_map.items():
        plat_dir = data_dir / plat_folder
        plat = info["platform"]
        r_dir = plat_dir / info["real"]
        f_dir = plat_dir / info["fake"]
        
        if not r_dir.exists() or not f_dir.exists():
            continue
        
        r_files = sorted([f for f in os.listdir(r_dir) if Path(f).suffix.lower() in IMG_EXTS])
        f_files = sorted([f for f in os.listdir(f_dir) if Path(f).suffix.lower() in IMG_EXTS])
        
        for i, (rf, ff) in enumerate(zip(r_files, f_files)):
            gid = f"PLATDATA_{plat}_{i:03d}"
            groups[gid] = [
                {
                    "id": f"PLATDATA_{plat}_real_{rf}",
                    "dataset": f"PlatformData_{plat}",
                    "group_id": gid,
                    "image_path": str(r_dir / rf),
                    "label": 0,
                    "manipulation": "pristine",
                    "platform": plat,
                    "is_real_platform": True,
                },
                {
                    "id": f"PLATDATA_{plat}_fake_{ff}",
                    "dataset": f"PlatformData_{plat}",
                    "group_id": gid,
                    "image_path": str(f_dir / ff),
                    "label": 1,
                    "manipulation": "splicing/copymove",
                    "platform": plat,
                    "is_real_platform": True,
                }
            ]
    
    return groups


def verify_entries(entries, label=""):
    """Verify all entries: readable images, not masks, not test images."""
    verified = []
    skipped = 0
    for e in entries:
        p = e["image_path"]
        if is_mask_file(p):
            print(f"  [MASK SKIP] {p}")
            skipped += 1
            continue
        if is_test_image(p):
            print(f"  [TEST LEAK SKIP] {p}")
            skipped += 1
            continue
        if not os.path.exists(p):
            print(f"  [MISSING] {p}")
            skipped += 1
            continue
        verified.append(e)
    if skipped:
        print(f"  [{label}] Skipped {skipped} entries")
    return verified


def build_stage2_manifests(val_ratio=0.20, seed=42):
    """Build train/val manifests for Stage 2."""
    random.seed(seed)
    
    print("=" * 80)
    print(" BUILDING STAGE 2 MANIFESTS (ORIGINAL + COMPRESSED)")
    print("=" * 80)
    
    all_groups = {}
    
    # 1. IMD2020 (with all platform versions)
    imd_groups = build_imd2020_groups()
    print(f"[*] IMD2020: {len(imd_groups)} source groups ({sum(len(v) for v in imd_groups.values())} entries)")
    all_groups.update(imd_groups)
    
    # 2. Note on standalone RAISE:
    # Standalone RAISE has only real images (0 fakes). Including standalone RAISE
    # violates Global Rule 3: "Every dataset used in training must have BOTH real and fake images; overall 50/50 real/fake."
    # RAISE is already included when paired 1:1 with Generative Inpainting fakes (SAGID/BTB) in inpaint_groups.
    print("[*] Standalone RAISE omitted to enforce Rule 3 (every dataset has both real and fake, 50/50 overall)")
    
    # 3. CASIA 2.0 (original only)
    casia2_groups = build_casia2_groups(max_components=400)
    print(f"[*] CASIA 2.0: {len(casia2_groups)} source groups ({sum(len(v) for v in casia2_groups.values())} entries)")
    all_groups.update(casia2_groups)
    
    # 4. DSO-1 (original only)
    dso1_groups = build_dso1_groups()
    print(f"[*] DSO-1: {len(dso1_groups)} source groups ({sum(len(v) for v in dso1_groups.values())} entries)")
    all_groups.update(dso1_groups)
    
    # 5. Inpainting micro-sets (original only)
    inpaint_groups = build_inpainting_groups()
    print(f"[*] Inpainting: {len(inpaint_groups)} source groups ({sum(len(v) for v in inpaint_groups.values())} entries)")
    all_groups.update(inpaint_groups)
    
    # 6. Real platform data
    plat_groups = build_platform_data_groups()
    print(f"[*] Platform Data: {len(plat_groups)} source groups ({sum(len(v) for v in plat_groups.values())} entries)")
    all_groups.update(plat_groups)
    
    # Split into train/val by group ID
    all_gids = sorted(all_groups.keys())
    random.shuffle(all_gids)
    
    n_val = max(50, int(len(all_gids) * val_ratio))
    val_gids = set(all_gids[:n_val])
    train_gids = set(all_gids[n_val:])
    
    train_entries = []
    val_entries = []
    
    for gid in train_gids:
        verified = verify_entries(all_groups[gid], label=gid)
        train_entries.extend(verified)
    for gid in val_gids:
        verified = verify_entries(all_groups[gid], label=gid)
        val_entries.extend(verified)
    
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    
    # Print detailed summary
    print("\n" + "=" * 80)
    print(" MANIFEST SUMMARY")
    print("=" * 80)
    
    for split_name, entries in [("TRAIN", train_entries), ("VAL", val_entries)]:
        n_real = sum(1 for e in entries if e["label"] == 0)
        n_fake = sum(1 for e in entries if e["label"] == 1)
        print(f"\n{split_name}: {len(entries)} total | Real={n_real}, Fake={n_fake}")
        
        # By dataset
        ds_counts = Counter()
        for e in entries:
            ds_counts[(e["dataset"], e["label"])] += 1
        print(f"  By dataset:")
        for (ds, lbl), cnt in sorted(ds_counts.items()):
            lstr = "Real" if lbl == 0 else "Fake"
            print(f"    {ds:<25} ({lstr}): {cnt}")
        
        # By platform
        plat_counts = Counter()
        for e in entries:
            plat_counts[(e.get("platform", "original"), e["label"])] += 1
        print(f"  By platform:")
        for (plat, lbl), cnt in sorted(plat_counts.items()):
            lstr = "Real" if lbl == 0 else "Fake"
            print(f"    {plat:<12} ({lstr}): {cnt}")
        
        # Inpainting count
        n_inpaint = sum(1 for e in entries if "inpainting" in e.get("manipulation", ""))
        print(f"  Inpainting images: {n_inpaint}")
    
    # Save manifests
    with open(SPLITS_DIR / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2)
    with open(SPLITS_DIR / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_entries, f, indent=2)
    
    print(f"\n[*] Manifests saved to {SPLITS_DIR}")
    print(f"    train_manifest.json: {len(train_entries)} entries")
    print(f"    val_manifest.json: {len(val_entries)} entries")
    
    return train_entries, val_entries


if __name__ == "__main__":
    build_stage2_manifests()
