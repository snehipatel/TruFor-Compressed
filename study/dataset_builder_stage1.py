"""
Stage 1 Dataset Builder & Manifest Generator for TruFor Uncompressed Fine-Tuning.
Strictly enforces:
1. COMPRESSED-DATA EXCLUSION RULE (whatsapp, instagram, facebook, telegram, fb, insta, wa, tg)
2. Zero Test Set Leakage (CocoGlide, Columbia, CASIA 1.0 strictly isolated)
3. Zero Source-Image Leakage (grouping by IMD2020 parent folder and CASIA 2.0 base stem)
4. No Mask Entries as Images (masks used only for ground truth evaluation)
5. 50/50 Real/Fake balance across all splits and within every dataset
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
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage1"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}

# ---------------------------------------------------------------------------
# 1. COMPRESSED-DATA EXCLUSION RULE
# ---------------------------------------------------------------------------
COMPRESSION_KEYWORDS = ["whatsapp", "instagram", "facebook", "telegram", "fb", "insta", "wa", "tg"]
TEST_DATASET_NAMES = ["cocoglide", "columbia", "casia1.0", "casia 1.0", "casia1corel"]

skipped_by_keyword = Counter()


def is_compressed_path(path: str) -> bool:
    """
    Returns True if any part of the path (folder names or file name)
    contains one of the social media compression keywords.
    Keywords: whatsapp, instagram, facebook, telegram, fb, insta, wa, tg.
    """
    p_str = str(path).replace("\\", "/").lower()
    parts = p_str.split("/")
    
    # 1. Check folder components
    for folder in parts[:-1]:
        for kw in ["whatsapp", "instagram", "facebook", "telegram", "insta"]:
            if kw in folder:
                skipped_by_keyword[kw] += 1
                return True
        for kw in ["fb", "wa", "tg"]:
            if folder == kw or folder.startswith(f"{kw}_") or folder.endswith(f"_{kw}") or f"_{kw}_" in folder or f"-{kw}-" in folder:
                skipped_by_keyword[kw] += 1
                return True

    # 2. Check filename component
    fname = parts[-1]
    for kw in ["whatsapp", "instagram", "facebook", "telegram", "insta"]:
        if kw in fname:
            skipped_by_keyword[kw] += 1
            return True
    for kw in ["fb", "wa", "tg"]:
        if re.search(rf"(?:^|[_\-.])(?:{kw}|{kw}_img|img_{kw})(?:[_\-.]|\d|$)", fname):
            skipped_by_keyword[kw] += 1
            return True

    return False


def assert_stage1_purity(file_path: str):
    """Asserts that a path is strictly uncompressed, not a test set image, and not a mask."""
    p_lower = str(file_path).lower().replace("\\", "/")
    
    # Check test set leakage
    for test_name in TEST_DATASET_NAMES:
        if f"/{test_name}/" in p_lower:
            raise ValueError(f"[PROTOCOL VIOLATION] Test image detected in training data: {file_path}")
            
    # Check compression exclusion
    if is_compressed_path(file_path):
        raise ValueError(f"[PROTOCOL VIOLATION] Compressed image detected in Stage 1: {file_path}")
        
    # Check mask bug
    if "_gt." in p_lower or "_mask." in p_lower:
        raise ValueError(f"[MASK BUG VIOLATION] Mask file passed as input image: {file_path}")


# ---------------------------------------------------------------------------
# 2. DATASET LOADERS
# ---------------------------------------------------------------------------

def build_imd2020_stage1(max_folders=80):
    """
    Scans IMD2020 original uncompressed images.
    Folder ID (e.g. 1a07yi) is the grouping key to prevent scene leakage.
    Samples 1 real and 1 fake per folder for 1:1 within-dataset balance.
    """
    imd_base = BASE_DIR / "IMD2020"
    if not imd_base.exists():
        print(f"[-] IMD2020 base not found at {imd_base}")
        return {}
        
    grouped = {}
    folders = sorted([f for f in os.listdir(imd_base) if (imd_base / f).is_dir()])
    
    count = 0
    for folder in folders:
        if count >= max_folders:
            break
        f_dir = imd_base / folder
        if is_compressed_path(str(f_dir)):
            continue
            
        group_id = f"IMD_{folder}"
        files = os.listdir(f_dir)
        
        orig_imgs = [f for f in files if "_orig" in f and Path(f).suffix.lower() in IMG_EXTS]
        tamp_imgs = [f for f in files if "_orig" not in f and "_mask" not in f and Path(f).suffix.lower() in IMG_EXTS]
        
        if not orig_imgs or not tamp_imgs:
            continue
            
        real_f = sorted(orig_imgs)[0]
        fake_f = sorted(tamp_imgs)[0]
        
        real_p = str(f_dir / real_f)
        fake_p = str(f_dir / fake_f)
        
        assert_stage1_purity(real_p)
        assert_stage1_purity(fake_p)
        
        fake_stem = Path(fake_f).stem
        mask_candidates = [f_dir / f"{fake_stem}_mask.png", f_dir / f"{fake_stem}_mask.jpg"]
        mask_p = None
        for mc in mask_candidates:
            if mc.exists():
                mask_p = os.path.relpath(str(mc), BASE_DIR).replace("\\", "/")
                break
                
        grouped[group_id] = [
            {
                "id": f"IMD_{folder}_{real_f}",
                "dataset": "IMD2020",
                "group_id": group_id,
                "image_path": os.path.relpath(real_p, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 0,
                "manipulation": "pristine"
            },
            {
                "id": f"IMD_{folder}_{fake_f}",
                "dataset": "IMD2020",
                "group_id": group_id,
                "image_path": os.path.relpath(fake_p, BASE_DIR).replace("\\", "/"),
                "mask_path": mask_p,
                "label": 1,
                "manipulation": "splicing/photoshop"
            }
        ]
        count += 1
        
    return grouped


def build_dso1_stage1(max_pairs=50):
    """
    Scans DSO-1 uncompressed images (100 real / 100 fake PNGs).
    """
    sml = Path("D:/sample_images_ml")
    real_dir = sml / "DSO-1_real"
    fake_dir = sml / "DSO-1_fake"
    
    if not real_dir.exists() or not fake_dir.exists():
        return {}
        
    grouped = {}
    r_files = sorted([f for f in os.listdir(real_dir) if Path(f).suffix.lower() in IMG_EXTS])[:max_pairs]
    f_files = sorted([f for f in os.listdir(fake_dir) if Path(f).suffix.lower() in IMG_EXTS])[:max_pairs]
    
    for i, (rf, ff) in enumerate(zip(r_files, f_files)):
        group_id = f"DSO1_{i:03d}"
        rp = str(real_dir / rf)
        fp = str(fake_dir / ff)
        
        assert_stage1_purity(rp)
        assert_stage1_purity(fp)
        
        grouped[group_id] = [
            {
                "id": f"DSO1_real_{rf}",
                "dataset": "DSO-1",
                "group_id": group_id,
                "image_path": os.path.relpath(rp, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 0,
                "manipulation": "pristine"
            },
            {
                "id": f"DSO1_fake_{ff}",
                "dataset": "DSO-1",
                "group_id": group_id,
                "image_path": os.path.relpath(fp, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 1,
                "manipulation": "splicing"
            }
        ]
        
    return grouped


def build_inpainting_micro_stage1():
    """
    Scans generative inpainting sets:
    - Beyond the Brush (50)
    - SAGID (25)
    - Normalized fake (25) paired with Normalized real (25) and RAISE training (75).
    Total: 100 inpainting fakes paired 1:1 with 100 pristine photos.
    """
    sml = Path("D:/sample_images_ml")
    grouped = {}
    
    # 1. Normalized pair (25 real, 25 fake)
    nr_dir = sml / "normalized_real"
    nf_dir = sml / "normalized_fake"
    if nr_dir.exists() and nf_dir.exists():
        nr_files = sorted(os.listdir(nr_dir))
        nf_files = sorted(os.listdir(nf_dir))
        for i, (rf, ff) in enumerate(zip(nr_files, nf_files)):
            rp = str(nr_dir / rf)
            fp = str(nf_dir / ff)
            assert_stage1_purity(rp)
            assert_stage1_purity(fp)
            gid = f"NORM_{i:03d}"
            grouped[gid] = [
                {
                    "id": f"NORM_real_{rf}",
                    "dataset": "Normalized",
                    "group_id": gid,
                    "image_path": os.path.relpath(rp, BASE_DIR).replace("\\", "/"),
                    "mask_path": None,
                    "label": 0,
                    "manipulation": "pristine"
                },
                {
                    "id": f"NORM_fake_{ff}",
                    "dataset": "Normalized",
                    "group_id": gid,
                    "image_path": os.path.relpath(fp, BASE_DIR).replace("\\", "/"),
                    "mask_path": None,
                    "label": 1,
                    "manipulation": "generative_inpainting"
                }
            ]
            
    # 2. SAGID and BTB paired with RAISE authentic images
    raise_dir = sml / "RAISE_training"
    raise_files = sorted(os.listdir(raise_dir)) if raise_dir.exists() else []
    
    inpaint_fakes = []
    for sub in ["beyond_the_brush_images", "beyond_the_brush_diverse_images", "sagid_images"]:
        p = sml / sub
        if p.exists():
            for f in sorted(os.listdir(p)):
                if Path(f).suffix.lower() in IMG_EXTS:
                    inpaint_fakes.append((sub, str(p / f)))
                    
    for i, (src_name, fake_p) in enumerate(inpaint_fakes):
        if i >= len(raise_files):
            break
        real_p = str(raise_dir / raise_files[i])
        assert_stage1_purity(fake_p)
        assert_stage1_purity(real_p)
        gid = f"INPAINT_{src_name}_{i:03d}"
        grouped[gid] = [
            {
                "id": f"RAISE_paired_{raise_files[i]}",
                "dataset": "RAISE_Inpainting_Pair",
                "group_id": gid,
                "image_path": os.path.relpath(real_p, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 0,
                "manipulation": "pristine"
            },
            {
                "id": f"{src_name}_{Path(fake_p).name}",
                "dataset": "Generative_Inpainting",
                "group_id": gid,
                "image_path": os.path.relpath(fake_p, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 1,
                "manipulation": "generative_inpainting"
            }
        ]
        
    return grouped


def build_casia2_stage1(max_components=100):
    """
    Scans CASIA 2.0 (Au and Tp) in stage1_data/CASIA2.0.
    Enforces stem-connected-component grouping to prevent donor/base leakage.
    Balances 1:1 within CASIA 2.0.
    Note: CASIA 2.0 images are original camera/edited JPEGs (not social-media compressed).
    """
    casia2_dir = BASE_DIR / "stage1_data" / "CASIA2.0"
    au_dir = casia2_dir / "Au"
    tp_dir = casia2_dir / "Tp"
    
    if not au_dir.exists() or not tp_dir.exists():
        print(f"[-] CASIA 2.0 not found in {casia2_dir}")
        return {}
        
    au_files = sorted([f for f in os.listdir(au_dir) if not f.startswith(".") and f != "Thumbs.db"])
    tp_files = sorted([f for f in os.listdir(tp_dir) if not f.startswith(".") and f != "Thumbs.db"])
    
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

    # Connected components
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

    # Shared components with both Au and Tp
    shared_comps = sorted(list(set(group_to_tp.keys()) & set(group_to_au.keys())))
    
    grouped = {}
    count = 0
    for comp in shared_comps:
        if count >= max_components:
            break
        t_list = group_to_tp[comp]
        a_list = group_to_au[comp]
        if not t_list or not a_list:
            continue
            
        real_f = a_list[0]
        fake_f = t_list[0]
        
        rp = str(au_dir / real_f)
        fp = str(tp_dir / fake_f)
        
        assert_stage1_purity(rp)
        assert_stage1_purity(fp)
        
        gid = f"CASIA2_comp_{comp}"
        grouped[gid] = [
            {
                "id": f"CASIA2_Au_{real_f}",
                "dataset": "CASIA2.0",
                "group_id": gid,
                "image_path": os.path.relpath(rp, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 0,
                "manipulation": "pristine"
            },
            {
                "id": f"CASIA2_Tp_{fake_f}",
                "dataset": "CASIA2.0",
                "group_id": gid,
                "image_path": os.path.relpath(fp, BASE_DIR).replace("\\", "/"),
                "mask_path": None,
                "label": 1,
                "manipulation": "splicing/copymove"
            }
        ]
        count += 1
        
    return grouped


def build_stage1_manifests(val_ratio=0.20, seed=42):
    """
    Integrates all stage 1 datasets, enforces source-level grouping,
    verifies image integrity, and builds train and validation manifests.
    """
    random.seed(seed)
    
    print("=" * 80)
    print(" BUILDING STAGE 1 MANIFESTS (UNCOMPRESSED ONLY)")
    print("=" * 80)
    
    all_grouped = {}
    
    # 1. IMD2020
    imd_grouped = build_imd2020_stage1(max_folders=30)
    print(f"[*] IMD2020: {len(imd_grouped)} balanced groups ({len(imd_grouped)*2} images)")
    all_grouped.update(imd_grouped)
    
    # 2. DSO-1
    dso_grouped = build_dso1_stage1(max_pairs=20)
    print(f"[*] DSO-1: {len(dso_grouped)} balanced groups ({len(dso_grouped)*2} images)")
    all_grouped.update(dso_grouped)
    
    # 3. Inpainting sets (all 40 pairs)
    inpaint_grouped = build_inpainting_micro_stage1()
    # Take first 40 balanced groups to balance representation
    inpaint_subset = {k: inpaint_grouped[k] for k in sorted(list(inpaint_grouped.keys()))[:40]}
    print(f"[*] Inpainting Micro-Sets: {len(inpaint_subset)} balanced groups ({len(inpaint_subset)*2} images)")
    all_grouped.update(inpaint_subset)
    
    # 4. CASIA 2.0
    casia2_grouped = build_casia2_stage1(max_components=35)
    print(f"[*] CASIA 2.0: {len(casia2_grouped)} balanced groups ({len(casia2_grouped)*2} images)")
    all_grouped.update(casia2_grouped)
    
    # Split by group ID (zero leakage)
    all_group_ids = sorted(list(all_grouped.keys()))
    random.shuffle(all_group_ids)
    
    n_val = int(len(all_group_ids) * val_ratio)
    val_groups = set(all_group_ids[:n_val])
    train_groups = set(all_group_ids[n_val:])
    
    train_samples = []
    val_samples = []
    
    for gid in train_groups:
        train_samples.extend(all_grouped[gid])
    for gid in val_groups:
        val_samples.extend(all_grouped[gid])
        
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    print("\n--- Verifying Image Integrity and Enforcing Stage 1 Rules ---")
    for s in train_samples + val_samples:
        full_p = BASE_DIR / s["image_path"]
        assert_stage1_purity(str(full_p))
        if not full_p.exists():
            raise FileNotFoundError(f"Missing file: {full_p}")
        try:
            with Image.open(full_p) as im:
                im.verify()
        except Exception as e:
            raise ValueError(f"Corrupted image cannot open: {full_p} ({e})")
            
    # Save manifests
    with open(SPLITS_DIR / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(SPLITS_DIR / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
        
    # Also save pilot manifests (exact 20 real / 20 fake in train, 5 real / 5 fake in val)
    train_reals = [s for s in train_samples if s["label"] == 0][:20]
    train_fakes = [s for s in train_samples if s["label"] == 1][:20]
    train_pilot = train_reals + train_fakes
    random.shuffle(train_pilot)

    val_reals = [s for s in val_samples if s["label"] == 0][:5]
    val_fakes = [s for s in val_samples if s["label"] == 1][:5]
    val_pilot = val_reals + val_fakes
    random.shuffle(val_pilot)

    with open(SPLITS_DIR / "train_pilot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_pilot, f, indent=2)
    with open(SPLITS_DIR / "val_pilot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_pilot, f, indent=2)
        
    print("\n" + "=" * 80)
    print(" MANIFEST SUMMARY & CLASS BALANCE")
    print("=" * 80)
    
    t_counts = Counter((s["dataset"], s["label"]) for s in train_samples)
    v_counts = Counter((s["dataset"], s["label"]) for s in val_samples)
    
    print(f"TRAIN SAMPLES: {len(train_samples)} total | Real={sum(1 for s in train_samples if s['label']==0)}, Fake={sum(1 for s in train_samples if s['label']==1)}")
    for k, cnt in sorted(t_counts.items()):
        lbl = "Real" if k[1] == 0 else "Fake"
        print(f"  {k[0]:<25} ({lbl}): {cnt}")
        
    print(f"\nVAL SAMPLES:   {len(val_samples)} total | Real={sum(1 for s in val_samples if s['label']==0)}, Fake={sum(1 for s in val_samples if s['label']==1)}")
    for k, cnt in sorted(v_counts.items()):
        lbl = "Real" if k[1] == 0 else "Fake"
        print(f"  {k[0]:<25} ({lbl}): {cnt}")
        
    print("\n--- COMPRESSION EXCLUSION AUDIT ---")
    print(f"Total compressed paths skipped: {sum(skipped_by_keyword.values())}")
    for kw, cnt in skipped_by_keyword.most_common():
        print(f"  Keyword '{kw}': {cnt} paths skipped")
        
    print("\n[*] Note: CASIA 2.0 is JPEG-based ('original camera/edited', not simulator-compressed), recorded for Stage 1 report.")
    return train_samples, val_samples


if __name__ == "__main__":
    build_stage1_manifests()
