"""
regenerate_compressed_sets_v3.py
Batch Regeneration Script for Simulator v3 Compressed Datasets
==============================================================
Rebuilds social-media compressed datasets using Simulator v3:
- WhatsApp: Always HD tables, baseline JPEG (progressive=False), 1600px cap.
- Facebook: Width cap at 1080px, Bilinear downsampling, progressive JPEG.
- Telegram: 1279px max dimension, progressive JPEG.
- Instagram: WebP quality 60, BICUBIC interpolation, upscale_small=True, 1440px cap.

Writes all regenerated images strictly to NEW target directories:
`results/simulated_v3/<dataset>/<platform>/`.
Never overwrites original datasets, v1, or v2 directories.

Usage:
    python regenerate_compressed_sets_v3.py --platform all --dataset all
    python regenerate_compressed_sets_v3.py --estimate-only
"""

import os
import io
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

PROJECT_ROOT = Path(r"D:\ml_project")
V3_DIR = PROJECT_ROOT / "simulator_v3"
sys.path.insert(0, str(V3_DIR))

import server as sim_v3

DATASET_BASE = PROJECT_ROOT / "DATASET"
OUTPUT_BASE = PROJECT_ROOT / "results" / "simulated_v3"

PLATFORMS = ["whatsapp", "facebook", "telegram", "instagram"]
DATASETS = ["CASIA1.0", "CocoGlide", "Columbia"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

OLD_SET_EXPECTED_COUNTS = {
    "CASIA1.0": 1721,
    "CocoGlide": 1024,
    "Columbia": 363,
}


def get_dataset_tasks(dataset_name):
    """
    Discovers source images and relative destinations for each dataset
    without scanning existing compressed folders.
    """
    tasks = []  # list of (src_path, rel_subfolder)
    
    if dataset_name == "CocoGlide":
        cg_dir = DATASET_BASE / "CocoGlide"
        for subset in ["real", "fake"]:
            s_dir = cg_dir / subset
            if s_dir.exists():
                for f in sorted(os.listdir(s_dir)):
                    if Path(f).suffix.lower() in IMAGE_EXTS:
                        tasks.append((s_dir / f, subset))
                        
    elif dataset_name == "CASIA1.0":
        casia_dir = DATASET_BASE / "CASIA1.0"
        # Au images:
        au_candidates = [
            casia_dir / "CASIA 1.0 dataset" / "Au" / "Au",
            casia_dir / "Au" / "Au",
            casia_dir / "CASIA 1.0 dataset" / "Au",
            casia_dir / "Au",
        ]
        au_dir = next((d for d in au_candidates if d.exists() and any(Path(f).suffix.lower() in IMAGE_EXTS for f in os.listdir(d))), None)
        if au_dir:
            for f in sorted(os.listdir(au_dir)):
                if Path(f).suffix.lower() in IMAGE_EXTS:
                    tasks.append((au_dir / f, "Au"))
                    
        # Modified Tp: Copy-Move (CM) and Splicing (Sp)
        for sub_type in ["CM", "Sp"]:
            tp_candidates = [
                casia_dir / "CASIA 1.0 dataset" / "Modified Tp" / "Tp" / sub_type,
                casia_dir / "Modified Tp" / "Tp" / sub_type,
                casia_dir / "CASIA 1.0 dataset" / "Modified Tp" / sub_type,
                casia_dir / "Modified Tp" / "Tp",
                casia_dir / "CASIA 1.0 dataset" / "Tp" / sub_type,
            ]
            tp_dir = next((d for d in tp_candidates if d.exists() and any(Path(f).suffix.lower() in IMAGE_EXTS for f in os.listdir(d))), None)
            if tp_dir:
                rel_sub = os.path.join("Modified Tp", sub_type)
                for f in sorted(os.listdir(tp_dir)):
                    if Path(f).suffix.lower() in IMAGE_EXTS:
                        tasks.append((tp_dir / f, rel_sub))
                        
    elif dataset_name == "Columbia":
        col_dir = DATASET_BASE / "Columbia"
        for subset in ["4cam_auth", "4cam_splc"]:
            s_dir = col_dir / subset
            if s_dir.exists():
                for f in sorted(os.listdir(s_dir)):
                    p = s_dir / f
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                        tasks.append((p, subset))
                        
    return tasks


def compress_worker(args):
    src_path, dst_path, platform_name = args
    profile = sim_v3.PLATFORM_PROFILES[platform_name]["standard"]
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return True, src_path.name
        
    try:
        with Image.open(src_path) as raw_im:
            comp_bytes, new_w, new_h, codec = sim_v3.compress_image(raw_im, profile)
        with open(dst_path, "wb") as f:
            f.write(comp_bytes)
        return True, src_path.name
    except Exception as e:
        return False, f"{src_path.name}: {e}"


def run_regeneration(platforms, datasets, max_workers=8, dry_run=False):
    print("=" * 80)
    print("STARTING SIMULATOR v3 FULL BATCH DATASET REGENERATION")
    print("=" * 80)
    print(f"Target Output Directory: {OUTPUT_BASE}")
    print(f"Platforms: {platforms}")
    print(f"Datasets: {datasets}")
    print(f"Worker Threads: {max_workers}")
    print("-" * 80)
    
    total_images_all = 0
    start_time_all = time.time()
    verification_results = {}
    
    for ds_name in datasets:
        tasks = get_dataset_tasks(ds_name)
        expected_cnt = OLD_SET_EXPECTED_COUNTS.get(ds_name, len(tasks))
        print(f"\n[*] Dataset: {ds_name} — Discovered {len(tasks)} source images (Expected old set count: {expected_cnt})")
        
        if len(tasks) != expected_cnt:
            print(f"  [WARNING] Task count {len(tasks)} differs from expected old count {expected_cnt}!")
            
        for plat in platforms:
            ext = ".webp" if plat == "instagram" else ".jpg"
            target_dir = OUTPUT_BASE / ds_name / plat
            
            job_args = []
            for src_p, rel_sub in tasks:
                dst_p = target_dir / rel_sub / f"{src_p.stem}{ext}"
                job_args.append((src_p, dst_p, plat))
                
            if dry_run:
                print(f"  [DRY-RUN] {ds_name} -> {plat.upper()}: {len(job_args)} files to {target_dir}")
                continue
                
            t0 = time.time()
            success_cnt = 0
            fail_cnt = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(compress_worker, arg) for arg in job_args]
                for fut in as_completed(futures):
                    ok, msg = fut.result()
                    if ok:
                        success_cnt += 1
                    else:
                        fail_cnt += 1
                        print(f"    [Error] {msg}")
                        
            dur = time.time() - t0
            speed = success_cnt / max(0.001, dur)
            print(f"  [COMPLETE] {ds_name} -> {plat.upper()}: {success_cnt}/{len(job_args)} created in {dur:.2f}s ({speed:.1f} img/s)")
            total_images_all += success_cnt
            
            # Count output files on disk to verify
            actual_count = sum(1 for root, _, files in os.walk(target_dir) for f in files if Path(f).suffix.lower() in IMAGE_EXTS)
            match = (actual_count == expected_cnt)
            verification_results[(ds_name, plat)] = {
                "expected": expected_cnt,
                "actual": actual_count,
                "match": match
            }
            
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE — VERIFICATION SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Dataset':12s} | {'Platform':12s} | {'Expected (Old)':15s} | {'Actual (v3)':15s} | {'Count Match?':12s}")
    print("-" * 80)
    all_matched = True
    for (ds_name, plat), info in verification_results.items():
        matched_str = "MATCH (OK)" if info["match"] else "MISMATCH (FAIL)"
        if not info["match"]:
            all_matched = False
        print(f"{ds_name:12s} | {plat:12s} | {info['expected']:15d} | {info['actual']:15d} | {matched_str:12s}")
        
    total_dur = time.time() - start_time_all
    print("-" * 80)
    print(f"Total Regenerated Files: {total_images_all} in {total_dur:.1f}s ({total_dur/60:.2f} minutes)")
    print(f"All Counts Match Old Sets: {all_matched}")
    print("=" * 80)
    return all_matched


def main():
    parser = argparse.ArgumentParser(description="Simulator v3 Full Dataset Regeneration Tool")
    parser.add_argument("--platform", choices=PLATFORMS + ["all"], default="all")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool worker count")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing files")
    
    args = parser.parse_args()
    
    plats = PLATFORMS if args.platform == "all" else [args.platform]
    dsets = DATASETS if args.dataset == "all" else [args.dataset]
    
    run_regeneration(plats, dsets, max_workers=args.workers, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
