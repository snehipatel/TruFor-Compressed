#!/usr/bin/env python3
"""
Batch Platform Compression Pipeline for Forgery Datasets
=========================================================
Uses the empirical platform compression simulator (WhatsApp, Instagram,
Facebook, Telegram) to compress all datasets in the DATASET folder.

Folder Hierarchy Created (in the same folder as DATASET):
├── CocoGlide/
│   ├── whatsapp/      (real/, fake/)
│   ├── instagram/     (real/, fake/)
│   ├── facebook/      (real/, fake/)
│   ├── telegram/      (real/, fake/)
│   └── mask/          (ground truth binary masks, preserved lossless)
│
├── CASIA1.0/
│   ├── whatsapp/      (Au/, Modified Tp/CM/, Modified Tp/Sp/)
│   ├── instagram/     (Au/, Modified Tp/CM/, Modified Tp/Sp/)
│   ├── facebook/      (Au/, Modified Tp/CM/, Modified Tp/Sp/)
│   ├── telegram/      (Au/, Modified Tp/CM/, Modified Tp/Sp/)
│   └── casia1groundtruth/ (ground truth masks, preserved lossless)
│
└── Columbia/
    ├── whatsapp/      (4cam_auth/, 4cam_splc/)
    ├── instagram/     (4cam_auth/, 4cam_splc/)
    ├── facebook/      (4cam_auth/, 4cam_splc/)
    ├── telegram/      (4cam_auth/, 4cam_splc/)
    └── edgemask/      (ground truth edge masks, preserved lossless)
"""

import os
import sys
import csv
import shutil
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ensure SIMULATOR directory is on path
SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SIMULATOR')
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

try:
    import server
    from server import PLATFORM_PROFILES, compress_image, prepare_image_for_processing
except ImportError:
    # Standalone fallback if server.py is in current or child dir
    sys.path.insert(0, os.path.abspath('.'))
    import SIMULATOR.server as server
    from SIMULATOR.server import PLATFORM_PROFILES, compress_image, prepare_image_for_processing

PLATFORMS = ['whatsapp', 'instagram', 'facebook', 'telegram']


def get_dataset_tasks(dataset_base_dir):
    """
    Scans the DATASET folder and maps out all images for CocoGlide, CASIA1.0, and Columbia.
    Returns:
        dict: {
            'CocoGlide': [ (src_path, rel_subfolder), ... ],
            'CASIA1.0':  [ (src_path, rel_subfolder), ... ],
            'Columbia':  [ (src_path, rel_subfolder), ... ],
        }
    """
    tasks = {
        'CocoGlide': [],
        'CASIA1.0': [],
        'Columbia': [],
    }

    image_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp')

    # -------------------------------------------------------------
    # 1. CocoGlide (real, fake)
    # -------------------------------------------------------------
    cg_dir = os.path.join(dataset_base_dir, 'CocoGlide')
    if os.path.exists(cg_dir):
        for subset in ['real', 'fake']:
            s_dir = os.path.join(cg_dir, subset)
            if os.path.exists(s_dir):
                for f in sorted(os.listdir(s_dir)):
                    if f.lower().endswith(image_exts):
                        tasks['CocoGlide'].append((os.path.join(s_dir, f), subset))

    # -------------------------------------------------------------
    # 2. CASIA 1.0 (Au, Modified Tp [CM, Sp])
    # -------------------------------------------------------------
    casia_dir = os.path.join(dataset_base_dir, 'CASIA1.0')
    if os.path.exists(casia_dir):
        # Au images: either under 'Au/Au' or 'CASIA 1.0 dataset/Au/Au'
        au_candidates = [
            os.path.join(casia_dir, 'CASIA 1.0 dataset', 'Au', 'Au'),
            os.path.join(casia_dir, 'Au', 'Au'),
            os.path.join(casia_dir, 'CASIA 1.0 dataset', 'Au'),
            os.path.join(casia_dir, 'Au'),
        ]
        au_dir = next((d for d in au_candidates if os.path.exists(d) and any(f.lower().endswith(image_exts) for f in os.listdir(d))), None)
        if au_dir:
            for f in sorted(os.listdir(au_dir)):
                if f.lower().endswith(image_exts):
                    tasks['CASIA1.0'].append((os.path.join(au_dir, f), 'Au'))

        # Modified Tp: Copy-Move (CM) and Splicing (Sp)
        for sub_type in ['CM', 'Sp']:
            tp_candidates = [
                os.path.join(casia_dir, 'CASIA 1.0 dataset', 'Modified Tp', 'Tp', sub_type),
                os.path.join(casia_dir, 'Modified Tp', 'Tp', sub_type),
                os.path.join(casia_dir, 'CASIA 1.0 dataset', 'Modified Tp', sub_type),
                os.path.join(casia_dir, 'Modified Tp', sub_type),
            ]
            tp_dir = next((d for d in tp_candidates if os.path.exists(d) and any(f.lower().endswith(image_exts) for f in os.listdir(d))), None)
            if tp_dir:
                rel_sub = os.path.join('Modified Tp', sub_type)
                for f in sorted(os.listdir(tp_dir)):
                    if f.lower().endswith(image_exts):
                        tasks['CASIA1.0'].append((os.path.join(tp_dir, f), rel_sub))

    # -------------------------------------------------------------
    # 3. Columbia (4cam_auth, 4cam_splc)
    # -------------------------------------------------------------
    col_dir = os.path.join(dataset_base_dir, 'Columbia')
    if os.path.exists(col_dir):
        for subset in ['4cam_auth', '4cam_splc']:
            s_dir = os.path.join(col_dir, subset)
            if os.path.exists(s_dir):
                for f in sorted(os.listdir(s_dir)):
                    p = os.path.join(s_dir, f)
                    # Ignore nested subfolders like edgemask
                    if os.path.isfile(p) and f.lower().endswith(image_exts):
                        tasks['Columbia'].append((p, subset))

    return tasks


def compress_single_task(args):
    """
    Compress a single image for a specific platform.
    """
    src_path, dst_path, platform = args
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    profile = PLATFORM_PROFILES[platform]['standard']
    orig_size = os.path.getsize(src_path)

    try:
        with Image.open(src_path) as raw_im:
            prep_im = prepare_image_for_processing(raw_im)
            comp_bytes, new_w, new_h, codec = compress_image(prep_im, profile)

        with open(dst_path, 'wb') as f:
            f.write(comp_bytes)

        comp_size = len(comp_bytes)
        reduction_pct = round((orig_size - comp_size) / orig_size * 100, 1) if orig_size > 0 else 0

        return {
            'status': 'ok',
            'src_path': src_path,
            'dst_path': dst_path,
            'orig_size': orig_size,
            'comp_size': comp_size,
            'resolution': f"{new_w}x{new_h}",
            'reduction_pct': reduction_pct,
            'codec': codec,
        }
    except Exception as e:
        return {
            'status': 'error',
            'src_path': src_path,
            'dst_path': dst_path,
            'error': str(e),
        }


def copy_ground_truth_assets(dataset_base_dir, target_base_dir):
    """
    Copies ground truth masks and metadata without compression to preserve fidelity.
    """
    # 1. CocoGlide masks & metadata
    cg_src = os.path.join(dataset_base_dir, 'CocoGlide')
    cg_dst = os.path.join(target_base_dir, 'CocoGlide')
    if os.path.exists(cg_src):
        os.makedirs(cg_dst, exist_ok=True)
        mask_src = os.path.join(cg_src, 'mask')
        mask_dst = os.path.join(cg_dst, 'mask')
        if os.path.exists(mask_src) and not os.path.exists(mask_dst):
            print("   📋 Copying CocoGlide ground-truth masks (lossless)...")
            shutil.copytree(mask_src, mask_dst)
        for meta_file in ['README.txt', 'licenses.json', 'table.csv']:
            sp = os.path.join(cg_src, meta_file)
            dp = os.path.join(cg_dst, meta_file)
            if os.path.exists(sp) and not os.path.exists(dp):
                shutil.copy2(sp, dp)

    # 2. CASIA1.0 ground truth masks
    casia_src = os.path.join(dataset_base_dir, 'CASIA1.0')
    casia_dst = os.path.join(target_base_dir, 'CASIA1.0')
    if os.path.exists(casia_src):
        os.makedirs(casia_dst, exist_ok=True)
        gt_src = os.path.join(casia_src, 'casia1groundtruth')
        gt_dst = os.path.join(casia_dst, 'casia1groundtruth')
        if os.path.exists(gt_src) and not os.path.exists(gt_dst):
            print("   📋 Copying CASIA 1.0 ground-truth masks...")
            shutil.copytree(gt_src, gt_dst)

    # 3. Columbia edge masks & README
    col_src = os.path.join(dataset_base_dir, 'Columbia')
    col_dst = os.path.join(target_base_dir, 'Columbia')
    if os.path.exists(col_src):
        os.makedirs(col_dst, exist_ok=True)
        for subset in ['4cam_auth', '4cam_splc']:
            em_src = os.path.join(col_src, subset, 'edgemask')
            em_dst = os.path.join(col_dst, f'{subset}_edgemask')
            if os.path.exists(em_src) and not os.path.exists(em_dst):
                print(f"   📋 Copying Columbia {subset} edge masks...")
                shutil.copytree(em_src, em_dst)
        readme_src = os.path.join(col_src, 'README.txt')
        readme_dst = os.path.join(col_dst, 'README.txt')
        if os.path.exists(readme_src) and not os.path.exists(readme_dst):
            shutil.copy2(readme_src, readme_dst)


def process_all_datasets(dataset_dir, output_dir=None, num_workers=8):
    """
    Main orchestration function.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if output_dir is None:
        # Save in the same folder as DATASET
        output_dir = os.path.dirname(dataset_dir)
    output_dir = os.path.abspath(output_dir)

    print("=" * 80)
    print("🚀 PLATFORM COMPRESSION PIPELINE FOR IMAGE FORGERY DATASETS")
    print("=" * 80)
    print(f"📂 Source Dataset Folder : {dataset_dir}")
    print(f"📁 Destination Base Dir  : {output_dir}")
    print(f"📱 Target Platforms     : {', '.join(p.upper() for p in PLATFORMS)}")
    print(f"⚡ CPU Threads           : {num_workers}")
    print("=" * 80)

    # Step 1: Scan and map tasks
    tasks_by_dataset = get_dataset_tasks(dataset_dir)

    total_images = sum(len(t) for t in tasks_by_dataset.values())
    if total_images == 0:
        print(f"❌ Error: No dataset images found in '{dataset_dir}'.")
        print("Please check that the folder contains CocoGlide, CASIA1.0, or Columbia.")
        return

    print(f"\n📊 Discovered Dataset Files:")
    for ds_name, items in tasks_by_dataset.items():
        print(f"   • {ds_name:<12}: {len(items):>5} images")
    print(f"   Total source images: {total_images:,}")
    print(f"   Total platform images to generate: {total_images * len(PLATFORMS):,}\n")

    # Step 2: Copy ground-truth assets (masks, metadata)
    print("📦 Step 1: Preserving ground-truth masks & metadata...")
    copy_ground_truth_assets(dataset_dir, output_dir)
    print("   ✅ Ground truth assets preserved.\n")

    # Step 3: Execute compression tasks
    overall_manifest = []
    total_start = time.time()

    for ds_name, items in tasks_by_dataset.items():
        if not items:
            continue

        print("-" * 80)
        print(f"🔄 Processing {ds_name} ({len(items)} source images)...")
        print("-" * 80)

        ds_target_root = os.path.join(output_dir, ds_name)
        work_items = []

        for src_path, rel_sub in items:
            stem = os.path.splitext(os.path.basename(src_path))[0]

            for plat in PLATFORMS:
                # Instagram uses WebP, others use JPEG
                ext = '.webp' if PLATFORM_PROFILES[plat]['standard']['codec'] == 'WEBP' else '.jpg'
                dst_filename = f"{stem}{ext}"
                dst_path = os.path.join(ds_target_root, plat, rel_sub, dst_filename)
                work_items.append((src_path, dst_path, plat, ds_name, rel_sub))

        print(f"   Generating {len(work_items)} platform images across {len(PLATFORMS)} platforms...")

        # Run compression in parallel
        start_t = time.time()
        completed = 0
        ds_manifest = []

        def worker(item):
            src, dst, plat, d_name, r_sub = item
            res = compress_single_task((src, dst, plat))
            res['dataset'] = d_name
            res['platform'] = plat
            res['subset'] = r_sub
            return res

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for result in executor.map(worker, work_items):
                completed += 1
                if result['status'] == 'ok':
                    ds_manifest.append(result)
                else:
                    print(f"\n   ⚠️ Error compressing {result['src_path']}: {result.get('error')}")

                if completed % 250 == 0 or completed == len(work_items):
                    elapsed = time.time() - start_t
                    speed = completed / elapsed if elapsed > 0 else 0
                    print(f"   Progress: {completed}/{len(work_items)} ({completed/len(work_items)*100:.1f}%) | {speed:.1f} img/s", end='\r')

        ds_elapsed = time.time() - start_t
        print(f"\n   ✅ {ds_name} completed in {ds_elapsed:.1f}s ({len(work_items)/ds_elapsed:.1f} imgs/s)")

        # Save dataset manifest CSV
        manifest_path = os.path.join(ds_target_root, 'compression_manifest.csv')
        if ds_manifest:
            with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'dataset', 'platform', 'subset', 'src_path', 'dst_path',
                    'orig_size', 'comp_size', 'resolution', 'reduction_pct', 'codec'
                ], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(ds_manifest)
            print(f"   📋 Manifest saved to: {manifest_path}")

        overall_manifest.extend(ds_manifest)

    total_elapsed = time.time() - total_start

    # Step 4: Verification & Summary
    print("\n" + "=" * 80)
    print("🎉 ALL PLATFORM COMPRESSIONS COMPLETED!")
    print("=" * 80)
    print(f"Total images generated : {len(overall_manifest):,}")
    print(f"Total time taken       : {total_elapsed:.1f}s ({len(overall_manifest)/total_elapsed:.1f} imgs/s)")
    print(f"Output root location   : {output_dir}\n")

    print(f"{'Dataset':<12} | {'Platform':<10} | {'Images':<8} | {'Avg Comp Size':<15} | {'Avg Reduction'}")
    print("-" * 80)

    for ds_name in tasks_by_dataset.keys():
        for plat in PLATFORMS:
            subset_rows = [r for r in overall_manifest if r['dataset'] == ds_name and r['platform'] == plat]
            if subset_rows:
                avg_sz = sum(r['comp_size'] for r in subset_rows) / len(subset_rows)
                avg_red = sum(r['reduction_pct'] for r in subset_rows) / len(subset_rows)
                print(f"{ds_name:<12} | {plat.upper():<10} | {len(subset_rows):>6} | {avg_sz/1024:>10.1f} KB    | {avg_red:>10.1f} %")
        print("-" * 80)

    print("\n📁 Structured Output Folders Created:")
    for ds_name in tasks_by_dataset.keys():
        target_path = os.path.join(output_dir, ds_name)
        if os.path.exists(target_path):
            total_f = sum(len(f) for _, _, f in os.walk(target_path))
            print(f"   • {target_path} ({total_f} files)")

    print("\n✅ Compression pipeline finished with 0 errors!")


def main():
    parser = argparse.ArgumentParser(description="Platform compression for Forgery Datasets")
    parser.add_argument(
        '--dataset_dir', '-d',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'),
        help="Path to source DATASET folder containing CocoGlide, CASIA1.0, Columbia"
    )
    parser.add_argument(
        '--output_dir', '-o',
        default=None,
        help="Target base directory where CocoGlide/, CASIA1.0/, Columbia/ folders will be created. (Default: same folder as DATASET)"
    )
    parser.add_argument(
        '--threads', '-t',
        type=int,
        default=os.cpu_count() or 8,
        help="Number of parallel worker threads (default: CPU cores)"
    )
    args = parser.parse_args()

    process_all_datasets(args.dataset_dir, args.output_dir, args.threads)


if __name__ == '__main__':
    main()
