"""
Generate compressed versions of all test datasets using the platform simulator.
Outputs: DATASET/{CocoGlide,Columbia,CASIA1.0}/{whatsapp,instagram,facebook,telegram}/...
"""
import os
import sys
import time
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

BASE_DIR = Path(__file__).resolve().parent.parent
SIM_DIR = BASE_DIR / "SIMULATOR"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from server import PLATFORM_PROFILES, compress_image, prepare_image_for_processing

PLATFORMS = ['whatsapp', 'instagram', 'facebook', 'telegram']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}
DATASET_DIR = BASE_DIR / "DATASET"


def compress_one(src_path, dst_path, platform):
    """Compress a single image for a given platform."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    profile = PLATFORM_PROFILES[platform]['standard']
    try:
        with Image.open(src_path) as raw_im:
            prep_im = prepare_image_for_processing(raw_im)
            comp_bytes, new_w, new_h, codec = compress_image(prep_im, profile)
        with open(dst_path, 'wb') as f:
            f.write(comp_bytes)
        return True
    except Exception as e:
        print(f"  ERROR compressing {src_path}: {e}")
        return False


def get_output_ext(platform):
    """Instagram -> .webp, others -> .jpg"""
    codec = PLATFORM_PROFILES[platform]['standard'].get('codec', 'JPEG')
    return '.webp' if codec == 'WEBP' else '.jpg'


def generate_cocoglide():
    """Generate compressed CocoGlide test set."""
    src_dir = DATASET_DIR / "CocoGlide"
    tasks = []
    for subset in ["real", "fake"]:
        s_dir = src_dir / subset
        if not s_dir.exists():
            continue
        for f in sorted(os.listdir(s_dir)):
            if Path(f).suffix.lower() in IMG_EXTS:
                src = str(s_dir / f)
                stem = Path(f).stem
                for p in PLATFORMS:
                    ext = get_output_ext(p)
                    dst = str(DATASET_DIR / "CocoGlide" / p / subset / f"{stem}{ext}")
                    if not os.path.exists(dst):
                        tasks.append((src, dst, p))
    return tasks


def generate_columbia():
    """Generate compressed Columbia test set."""
    src_dir = DATASET_DIR / "Columbia"
    tasks = []
    for subset in ["4cam_auth", "4cam_splc"]:
        s_dir = src_dir / subset
        if not s_dir.exists():
            continue
        for f in sorted(os.listdir(s_dir)):
            fp = s_dir / f
            if fp.is_file() and Path(f).suffix.lower() in IMG_EXTS:
                src = str(fp)
                stem = Path(f).stem
                for p in PLATFORMS:
                    ext = get_output_ext(p)
                    dst = str(DATASET_DIR / "Columbia" / p / subset / f"{stem}{ext}")
                    if not os.path.exists(dst):
                        tasks.append((src, dst, p))
    return tasks


def generate_casia10():
    """Generate compressed CASIA 1.0 test set."""
    src_base = DATASET_DIR / "CASIA1.0" / "CASIA 1.0 dataset"
    tasks = []
    
    # Au images
    au_dir = src_base / "Au" / "Au"
    if au_dir.exists():
        for f in sorted(os.listdir(au_dir)):
            if Path(f).suffix.lower() in IMG_EXTS:
                src = str(au_dir / f)
                stem = Path(f).stem
                for p in PLATFORMS:
                    ext = get_output_ext(p)
                    dst = str(DATASET_DIR / "CASIA1.0" / p / "Au" / f"{stem}{ext}")
                    if not os.path.exists(dst):
                        tasks.append((src, dst, p))
    
    # Modified Tp (CM and Sp)
    for sub in ["CM", "Sp"]:
        tp_dir = src_base / "Modified Tp" / "Tp" / sub
        if tp_dir.exists():
            for f in sorted(os.listdir(tp_dir)):
                if Path(f).suffix.lower() in IMG_EXTS:
                    src = str(tp_dir / f)
                    stem = Path(f).stem
                    for p in PLATFORMS:
                        ext = get_output_ext(p)
                        dst = str(DATASET_DIR / "CASIA1.0" / p / "Modified Tp" / sub / f"{stem}{ext}")
                        if not os.path.exists(dst):
                            tasks.append((src, dst, p))
    return tasks


def worker(args):
    return compress_one(*args)


def main():
    print("=" * 80)
    print(" GENERATING COMPRESSED TEST SETS FOR STAGE 2")
    print("=" * 80)
    
    all_tasks = []
    
    cg_tasks = generate_cocoglide()
    print(f"CocoGlide:  {len(cg_tasks)} images to compress")
    all_tasks.extend(cg_tasks)
    
    col_tasks = generate_columbia()
    print(f"Columbia:   {len(col_tasks)} images to compress")
    all_tasks.extend(col_tasks)
    
    c1_tasks = generate_casia10()
    print(f"CASIA 1.0:  {len(c1_tasks)} images to compress")
    all_tasks.extend(c1_tasks)
    
    if not all_tasks:
        print("All compressed test sets already exist!")
        return
    
    print(f"\nTotal: {len(all_tasks)} images to compress")
    print(f"Using {min(8, os.cpu_count() or 4)} threads\n")
    
    t0 = time.time()
    done = 0
    errors = 0
    
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        for ok in executor.map(worker, all_tasks):
            done += 1
            if not ok:
                errors += 1
            if done % 200 == 0 or done == len(all_tasks):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                print(f"  Progress: {done}/{len(all_tasks)} ({done/len(all_tasks)*100:.1f}%) | {speed:.1f} img/s | Errors: {errors}")
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s | {done} processed | {errors} errors")
    
    # Verify counts
    print("\n--- Verification ---")
    for ds in ["CocoGlide", "Columbia", "CASIA1.0"]:
        for p in PLATFORMS:
            d = DATASET_DIR / ds / p
            if d.exists():
                cnt = sum(1 for _ in d.rglob("*") if _.is_file())
                print(f"  {ds}/{p}: {cnt} files")
            else:
                print(f"  {ds}/{p}: MISSING")


if __name__ == "__main__":
    main()
