"""
run_trufor_downstream_check.py
Optimized TruFor Downstream Evaluation on CocoGlide (512x512 images):
- Evaluates 30 real + 30 fake CocoGlide images under Simulator v1 vs Simulator v2 across platforms.
- Compares against ground truth Real-Platform distributions from evaluation_metrics_all.csv.
- Saves scores incrementally to JSON cache.
"""

import os
import io
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(r"D:\ml_project")
SRC_DIR = PROJECT_ROOT / "test_docker" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
from config import _C as config, update_config

# Paths
COCO_ROOT = PROJECT_ROOT / "DATASET" / "CocoGlide"
CACHE_FILE = PROJECT_ROOT / "simulator_audit" / "per_platform" / "trufor_downstream_cache.json"
OUTPUT_FILE = PROJECT_ROOT / "simulator_audit" / "per_platform" / "trufor_downstream_results.json"
PROGRESS_FILE = PROJECT_ROOT / "simulator_audit" / "PROGRESS.md"

# Q-Tables for v1 and v2
WHATSAPP_HD_QTABLES = {
    0: [3, 2, 2, 3, 4, 6, 8, 10, 2, 2, 2, 3, 4, 9, 10, 9, 2, 2, 3, 4, 6, 9, 11, 9, 2, 3, 4, 5, 8, 14, 13, 10, 3, 4, 6, 9, 11, 17, 16, 12, 4, 6, 9, 10, 13, 17, 18, 15, 8, 10, 12, 14, 16, 19, 19, 16, 12, 15, 15, 16, 18, 16, 16, 16],
    1: [3, 3, 4, 8, 16, 16, 16, 16, 3, 3, 4, 11, 16, 16, 16, 16, 4, 4, 9, 16, 16, 16, 16, 16, 8, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
}
WHATSAPP_STANDARD_QTABLES = {
    0: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 6, 8, 10, 12, 17, 25, 36, 54, 7, 11, 12, 16, 21, 30, 42, 62, 10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95, 22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167],
    1: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 6, 8, 10, 12, 17, 25, 36, 54, 7, 11, 12, 16, 21, 30, 42, 62, 10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95, 22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167]
}
FACEBOOK_QTABLES_SIM = {
    0: [9, 8, 16, 16, 23, 24, 26, 27, 8, 11, 15, 17, 24, 22, 28, 28, 16, 15, 18, 20, 24, 27, 26, 29, 16, 17, 20, 23, 29, 31, 31, 32, 23, 24, 24, 29, 32, 34, 38, 38, 24, 22, 27, 31, 34, 38, 35, 42, 26, 28, 26, 31, 38, 35, 38, 37, 27, 28, 29, 32, 38, 42, 37, 30],
    1: [8, 7, 14, 12, 19, 17, 20, 22, 7, 10, 13, 13, 17, 15, 19, 20, 14, 13, 15, 14, 16, 18, 18, 21, 12, 13, 14, 15, 19, 17, 19, 22, 19, 17, 16, 19, 15, 22, 17, 21, 17, 15, 18, 17, 22, 19, 22, 32, 20, 19, 18, 19, 17, 22, 35, 32, 22, 20, 21, 22, 21, 32, 32, 255]
}
TELEGRAM_QTABLES = {
    0: [4, 3, 3, 4, 6, 10, 13, 16, 3, 3, 4, 5, 7, 15, 16, 14, 4, 3, 4, 6, 10, 15, 18, 15, 4, 4, 6, 8, 13, 23, 21, 16, 5, 6, 10, 15, 18, 28, 27, 20, 6, 9, 14, 17, 21, 27, 29, 24, 13, 17, 20, 23, 27, 31, 31, 26, 19, 24, 25, 25, 29, 26, 27, 26],
    1: [4, 5, 6, 12, 26, 26, 26, 26, 5, 5, 7, 17, 26, 26, 26, 26, 6, 7, 15, 26, 26, 26, 26, 26, 12, 17, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]
}

def sim_v1(img, platform):
    img = ImageOps.exif_transpose(img).convert('RGB')
    orig_w, orig_h = img.size
    if platform == 'whatsapp':
        max_dim = 1600
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        qt = WHATSAPP_STANDARD_QTABLES if max(orig_w, orig_h) > 800 else WHATSAPP_HD_QTABLES
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=qt, subsampling=2, optimize=True, progressive=True)
        return Image.open(buf)
    elif platform == 'facebook':
        max_dim = 1080
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=FACEBOOK_QTABLES_SIM, subsampling=2, optimize=True)
        return Image.open(buf)
    elif platform == 'telegram':
        max_dim = 1280
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=TELEGRAM_QTABLES, subsampling=2, optimize=True)
        return Image.open(buf)
    elif platform == 'instagram':
        max_dim = 1440
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=65, method=4)
        return Image.open(buf)

def sim_v2(img, platform):
    img = ImageOps.exif_transpose(img).convert('RGB')
    orig_w, orig_h = img.size
    if platform == 'whatsapp':
        max_dim = 1600
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=WHATSAPP_HD_QTABLES, subsampling=2, optimize=True, progressive=False)
        return Image.open(buf)
    elif platform == 'facebook':
        max_width = 1080
        if orig_w > max_width:
            scale = max_width / orig_w
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=FACEBOOK_QTABLES_SIM, subsampling=2, optimize=True, progressive=True)
        return Image.open(buf)
    elif platform == 'telegram':
        max_dim = 1279
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=TELEGRAM_QTABLES, subsampling=2, optimize=True, progressive=True)
        return Image.open(buf)
    elif platform == 'instagram':
        max_dim = 1440
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=71, method=4)
        return Image.open(buf)

def prepare_tensor(pil_img, max_size=1024):
    pil_img = pil_img.convert('RGB')
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = float(max_size) / float(max(w, h))
        new_w = max(8, (int(round(w * scale)) // 8) * 8)
        new_h = max(8, (int(round(h * scale)) // 8) * 8)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w = max(8, (w // 8) * 8)
        new_h = max(8, (h // 8) * 8)
        if (new_w, new_h) != (w, h):
            pil_img = pil_img.crop((0, 0, new_w, new_h))
    arr = np.array(pil_img).transpose(2, 0, 1) / 256.0
    return torch.tensor(arr, dtype=torch.float).unsqueeze(0)

def main():
    print("=" * 80)
    print("TruFor Downstream Evaluation: Real Platform vs v1 vs v2 on CocoGlide")
    print("=" * 80)

    # Load Model
    class DummyArgs:
        opts = None
    update_config(config, DummyArgs())
    device = torch.device('cpu')
    model = confcmx(cfg=config)
    weights_path = PROJECT_ROOT / "pretrained_models" / "trufor.pth.tar"
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("[*] TruFor model loaded.")

    # Cache
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            print(f"[*] Loaded {len(cache)} cached items.")
        except Exception:
            cache = {}

    def get_score(key, pil_img):
        if key in cache:
            return cache[key]
        t = prepare_tensor(pil_img)
        with torch.no_grad():
            _, _, det, _ = model(t)
            score = float(torch.sigmoid(det).item())
        cache[key] = score
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        return score

    # Ground truth reference from evaluation_metrics_all.csv
    real_gt_reference = {
        'whatsapp': {'real_mean': 0.1414, 'fake_mean': 0.8161, 'count': 50},
        'facebook': {'real_mean': 0.2632, 'fake_mean': 0.7001, 'count': 50},
        'telegram': {'real_mean': 0.2435, 'fake_mean': 0.7298, 'count': 60},
        'instagram': {'real_mean': 0.2447, 'fake_mean': 0.3656, 'count': 50},
    }

    # Select 30 real + 30 fake images from CocoGlide (512x512)
    coco_real = sorted((COCO_ROOT / "real").glob("*.png"))[:30]
    coco_fake = sorted((COCO_ROOT / "fake").glob("*.png"))[:30]
    print(f"[*] Processing {len(coco_real)} real and {len(coco_fake)} fake images from CocoGlide...")

    # Clean CocoGlide baseline
    clean_r_scores = [get_score(f"clean_r_{f.name}", Image.open(f)) for f in coco_real]
    clean_f_scores = [get_score(f"clean_f_{f.name}", Image.open(f)) for f in coco_fake]
    clean_stats = {
        'real_mean': round(float(np.mean(clean_r_scores)), 4),
        'real_median': round(float(np.median(clean_r_scores)), 4),
        'fake_mean': round(float(np.mean(clean_f_scores)), 4),
        'fake_median': round(float(np.median(clean_f_scores)), 4),
    }
    print(f"  Clean CocoGlide Baseline: Real mean={clean_stats['real_mean']}, Fake mean={clean_stats['fake_mean']}")

    results = {}
    for p in ['whatsapp', 'facebook', 'telegram', 'instagram']:
        print(f"\nEvaluating {p.upper()} on CocoGlide...")
        t0 = time.time()
        
        # v1 and v2 on 30 real
        v1_r, v2_r = [], []
        for f in coco_real:
            im = Image.open(f)
            # v1
            im_v1 = sim_v1(im, p)
            v1_r.append(get_score(f"coco_{p}_v1_r_{f.name}", im_v1))
            im_v1.close()
            # v2
            im_v2 = sim_v2(im, p)
            v2_r.append(get_score(f"coco_{p}_v2_r_{f.name}", im_v2))
            im_v2.close()
            im.close()

        # v1 and v2 on 30 fake
        v1_f, v2_f = [], []
        for f in coco_fake:
            im = Image.open(f)
            # v1
            im_v1 = sim_v1(im, p)
            v1_f.append(get_score(f"coco_{p}_v1_f_{f.name}", im_v1))
            im_v1.close()
            # v2
            im_v2 = sim_v2(im, p)
            v2_f.append(get_score(f"coco_{p}_v2_f_{f.name}", im_v2))
            im_v2.close()
            im.close()

        t1 = time.time()

        ref = real_gt_reference[p]
        gt_real_mean = ref['real_mean']
        gt_fake_mean = ref['fake_mean']

        v1_rm, v1_fm = float(np.mean(v1_r)), float(np.mean(v1_f))
        v2_rm, v2_fm = float(np.mean(v2_r)), float(np.mean(v2_f))

        mae_v1 = (abs(v1_rm - gt_real_mean) + abs(v1_fm - gt_fake_mean)) / 2.0
        mae_v2 = (abs(v2_rm - gt_real_mean) + abs(v2_fm - gt_fake_mean)) / 2.0

        results[p] = {
            'real_platform_gt': {
                'real_mean': gt_real_mean,
                'fake_mean': gt_fake_mean,
            },
            'v1_simulated': {
                'real_mean': round(v1_rm, 4),
                'real_median': round(float(np.median(v1_r)), 4),
                'fake_mean': round(v1_fm, 4),
                'fake_median': round(float(np.median(v1_f)), 4),
                'mae_to_gt': round(mae_v1, 4),
            },
            'v2_simulated': {
                'real_mean': round(v2_rm, 4),
                'real_median': round(float(np.median(v2_r)), 4),
                'fake_mean': round(v2_fm, 4),
                'fake_median': round(float(np.median(v2_f)), 4),
                'mae_to_gt': round(mae_v2, 4),
            },
            'v2_closer': bool(mae_v2 <= mae_v1),
            'improvement_delta': round(mae_v1 - mae_v2, 4),
            'elapsed_sec': round(t1 - t0, 1),
        }

        print(f"  GT Real Ref : Real={gt_real_mean:.4f}, Fake={gt_fake_mean:.4f}")
        print(f"  v1 Scores   : Real={v1_rm:.4f}, Fake={v1_fm:.4f} (MAE to GT={mae_v1:.4f})")
        print(f"  v2 Scores   : Real={v2_rm:.4f}, Fake={v2_fm:.4f} (MAE to GT={mae_v2:.4f})")
        verdict_str = "v2 is CLOSER to Real Ground Truth" if mae_v2 <= mae_v1 else "v1 was closer"
        print(f"  Verdict     : {verdict_str} (delta = {mae_v1 - mae_v2:+.4f})")

    final_payload = {
        'clean_cocoglide': clean_stats,
        'platforms': results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_payload, f, indent=2)

    # Log to PROGRESS.md
    with open(PROGRESS_FILE, 'a') as f:
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} - Step 6: TruFor Downstream Check Complete\n")
        f.write(f"- Output saved to: {OUTPUT_FILE}\n")
        for p, r in results.items():
            f.write(f"- {p.upper()}: v1 MAE={r['v1_simulated']['mae_to_gt']:.4f}, v2 MAE={r['v2_simulated']['mae_to_gt']:.4f} (v2 closer: {r['v2_closer']})\n")

    print("\n" + "=" * 80)
    print("ALL DOWNSTREAM CHECKS FINISHED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
