"""
compute_spectral_and_noise_stats.py
Calculates DCT-domain statistics, blocking artifact strength, and high-frequency
noise residual energy between Real platform images, Simulator v1, and Simulator v2.
"""

import os
import io
import json
import statistics
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps

DATA_ROOT = Path(r"D:\ml_project\DATA")
OUTPUT_FILE = Path(r"D:\ml_project\simulator_audit\per_platform\dct_and_noise_stats.json")

# Import simulation logic from audit_pipeline or define here
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
    w, h = img.size
    if platform == 'whatsapp':
        max_dim = 1600
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        qt = WHATSAPP_STANDARD_QTABLES if max(w, h) > 800 else WHATSAPP_HD_QTABLES
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=qt, subsampling=2, optimize=True, progressive=True)
        return Image.open(buf)
    elif platform == 'facebook':
        max_dim = 1080
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=FACEBOOK_QTABLES_SIM, subsampling=2, optimize=True)
        return Image.open(buf)
    elif platform == 'telegram':
        max_dim = 1280
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=TELEGRAM_QTABLES, subsampling=2, optimize=True)
        return Image.open(buf)
    elif platform == 'instagram':
        max_dim = 1440
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=65, method=4)
        return Image.open(buf)

def sim_v2(img, platform):
    img = ImageOps.exif_transpose(img).convert('RGB')
    w, h = img.size
    if platform == 'whatsapp':
        max_dim = 1600
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=WHATSAPP_HD_QTABLES, subsampling=2, optimize=True, progressive=False)
        return Image.open(buf)
    elif platform == 'facebook':
        max_w = 1080
        if w > max_w:
            scale = max_w / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=FACEBOOK_QTABLES_SIM, subsampling=2, optimize=True, progressive=True)
        return Image.open(buf)
    elif platform == 'telegram':
        max_dim = 1279
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', qtables=TELEGRAM_QTABLES, subsampling=2, optimize=True, progressive=True)
        return Image.open(buf)
    elif platform == 'instagram':
        max_dim = 1440
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=82, method=4)
        return Image.open(buf)

def compute_image_metrics(pil_img):
    gray = np.array(pil_img.convert('L'), dtype=np.float32)
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    g = gray[:h8, :w8]

    # 1. Blocking artifact strength (horizontal & vertical boundary ratio)
    diff_h = np.abs(g[:, 1:] - g[:, :-1])
    diff_v = np.abs(g[1:, :] - g[:-1, :])

    b_cols = np.arange(7, w8 - 1, 8)
    i_cols = np.setdiff1d(np.arange(w8 - 1), b_cols)
    b_h = np.mean(diff_h[:, b_cols]) if len(b_cols) > 0 else 1.0
    s_h = np.mean(diff_h[:, i_cols]) if len(i_cols) > 0 else 1.0

    b_rows = np.arange(7, h8 - 1, 8)
    i_rows = np.setdiff1d(np.arange(h8 - 1), b_rows)
    b_v = np.mean(diff_v[b_rows, :]) if len(b_rows) > 0 else 1.0
    s_v = np.mean(diff_v[i_rows, :]) if len(i_rows) > 0 else 1.0

    blocking = float(((b_h / s_h) + (b_v / s_v)) / 2.0)

    # 2. High-frequency noise residual energy (cv2 Laplacian filter)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    noise_energy = float(np.var(lap))

    # 3. Median filter residual energy
    med = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    med_energy = float(np.var(g - med))

    # 4. 8x8 Block DCT statistics
    # Sample up to 1000 blocks evenly
    blocks_h = h8 // 8
    blocks_w = w8 // 8
    total_blocks = blocks_h * blocks_w
    step = max(1, total_blocks // 1000)

    ac_energies = []
    hf_energies = []
    
    # 2D DCT on 8x8 blocks
    for bh in range(0, blocks_h, max(1, int(np.sqrt(step)))):
        for bw in range(0, blocks_w, max(1, int(np.sqrt(step)))):
            block = g[bh*8:(bh+1)*8, bw*8:(bw+1)*8]
            dct_b = cv2.dct(block)
            # AC coefficients (exclude DC [0,0])
            ac = dct_b.copy()
            ac[0, 0] = 0
            ac_energy = float(np.sum(ac ** 2))
            ac_energies.append(ac_energy)
            # High frequency (u + v >= 6)
            mask_hf = np.fromfunction(lambda u, v: (u + v) >= 6, (8, 8))
            hf_energy = float(np.sum((dct_b * mask_hf) ** 2))
            hf_energies.append(hf_energy)

    return {
        'blocking_artifact_strength': round(blocking, 4),
        'noise_residual_energy': round(noise_energy, 2),
        'median_residual_energy': round(med_energy, 2),
        'dct_ac_energy': round(float(np.mean(ac_energies)), 2),
        'dct_hf_energy': round(float(np.mean(hf_energies)), 2),
    }

def summarize(metric_list):
    keys = metric_list[0].keys()
    summary = {}
    for k in keys:
        vals = [m[k] for m in metric_list]
        summary[k] = {
            'mean': round(float(np.mean(vals)), 2),
            'median': round(float(np.median(vals)), 2),
            'min': round(float(np.min(vals)), 2),
            'max': round(float(np.max(vals)), 2)
        }
    return summary

def main():
    print("Computing DCT and noise residual statistics across platforms...")
    real_origs = sorted((DATA_ROOT / "REAL").glob("*.png"))[:25]

    real_folders = {
        'whatsapp': DATA_ROOT / "WHATSAPP" / "whatsapp_real",
        'facebook': DATA_ROOT / "FACEBOOK" / "facebook_real",
        'telegram': DATA_ROOT / "TELEGRAM" / "telegram_real",
        'instagram': DATA_ROOT / "INSTAGRAM" / "Instagram_Real",
    }

    results = {}

    for platform in ['whatsapp', 'facebook', 'telegram', 'instagram']:
        print(f"Processing {platform.upper()}...")
        # 1. Real platform images
        real_files = sorted(real_folders[platform].glob("*"))
        real_metrics = []
        for rf in real_files:
            try:
                im = Image.open(rf)
                real_metrics.append(compute_image_metrics(im))
                im.close()
            except Exception as e:
                print(f"Error on real {rf.name}: {e}")

        # 2. Simulated v1 and v2 on originals
        v1_metrics = []
        v2_metrics = []
        for orig_f in real_origs:
            try:
                im = Image.open(orig_f)
                im_v1 = sim_v1(im, platform)
                v1_metrics.append(compute_image_metrics(im_v1))
                im_v1.close()

                im_v2 = sim_v2(im, platform)
                v2_metrics.append(compute_image_metrics(im_v2))
                im_v2.close()
                im.close()
            except Exception as e:
                print(f"Error on orig {orig_f.name}: {e}")

        results[platform] = {
            'real': summarize(real_metrics),
            'v1': summarize(v1_metrics),
            'v2': summarize(v2_metrics),
        }
        print(f"  {platform.upper()} Real noise energy: {results[platform]['real']['noise_residual_energy']['median']}")
        print(f"  {platform.upper()} v1 noise energy:   {results[platform]['v1']['noise_residual_energy']['median']}")
        print(f"  {platform.upper()} v2 noise energy:   {results[platform]['v2']['noise_residual_energy']['median']}")
        print(f"  {platform.upper()} Real blocking:       {results[platform]['real']['blocking_artifact_strength']['median']}")
        print(f"  {platform.upper()} v1 blocking:         {results[platform]['v1']['blocking_artifact_strength']['median']}")
        print(f"  {platform.upper()} v2 blocking:         {results[platform]['v2']['blocking_artifact_strength']['median']}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Done! Saved results to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
