import os, json, io
import numpy as np
import cv2
from PIL import Image

def calc_noise_energy(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))

def calc_psnr_ssim(im1, im2):
    # Ensure exact same size
    if im1.size != im2.size:
        raise ValueError(f"Size mismatch: {im1.size} vs {im2.size}")
    arr1 = np.array(im1).astype(np.float64)
    arr2 = np.array(im2).astype(np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    
    g1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2GRAY)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2.astype(np.float64), (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1.astype(np.float64) * g2.astype(np.float64), (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(psnr), float(np.mean(ssim_map))

def compress_webp(img, quality):
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=quality, method=4)
    data = buf.getvalue()
    dec = Image.open(io.BytesIO(data))
    bpp = (len(data) * 8.0) / (img.size[0] * img.size[1])
    return dec, len(data), bpp

with open('D:/ml_project/simulator_v3/audit_results/step1_pairing_instagram_real.json') as f:
    real_pairs = json.load(f)['pairs']

with open('D:/ml_project/simulator_v3/audit_results/step1_pairing_instagram_fake.json') as f:
    fake_pairs = json.load(f)['pairs']

filters = {
    'NEAREST': Image.NEAREST,
    'BILINEAR': Image.BILINEAR,
    'BICUBIC': Image.BICUBIC,
    'LANCZOS': Image.LANCZOS
}

results = {'real': {}, 'fake': {}}

# =========================================================================
# PART A: LARGE ORIGINALS (DATA/REAL) -> Instagram_Real (all 25 are 1440x1440)
# =========================================================================
print("Starting Part A: Large Originals (DATA/REAL)...")
real_gt_bpp = []
real_gt_noise = []
for p in real_pairs:
    im_real = Image.open(p['posted_path'])
    real_gt_noise.append(calc_noise_energy(np.array(im_real)))
    real_gt_bpp.append((os.path.getsize(p['posted_path']) * 8.0) / (im_real.size[0] * im_real.size[1]))

results['real']['gt_noise_median'] = float(np.median(real_gt_noise))
results['real']['gt_bpp_median'] = float(np.median(real_gt_bpp))

# 1. Test downscale filters at quality 60
filter_comp_real = {}
for fname, filt in filters.items():
    bpps, psnrs, ssims, noises = [], [], [], []
    for p in real_pairs:
        im_o = Image.open(p['orig_path'])
        im_real = Image.open(p['posted_path'])
        w, h = im_o.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        crop = im_o.crop((left, top, left + min_dim, top + min_dim))
        resized = crop.resize((1440, 1440), filt)
        dec, size_bytes, bpp = compress_webp(resized, quality=60)
        p_val, s_val = calc_psnr_ssim(dec, im_real)
        n_val = calc_noise_energy(np.array(dec))
        bpps.append(bpp)
        psnrs.append(p_val)
        ssims.append(s_val)
        noises.append(n_val)
    
    filter_comp_real[fname] = {
        'bpp_median': float(np.median(bpps)),
        'bpp_ratio': float(np.median(bpps) / np.median(real_gt_bpp)),
        'psnr_median': float(np.median(psnrs)),
        'ssim_median': float(np.median(ssims)),
        'noise_median': float(np.median(noises))
    }
results['real']['filters_at_q60'] = filter_comp_real

# 2. Quality sweep on large originals with BICUBIC and BILINEAR
q_sweep_real = {}
for q in [50, 55, 60, 65, 70, 75, 80]:
    bpps, psnrs, ssims, noises = [], [], [], []
    for p in real_pairs:
        im_o = Image.open(p['orig_path'])
        im_real = Image.open(p['posted_path'])
        w, h = im_o.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        crop = im_o.crop((left, top, left + min_dim, top + min_dim))
        resized = crop.resize((1440, 1440), Image.BICUBIC)
        dec, size_bytes, bpp = compress_webp(resized, quality=q)
        p_val, s_val = calc_psnr_ssim(dec, im_real)
        n_val = calc_noise_energy(np.array(dec))
        bpps.append(bpp)
        psnrs.append(p_val)
        ssims.append(s_val)
        noises.append(n_val)
    q_sweep_real[q] = {
        'bpp_median': float(np.median(bpps)),
        'bpp_ratio': float(np.median(bpps) / np.median(real_gt_bpp)),
        'psnr_median': float(np.median(psnrs)),
        'ssim_median': float(np.median(ssims)),
        'noise_median': float(np.median(noises))
    }
results['real']['q_sweep_bicubic'] = q_sweep_real

# =========================================================================
# PART B: SMALL ORIGINALS (DATA/FAKE, CASIA 384x256) -> Instagram_Fake
# =========================================================================
print("Starting Part B: Small Originals (DATA/FAKE)...")
# For the fake pairs, 20 are 1440x1440, 3 are 1440x1152, 2 are 1440x1148
# Let's inspect them. For alignment, we test on the 20 1440x1440 square pairs
# (or align to each pair's posted target size).
fake_square_pairs = []
for p in fake_pairs:
    with Image.open(p['posted_path']) as im_r:
        if im_r.size == (1440, 1440):
            fake_square_pairs.append(p)

fake_gt_bpp = []
fake_gt_noise = []
for p in fake_square_pairs:
    im_real = Image.open(p['posted_path'])
    fake_gt_noise.append(calc_noise_energy(np.array(im_real)))
    fake_gt_bpp.append((os.path.getsize(p['posted_path']) * 8.0) / (im_real.size[0] * im_real.size[1]))

results['fake']['gt_noise_median'] = float(np.median(fake_gt_noise))
results['fake']['gt_bpp_median'] = float(np.median(fake_gt_bpp))

# 1. Test upscale filters on small originals (quality 60)
filter_comp_fake = {}
for fname, filt in filters.items():
    bpps, psnrs, ssims, noises = [], [], [], []
    for p in fake_square_pairs:
        im_o = Image.open(p['orig_path'])
        im_real = Image.open(p['posted_path'])
        w, h = im_o.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        crop = im_o.crop((left, top, left + min_dim, top + min_dim))
        # Upscale 256x256 -> 1440x1440
        upscaled = crop.resize((1440, 1440), filt)
        dec, size_bytes, bpp = compress_webp(upscaled, quality=60)
        p_val, s_val = calc_psnr_ssim(dec, im_real)
        n_val = calc_noise_energy(np.array(dec))
        bpps.append(bpp)
        psnrs.append(p_val)
        ssims.append(s_val)
        noises.append(n_val)
    filter_comp_fake[fname] = {
        'bpp_median': float(np.median(bpps)),
        'bpp_ratio': float(np.median(bpps) / np.median(fake_gt_bpp)),
        'psnr_median': float(np.median(psnrs)),
        'ssim_median': float(np.median(ssims)),
        'noise_median': float(np.median(noises))
    }
results['fake']['filters_at_q60'] = filter_comp_fake

# 2. Quality sweep on small originals with BICUBIC / BILINEAR
q_sweep_fake = {}
for q in [50, 55, 60, 65, 70, 75, 80]:
    bpps, psnrs, ssims, noises = [], [], [], []
    for p in fake_square_pairs:
        im_o = Image.open(p['orig_path'])
        im_real = Image.open(p['posted_path'])
        w, h = im_o.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        crop = im_o.crop((left, top, left + min_dim, top + min_dim))
        upscaled = crop.resize((1440, 1440), Image.BICUBIC)
        dec, size_bytes, bpp = compress_webp(upscaled, quality=q)
        p_val, s_val = calc_psnr_ssim(dec, im_real)
        n_val = calc_noise_energy(np.array(dec))
        bpps.append(bpp)
        psnrs.append(p_val)
        ssims.append(s_val)
        noises.append(n_val)
    q_sweep_fake[q] = {
        'bpp_median': float(np.median(bpps)),
        'bpp_ratio': float(np.median(bpps) / np.median(fake_gt_bpp)),
        'psnr_median': float(np.median(psnrs)),
        'ssim_median': float(np.median(ssims)),
        'noise_median': float(np.median(noises))
    }
results['fake']['q_sweep_bicubic'] = q_sweep_fake

with open('D:/ml_project/simulator_audit/instagram_step2_fitting.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Done! Saved to D:/ml_project/simulator_audit/instagram_step2_fitting.json")
