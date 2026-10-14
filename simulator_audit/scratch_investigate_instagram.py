import os, json, io
import numpy as np
import cv2
from PIL import Image

def calc_noise_energy(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))

def calc_psnr_ssim(im1, im2):
    # Expects numpy RGB arrays of exact same shape
    arr1 = np.array(im1).astype(np.float64)
    arr2 = np.array(im2).astype(np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    
    # SSIM on gray
    g1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2GRAY)
    # OpenCV ssim or basic formula
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

with open('D:/ml_project/simulator_v3/audit_results/step1_pairing_instagram_real.json') as f:
    real_pairs = json.load(f)['pairs']

with open('D:/ml_project/simulator_v3/audit_results/step1_pairing_instagram_fake.json') as f:
    fake_pairs = json.load(f)['pairs']

filters = {
    'NEAREST': Image.NEAREST,
    'BOX': Image.BOX,
    'BILINEAR': Image.BILINEAR,
    'BICUBIC': Image.BICUBIC,
    'LANCZOS': Image.LANCZOS
}

print('=== REAL INSTAGRAM: LARGE ORIGINALS (DATA/REAL) ===')
# Let us check crop alignment
# In DATA/REAL, images are 4928x3264 (landscape) or 3264x4928 (portrait)
# Real Instagram posted images are 1440x1440
# Is it exact center crop? Let's check!
diffs_lanc = []
diffs_bil = []
diffs_bic = []
diffs_box = []

real_noise_list = []
v2_no_crop_noise = []
v3_crop_lanc_noise = []
v3_crop_bic_noise = []
v3_crop_bil_noise = []

psnr_lanc_list = []
ssim_lanc_list = []
psnr_bil_list = []
ssim_bil_list = []
psnr_bic_list = []
ssim_bic_list = []

for p in real_pairs:
    im_o = Image.open(p['orig_path'])
    im_real = Image.open(p['posted_path'])
    rn = calc_noise_energy(np.array(im_real))
    real_noise_list.append(rn)
    
    w, h = im_o.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    im_crop = im_o.crop((left, top, left + min_dim, top + min_dim))
    
    # Test Lanczos
    res_lanc = im_crop.resize((1440, 1440), Image.LANCZOS)
    buf = io.BytesIO()
    res_lanc.save(buf, format='WEBP', quality=60)
    sim_lanc = Image.open(buf)
    v3_crop_lanc_noise.append(calc_noise_energy(np.array(sim_lanc)))
    p_l, s_l = calc_psnr_ssim(sim_lanc, im_real)
    psnr_lanc_list.append(p_l)
    ssim_lanc_list.append(s_l)
    
    # Test Bicubic
    res_bic = im_crop.resize((1440, 1440), Image.BICUBIC)
    buf = io.BytesIO()
    res_bic.save(buf, format='WEBP', quality=60)
    sim_bic = Image.open(buf)
    v3_crop_bic_noise.append(calc_noise_energy(np.array(sim_bic)))
    p_bc, s_bc = calc_psnr_ssim(sim_bic, im_real)
    psnr_bic_list.append(p_bc)
    ssim_bic_list.append(s_bc)

    # Test Bilinear
    res_bil = im_crop.resize((1440, 1440), Image.BILINEAR)
    buf = io.BytesIO()
    res_bil.save(buf, format='WEBP', quality=60)
    sim_bil = Image.open(buf)
    v3_crop_bil_noise.append(calc_noise_energy(np.array(sim_bil)))
    p_b, s_b = calc_psnr_ssim(sim_bil, im_real)
    psnr_bil_list.append(p_b)
    ssim_bil_list.append(s_b)

print(f'Real Noise (Median): {np.median(real_noise_list):.2f}')
print(f'Crop + LANCZOS Noise: {np.median(v3_crop_lanc_noise):.2f} | PSNR: {np.median(psnr_lanc_list):.2f} dB, SSIM: {np.median(ssim_lanc_list):.4f}')
print(f'Crop + BICUBIC Noise: {np.median(v3_crop_bic_noise):.2f} | PSNR: {np.median(psnr_bic_list):.2f} dB, SSIM: {np.median(ssim_bic_list):.4f}')
print(f'Crop + BILINEAR Noise: {np.median(v3_crop_bil_noise):.2f} | PSNR: {np.median(psnr_bil_list):.2f} dB, SSIM: {np.median(ssim_bil_list):.4f}')
