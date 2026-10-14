import os, sys, time, json, io
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

torch.set_num_threads(8)

PROJECT_ROOT = Path("D:/ml_project")
sys.path.insert(0, str(PROJECT_ROOT / "test_docker" / "src"))
from config import update_config, _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

# Load TruFor Model
class DummyArgs:
    opts = None
update_config(config, DummyArgs())
model = confcmx(cfg=config)
weights_path = PROJECT_ROOT / "pretrained_models" / "trufor.pth.tar"
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Select 15 authentic and 15 tampered CASIA 1.0 images
au_dir = PROJECT_ROOT / "DATASET" / "CASIA1.0" / "CASIA 1.0 dataset" / "Au"
sp_dir = PROJECT_ROOT / "DATASET" / "CASIA1.0" / "CASIA 1.0 dataset" / "Tp" / "Sp"

au_files = sorted([f for f in os.listdir(au_dir) if f.lower().endswith(('.jpg', '.png'))])[:15]
sp_files = sorted([f for f in os.listdir(sp_dir) if f.lower().endswith(('.jpg', '.png'))])[:15]

print(f"Selected {len(au_files)} authentic and {len(sp_files)} spliced CASIA images.")

# Import v1 and v3 simulators
sys.path.insert(0, str(PROJECT_ROOT / "SIMULATOR"))
import server as sim_v1

sys.path.insert(0, str(PROJECT_ROOT / "simulator_v3"))
import server as sim_v3

def score_tensor(pil_img):
    # Standard TruFor tensor preparation
    w, h = pil_img.size
    new_w = max(8, (w // 8) * 8)
    new_h = max(8, (h // 8) * 8)
    if (new_w, new_h) != (w, h):
        pil_img = pil_img.crop((0, 0, new_w, new_h))
    arr = np.array(pil_img.convert("RGB")).transpose(2, 0, 1) / 256.0
    t = torch.tensor(arr, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        _, _, det, _ = model(t)
        score = float(torch.sigmoid(det).item())
    return score

# Compression functions
def compress_v1_instagram(pil_img):
    # v1: long-side capping at 1440, WebP quality 65 (no upscale, no crop)
    w, h = pil_img.size
    if max(w, h) > 1440:
        scale = 1440.0 / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format='WEBP', quality=65)
    return Image.open(buf)

def compress_v3_instagram(pil_img):
    # v3: upscale small images to 1440 with BICUBIC, WebP quality 60
    w, h = pil_img.size
    scale = 1440.0 / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    buf = io.BytesIO()
    pil_img.save(buf, format='WEBP', quality=60, method=4)
    return Image.open(buf)

y_true = [0] * len(au_files) + [1] * len(sp_files)
v1_scores = []
v3_scores = []
orig_scores = []

all_samples = [(au_dir / f, 0) for f in au_files] + [(sp_dir / f, 1) for f in sp_files]

print("Scoring images through Original, v1 Instagram, and v3 Instagram...")
for idx, (p, label) in enumerate(all_samples):
    t0 = time.time()
    im_orig = Image.open(p)
    s_orig = score_tensor(im_orig)
    
    im_v1 = compress_v1_instagram(im_orig)
    s_v1 = score_tensor(im_v1)
    
    im_v3 = compress_v3_instagram(im_orig)
    s_v3 = score_tensor(im_v3)
    
    orig_scores.append(s_orig)
    v1_scores.append(s_v1)
    v3_scores.append(s_v3)
    
    print(f"[{idx+1}/30] {p.name}: label={label} | Orig={s_orig:.4f} | v1={s_v1:.4f} | v3={s_v3:.4f} ({time.time()-t0:.2f}s)")

auc_orig = roc_auc_score(y_true, orig_scores)
auc_v1 = roc_auc_score(y_true, v1_scores)
auc_v3 = roc_auc_score(y_true, v3_scores)

print("\n" + "="*60)
print(f"CASIA 1.0 (N=30: 15 Au, 15 Sp) Downstream TruFor AUC Results:")
print(f"Original Uncompressed AUC: {auc_orig:.4f}")
print(f"v1 Instagram (384x256, q=65) AUC: {auc_v1:.4f}")
print(f"v3 Instagram (1440px upscaled, q=60) AUC: {auc_v3:.4f}")
print(f"AUC Difference (v3 - v1): {auc_v3 - auc_v1:+.4f}")
print("="*60)

res = {
    'auc_orig': float(auc_orig),
    'auc_v1': float(auc_v1),
    'auc_v3': float(auc_v3),
    'auc_diff': float(auc_v3 - auc_v1),
    'y_true': y_true,
    'orig_scores': orig_scores,
    'v1_scores': v1_scores,
    'v3_scores': v3_scores
}
with open('D:/ml_project/simulator_audit/casia_collapse_test.json', 'w') as f:
    json.dump(res, f, indent=2)
