import os, sys, time, json, io
from pathlib import Path
import numpy as np
import torch
from PIL import Image

torch.set_num_threads(10)

PROJECT_ROOT = Path("D:/ml_project")
sys.path.insert(0, str(PROJECT_ROOT / "test_docker" / "src"))
from config import update_config, _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

# Load TruFor Model
class DummyArgs: opts = None
update_config(config, DummyArgs())
model = confcmx(cfg=config)
weights_path = PROJECT_ROOT / "pretrained_models" / "trufor.pth.tar"
ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Load Simulators
sys.path.insert(0, str(PROJECT_ROOT / "SIMULATOR"))
import server as sim_v1

sys.path.insert(0, str(PROJECT_ROOT / "simulator_v3"))
import server as sim_v3

def prepare_tensor(pil_img, max_size=512):
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    if max_size and max(w, h) > max_size:
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

def score_pil(pil_img, max_size=512):
    t = prepare_tensor(pil_img, max_size=max_size)
    with torch.no_grad():
        _, _, det, _ = model(t)
        score = float(torch.sigmoid(det).item())
    return score

# Load pairs
with open('D:/ml_project/simulator_v3/audit_results/step1_pairing_whatsapp_real.json') as f:
    wa_pairs = json.load(f)['pairs']

with open('D:/ml_project/simulator_v3/audit_results/step1_pairing_instagram_real.json') as f:
    ig_pairs = json.load(f)['pairs']

results = {}

for platform, pairs, max_eval_size in [('whatsapp', wa_pairs, 512), ('instagram', ig_pairs, 512)]:
    print(f"\n=======================================================")
    print(f"Rerunning 25-image Paired Comparison for {platform.upper()} (max_size={max_eval_size})")
    print(f"=======================================================")
    
    real_scores = []
    v1_scores = []
    v3_scores = []
    
    for i, p in enumerate(pairs):
        t0 = time.time()
        im_orig = Image.open(p['orig_path'])
        im_real = Image.open(p['posted_path'])
        
        # Real score
        s_real = score_pil(im_real, max_size=max_eval_size)
        
        # v1 simulation
        b_v1, _, _, _ = sim_v1.compress_image(im_orig, sim_v1.PLATFORM_PROFILES[platform]['standard'])
        im_v1 = Image.open(io.BytesIO(b_v1))
        s_v1 = score_pil(im_v1, max_size=max_eval_size)
        
        # v3 simulation
        if platform == 'instagram':
            # In v3 for Instagram, center crop to 1:1 and resize to 1440 with BICUBIC
            w, h = im_orig.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            crop = im_orig.crop((left, top, left + min_dim, top + min_dim))
            res = crop.resize((1440, 1440), Image.BICUBIC)
            buf = io.BytesIO()
            res.save(buf, format='WEBP', quality=60, method=4)
            im_v3 = Image.open(buf)
        else:
            b_v3, _, _, _ = sim_v3.compress_image(im_orig, sim_v3.PLATFORM_PROFILES[platform]['standard'])
            im_v3 = Image.open(io.BytesIO(b_v3))
        s_v3 = score_pil(im_v3, max_size=max_eval_size)
        
        real_scores.append(s_real)
        v1_scores.append(s_v1)
        v3_scores.append(s_v3)
        
        print(f"[{i+1}/25] {p['orig_filename']}: Real={s_real:.4f} | v1={s_v1:.4f} (diff={s_v1-s_real:+.4f}) | v3={s_v3:.4f} (diff={s_v3-s_real:+.4f}) [{time.time()-t0:.2f}s]")
    
    mae_v1 = float(np.mean(np.abs(np.array(v1_scores) - np.array(real_scores))))
    mae_v3 = float(np.mean(np.abs(np.array(v3_scores) - np.array(real_scores))))
    mean_diff_v1 = float(np.mean(np.array(v1_scores) - np.array(real_scores)))
    mean_diff_v3 = float(np.mean(np.array(v3_scores) - np.array(real_scores)))
    corr_v1 = float(np.corrcoef(v1_scores, real_scores)[0, 1])
    corr_v3 = float(np.corrcoef(v3_scores, real_scores)[0, 1])
    
    print(f"\n--- {platform.upper()} SUMMARY (max_size={max_eval_size}) ---")
    print(f"Real Mean: {np.mean(real_scores):.4f}")
    print(f"v1 Mean:   {np.mean(v1_scores):.4f} | v1 MAE: {mae_v1:.4f} | Mean Error: {mean_diff_v1:+.4f} | Corr: {corr_v1:.4f}")
    print(f"v3 Mean:   {np.mean(v3_scores):.4f} | v3 MAE: {mae_v3:.4f} | Mean Error: {mean_diff_v3:+.4f} | Corr: {corr_v3:.4f}")
    
    results[platform] = {
        'max_size': max_eval_size,
        'real_mean': float(np.mean(real_scores)),
        'v1_mean': float(np.mean(v1_scores)),
        'v3_mean': float(np.mean(v3_scores)),
        'mae_v1': mae_v1,
        'mae_v3': mae_v3,
        'mean_diff_v1': mean_diff_v1,
        'mean_diff_v3': mean_diff_v3,
        'corr_v1': corr_v1,
        'corr_v3': corr_v3,
        'real_scores': real_scores,
        'v1_scores': v1_scores,
        'v3_scores': v3_scores
    }

with open('D:/ml_project/simulator_audit/step3_rerun_resolution_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to D:/ml_project/simulator_audit/step3_rerun_resolution_results.json")
