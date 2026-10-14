import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from PIL import Image

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path("D:/ml_project")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from study.model_manager import get_trufor_model
from study.evaluator import collect_test_dataset_pairs
from test_docker.src.models.cmx.layer_utils import weighted_statistics_pooling

device = torch.device("cpu")
RESULTS_DIR = BASE_DIR / "study" / "results_stage1"
CHECKPOINT_DIR = BASE_DIR / "study" / "checkpoints_stage1_uncompressed"

print("=" * 80)
print(" RUNNING STAGE 1 UNCOMPRESSED TEST EVALUATION (SIMULTANEOUS DUAL-HEAD)")
print("=" * 80)

suite = collect_test_dataset_pairs(max_per_class=50)

# Load baseline model
base_model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
base_model.eval()

# Load Phase 1 detection head weights
p1_ckpt = CHECKPOINT_DIR / "phase1_head_only_epoch3.pth"
p1_state = torch.load(p1_ckpt, weights_only=False)["state_dict"]

# Clone detection head for Phase 1
p1_det_head = base_model.detection.__class__(
    torch.nn.Linear(8, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(128, 1)
).to(device)
p1_det_dict = {k.replace("detection.", ""): v for k, v in p1_state.items() if k.startswith("detection.")}
p1_det_head.load_state_dict(p1_det_dict)
p1_det_head.eval()

# Optimal validation thresholds
val_tau_base = 0.37
val_tau_p1 = 0.38

def extract_features_and_scores(img_path, max_size=512):
    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = max(8, (int(round(w * scale)) // 8) * 8)
        new_h = max(8, (int(round(h * scale)) // 8) * 8)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w = max(8, (w // 8) * 8)
        new_h = max(8, (h // 8) * 8)
        if (new_w, new_h) != (w, h):
            pil_img = pil_img.crop((0, 0, new_w, new_h))
            
    img_np = np.array(pil_img, dtype=np.float32) / 255.0
    t_rgb = torch.tensor(img_np.transpose(2, 0, 1), device=device).unsqueeze(0)
    
    with torch.no_grad():
        out, conf, det_base, _ = base_model(t_rgb)
        f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
        f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(out.shape[0], -1)
        feat = torch.cat((f1, f2), -1)
        
        score_base = torch.sigmoid(det_base).item()
        score_p1 = torch.sigmoid(p1_det_head(feat)).item()
        
    return score_base, score_p1

all_results = {"Pretrained_Baseline": {}, "Phase1_Head_Only": {}}

for ds_name in ["Columbia", "CocoGlide", "CASIA1.0"]:
    print(f"\n[*] Evaluating {ds_name} (50 Real, 50 Fake)...", flush=True)
    t0 = time.time()
    orig = suite[ds_name]["original"]
    real_imgs = orig["real"][:50]
    fake_imgs = orig["fake"][:50]
    
    y_true = []
    base_scores = []
    p1_scores = []
    
    for i, p in enumerate(real_imgs):
        sb, sp1 = extract_features_and_scores(p)
        base_scores.append(sb)
        p1_scores.append(sp1)
        y_true.append(0)
        
    for i, p in enumerate(fake_imgs):
        sb, sp1 = extract_features_and_scores(p)
        base_scores.append(sb)
        p1_scores.append(sp1)
        y_true.append(1)
        
    y_true = np.array(y_true)
    base_scores = np.array(base_scores)
    p1_scores = np.array(p1_scores)
    elapsed = time.time() - t0
    print(f"    Completed in {elapsed:.1f}s ({elapsed/len(y_true):.2f}s/img)")
    
    for m_name, scores, tau_star in [
        ("Pretrained_Baseline", base_scores, val_tau_base),
        ("Phase1_Head_Only", p1_scores, val_tau_p1)
    ]:
        auc = roc_auc_score(y_true, scores) * 100
        
        def calc_metrics(tau):
            preds = (scores >= tau).astype(int)
            bacc = balanced_accuracy_score(y_true, preds) * 100
            tp = np.sum((preds == 1) & (y_true == 1))
            fn = np.sum((preds == 0) & (y_true == 1))
            tn = np.sum((preds == 0) & (y_true == 0))
            fp = np.sum((preds == 1) & (y_true == 0))
            tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
            return bacc, tpr, tnr
            
        bacc_50, tpr_50, tnr_50 = calc_metrics(0.50)
        bacc_tau, tpr_tau, tnr_tau = calc_metrics(tau_star)
        
        real_sc = scores[y_true == 0]
        fake_sc = scores[y_true == 1]
        
        res = {
            "dataset": ds_name,
            "AUC": auc,
            "bACC_050": bacc_50,
            "TPR_050": tpr_50,
            "TNR_050": tnr_50,
            "tau_star": tau_star,
            "bACC_tau_star": bacc_tau,
            "TPR_tau_star": tpr_tau,
            "TNR_tau_star": tnr_tau,
            "real_mean": float(real_sc.mean()),
            "real_med": float(np.median(real_sc)),
            "real_std": float(real_sc.std()),
            "fake_mean": float(fake_sc.mean()),
            "fake_med": float(np.median(fake_sc)),
            "fake_std": float(fake_sc.std()),
            "margin": float(np.median(fake_sc) - np.median(real_sc)),
            "n_real": len(real_sc),
            "n_fake": len(fake_sc)
        }
        all_results[m_name][ds_name] = res
        print(f"    [{m_name:<19}] AUC={auc:6.2f}% | bACC(0.50)={bacc_50:5.2f}% | bACC(tau*={tau_star:.2f})={bacc_tau:5.2f}% | TPR={tpr_tau:5.2f}% | TNR={tnr_tau:5.2f}% | Margin={res['margin']:+.4f}")
        print(f"                         Real Med={res['real_med']:.4f} (Mean {res['real_mean']:.4f}) | Fake Med={res['fake_med']:.4f} (Mean {res['fake_mean']:.4f})")

out_file = RESULTS_DIR / "uncompressed_benchmark_evaluation.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"\n[+] Successfully saved benchmark results to {out_file.name}")
