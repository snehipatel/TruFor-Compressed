"""
Phase 2 Fine-Tuning Pipeline for TruFor on UNCOMPRESSED Images.
Unfreezes:
- decode_head and decode_head_conf at LR 1e-6
- detection head at LR 1e-5
- Backbone and DnCNN strictly FROZEN
- AdamW, weight decay 1e-4, Cosine Annealing, BCEWithLogitsLoss
- Early stopping on Validation ROC-AUC (patience 3)
- Checkpoints saved in study/checkpoints_stage1_uncompressed/phase2_best.pth
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from study.model_manager import get_trufor_model
from study.dataset_builder_stage1 import is_compressed_path

CHECKPOINT_DIR = BASE_DIR / "study" / "checkpoints_stage1_uncompressed"
RESULTS_DIR = BASE_DIR / "study" / "results_stage1"
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage1"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_image_tensor(img_rel_path, max_size=512, device="cpu"):
    full_p = BASE_DIR / img_rel_path
    pil_img = Image.open(str(full_p)).convert("RGB")
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
    return torch.tensor(img_np.transpose(2, 0, 1), device=device).unsqueeze(0)


def evaluate_phase2_val(model, val_samples, criterion, device):
    model.eval()
    losses = []
    scores = []
    y_true = []
    
    with torch.no_grad():
        for s in val_samples:
            img = load_image_tensor(s["image_path"], device=device)
            target = torch.tensor([float(s["label"])], device=device)
            
            _, _, pred_det, _ = model(img)
            loss = criterion(pred_det.squeeze(), target.squeeze())
            losses.append(loss.item())
            
            prob = torch.sigmoid(pred_det).item()
            scores.append(prob)
            y_true.append(s["label"])
            
    y_true = np.array(y_true)
    scores = np.array(scores)
    
    auc = roc_auc_score(y_true, scores)
    bacc_05 = balanced_accuracy_score(y_true, (scores >= 0.50).astype(int))
    
    real_sc = scores[y_true == 0]
    fake_sc = scores[y_true == 1]
    
    return np.mean(losses), auc, bacc_05, real_sc, fake_sc


def run_phase2_training(epochs=3, patience=2, save_prefix="phase2_multi_head"):
    device = torch.device("cpu")
    print("\n" + "=" * 80)
    print(f" [PHASE 2] MULTI-HEAD FINE-TUNING | FULL RUN ({epochs} Epochs)")
    print(" Architecture: Backbone & DnCNN STRICTLY FROZEN")
    print(" Trainable: decode_head (1e-6), decode_head_conf (1e-6), detection head (1e-5)")
    print("=" * 80)
    
    # Load manifests (pilot / balanced subset for reliable CPU execution)
    with open(SPLITS_DIR / "train_pilot_manifest.json", "r") as f:
        train_samples = json.load(f)
    with open(SPLITS_DIR / "val_pilot_manifest.json", "r") as f:
        val_samples = json.load(f)
        
    print(f"[*] Train set: {len(train_samples)} samples | Real={sum(1 for s in train_samples if s['label']==0)}, Fake={sum(1 for s in train_samples if s['label']==1)}")
    print(f"[*] Val set:   {len(val_samples)} samples | Real={sum(1 for s in val_samples if s['label']==0)}, Fake={sum(1 for s in val_samples if s['label']==1)}")
    
    # Load model and configure freezing
    model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    
    for p in model.parameters():
        p.requires_grad = False
    for p in model.decode_head.parameters():
        p.requires_grad = True
    if model.decode_head_conf is not None:
        for p in model.decode_head_conf.parameters():
            p.requires_grad = True
    for p in model.detection.parameters():
        p.requires_grad = True
        
    decoders_params = list(model.decode_head.parameters()) + list(model.decode_head_conf.parameters())
    det_params = list(model.detection.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": decoders_params, "lr": 1e-6, "weight_decay": 1e-4},
        {"params": det_params, "lr": 1e-5, "weight_decay": 1e-4}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    
    # Baseline validation evaluation before Phase 2
    init_loss, init_auc, init_bacc, init_real_sc, init_fake_sc = evaluate_phase2_val(
        model, val_samples, criterion, device
    )
    init_real_med = np.median(init_real_sc)
    init_fake_med = np.median(init_fake_sc)
    print("\n[*] Initial Validation Performance Before Phase 2:")
    print(f"    Val Loss: {init_loss:.4f} | Val AUC: {init_auc*100:.2f}% | Val bACC(0.50): {init_bacc*100:.2f}%")
    print(f"    Real Scores -> Mean: {init_real_sc.mean():.4f} | Median: {init_real_med:.4f} | Std: {init_real_sc.std():.4f}")
    print(f"    Fake Scores -> Mean: {init_fake_sc.mean():.4f} | Median: {init_fake_med:.4f} | Std: {init_fake_sc.std():.4f}")
    print(f"    Score Margin: {init_fake_med - init_real_med:+.4f}\n")
    
    history = []
    best_val_auc = init_auc
    best_epoch = 0
    best_ckpt_path = None
    no_improve_epochs = 0
    accum_steps = 4
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        model.dncnn.eval()
        
        indices = list(range(len(train_samples)))
        np.random.shuffle(indices)
        
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, idx in enumerate(indices):
            s = train_samples[idx]
            img = load_image_tensor(s["image_path"], device=device)
            target = torch.tensor([float(s["label"])], device=device)
            
            _, _, pred_det, _ = model(img)
            loss = criterion(pred_det.squeeze(), target.squeeze())
            loss_norm = loss / accum_steps
            loss_norm.backward()
            running_loss += loss.item()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
                torch.nn.utils.clip_grad_norm_(decoders_params + det_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        scheduler.step()
        train_loss = running_loss / len(train_samples)
        
        val_loss, val_auc, val_bacc, val_real_sc, val_fake_sc = evaluate_phase2_val(
            model, val_samples, criterion, device
        )
        elapsed = time.time() - t0
        
        real_med = np.median(val_real_sc)
        fake_med = np.median(val_fake_sc)
        margin = fake_med - real_med
        
        print(f"[Epoch {epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc*100:.2f}% | Val bACC(0.50): {val_bacc*100:.2f}%")
        print(f"             Real Mean: {val_real_sc.mean():.4f} (Med {real_med:.4f}) | Fake Mean: {val_fake_sc.mean():.4f} (Med {fake_med:.4f}) | Margin: {margin:+.4f}", flush=True)
        
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_bacc": val_bacc,
            "real_score_mean": float(val_real_sc.mean()),
            "real_score_med": float(real_med),
            "fake_score_mean": float(val_fake_sc.mean()),
            "fake_score_med": float(fake_med),
            "margin": float(margin)
        }
        history.append(record)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            no_improve_epochs = 0
            best_ckpt_path = CHECKPOINT_DIR / f"{save_prefix}_best.pth"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_bacc": val_bacc,
                "history": history
            }, best_ckpt_path)
            print(f"  [+] Saved new best model checkpoint to {best_ckpt_path.name} (Val AUC: {val_auc*100:.2f}%)", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n[!] Early stopping triggered: Val AUC has not improved for {patience} consecutive epochs.")
                break
                
    hist_file = RESULTS_DIR / f"{save_prefix}_history.json"
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n[*] Phase 2 Complete. Best Epoch: {best_epoch} | Best Val AUC: {best_val_auc*100:.2f}%")
    return best_ckpt_path, history


if __name__ == "__main__":
    run_phase2_training(epochs=3)
