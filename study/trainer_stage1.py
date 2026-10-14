"""
Stage 1 Fine-Tuning Pipeline for TruFor on UNCOMPRESSED Images.
Implements:
- Phase 1: Detection-Head-Only Fine-Tuning (~1,281 params)
  (Backbone, DnCNN, Decoders strictly frozen)
- Phase 2: Low-LR Multi-Head Fine-Tuning (decode_head + conf_head at 1e-6, det at 1e-5)
- AdamW, weight decay 1e-4, Cosine Annealing scheduler, BCEWithLogitsLoss
- Early stopping on Validation ROC-AUC (patience 3)
- Real/Fake Score Distribution Tracking (Mean, Median, Margin)
- Saves checkpoints to study/checkpoints_stage1_uncompressed/
- Saves history and metrics to study/results_stage1/
"""

import os
import sys
import json
import time
from pathlib import Path
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

CHECKPOINT_DIR = BASE_DIR / "study" / "checkpoints_stage1_uncompressed"
RESULTS_DIR = BASE_DIR / "study" / "results_stage1"
CACHE_DIR = BASE_DIR / "study" / "cache_features"
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage1"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_detection_head(det_head, features, labels, criterion, device):
    det_head.eval()
    with torch.no_grad():
        x = features.to(device)
        y = labels.to(device)
        logits = det_head(x).squeeze(-1)
        loss = criterion(logits, y).item()
        scores = torch.sigmoid(logits).cpu().numpy()
        y_true = y.cpu().numpy()
        
    auc = roc_auc_score(y_true, scores)
    bacc_05 = balanced_accuracy_score(y_true, (scores >= 0.50).astype(int))
    
    real_sc = scores[y_true == 0]
    fake_sc = scores[y_true == 1]
    
    return loss, auc, bacc_05, real_sc, fake_sc


def run_phase1_training(
    epochs=10,
    lr=1e-5,
    weight_decay=1e-4,
    patience=3,
    batch_size=8,
    is_pilot=False,
    save_prefix="phase1_head_only"
):
    device = torch.device("cpu")
    print("\n" + "=" * 80)
    mode_str = "PILOT RUN (3 Epochs)" if is_pilot else f"FULL RUN ({epochs} Epochs)"
    print(f" [PHASE 1] DETECTION HEAD ONLY FINE-TUNING | {mode_str}")
    print(" Architecture: Backbone, DnCNN, Decode Head, Conf Head STRICTLY FROZEN")
    print(f" Trainable Parameters: Only model.detection (~1,281 params) | Device: {device}")
    print("=" * 80)
    
    # Load cached features
    train_cache = CACHE_DIR / ("train_pilot_features.pt" if is_pilot else "train_features.pt")
    val_cache = CACHE_DIR / ("val_pilot_features.pt" if is_pilot else "val_features.pt")
    
    if not train_cache.exists() or not val_cache.exists():
        raise FileNotFoundError(f"Feature caches not found! Run study/cache_features.py first.")
        
    train_data = torch.load(train_cache, weights_only=False)
    val_data = torch.load(val_cache, weights_only=False)
    
    X_train = train_data["features"]
    y_train = train_data["labels"]
    X_val = val_data["features"]
    y_val = val_data["labels"]
    
    n_train_real = int((y_train == 0).sum().item())
    n_train_fake = int((y_train == 1).sum().item())
    n_val_real = int((y_val == 0).sum().item())
    n_val_fake = int((y_val == 1).sum().item())
    
    print(f"[*] Train set: {len(X_train)} samples | Real={n_train_real}, Fake={n_train_fake}")
    print(f"[*] Val set:   {len(X_val)} samples | Real={n_val_real}, Fake={n_val_fake}")
    
    # Load model and isolate detection head
    full_model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    det_head = full_model.detection.to(device)
    
    optimizer = torch.optim.AdamW(det_head.parameters(), lr=lr, weight_decay=weight_decay)
    total_epochs = 3 if is_pilot else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    
    # Baseline validation evaluation before any fine-tuning
    init_loss, init_auc, init_bacc, init_real_sc, init_fake_sc = evaluate_detection_head(
        det_head, X_val, y_val, criterion, device
    )
    init_real_med = np.median(init_real_sc)
    init_fake_med = np.median(init_fake_sc)
    print("\n[*] Pretrained TruFor Baseline on Validation Set Before Fine-Tuning:")
    print(f"    Val Loss: {init_loss:.4f} | Val AUC: {init_auc*100:.2f}% | Val bACC(0.50): {init_bacc*100:.2f}%")
    print(f"    Real Scores -> Mean: {init_real_sc.mean():.4f} | Median: {init_real_med:.4f} | Std: {init_real_sc.std():.4f}")
    print(f"    Fake Scores -> Mean: {init_fake_sc.mean():.4f} | Median: {init_fake_med:.4f} | Std: {init_fake_sc.std():.4f}")
    print(f"    Score Margin (Fake Median - Real Median): {init_fake_med - init_real_med:+.4f}\n")
    
    history = []
    best_val_auc = init_auc
    best_epoch = 0
    best_ckpt_path = None
    no_improve_epochs = 0
    
    # Save baseline checkpoint as initial reference
    base_ckpt = CHECKPOINT_DIR / f"{save_prefix}_epoch0_baseline.pth"
    torch.save({
        "epoch": 0,
        "state_dict": full_model.state_dict(),
        "val_auc": init_auc,
        "val_bacc": init_bacc
    }, base_ckpt)
    
    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        det_head.train()
        
        # Shuffle train data
        indices = torch.randperm(len(X_train))
        X_train_shuff = X_train[indices]
        y_train_shuff = y_train[indices]
        
        running_loss = 0.0
        n_batches = int(np.ceil(len(X_train) / batch_size))
        
        for b in range(n_batches):
            b_start = b * batch_size
            b_end = min(b_start + batch_size, len(X_train))
            xb = X_train_shuff[b_start:b_end].to(device)
            yb = y_train_shuff[b_start:b_end].to(device)
            
            optimizer.zero_grad()
            logits = det_head(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det_head.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * (b_end - b_start)
            
        scheduler.step()
        train_loss = running_loss / len(X_train)
        
        # Validation Evaluation
        val_loss, val_auc, val_bacc, val_real_sc, val_fake_sc = evaluate_detection_head(
            det_head, X_val, y_val, criterion, device
        )
        elapsed = time.time() - t0
        
        real_med = np.median(val_real_sc)
        fake_med = np.median(val_fake_sc)
        margin = fake_med - real_med
        cur_lr = scheduler.get_last_lr()[0]
        
        print(f"[Epoch {epoch:02d}/{total_epochs:02d}] ({elapsed:.2f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc*100:.2f}% | Val bACC(0.50): {val_bacc*100:.2f}%")
        print(f"             Real Mean: {val_real_sc.mean():.4f} (Med {real_med:.4f}) | Fake Mean: {val_fake_sc.mean():.4f} (Med {fake_med:.4f}) | Margin: {margin:+.4f} | LR: {cur_lr:.2e}", flush=True)
        
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_bacc": val_bacc,
            "real_score_mean": float(val_real_sc.mean()),
            "real_score_med": float(real_med),
            "real_score_std": float(val_real_sc.std()),
            "fake_score_mean": float(val_fake_sc.mean()),
            "fake_score_med": float(fake_med),
            "fake_score_std": float(val_fake_sc.std()),
            "margin": float(margin),
            "lr": float(cur_lr)
        }
        history.append(record)
        
        # Checkpoint selection on Val AUC (Rule 1: strictly on validation)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            no_improve_epochs = 0
            best_ckpt_path = CHECKPOINT_DIR / f"{save_prefix}_best.pth"
            # Update full model detection weights and save state dict
            full_model.detection.load_state_dict(det_head.state_dict())
            torch.save({
                "epoch": epoch,
                "state_dict": full_model.state_dict(),
                "val_auc": val_auc,
                "val_bacc": val_bacc,
                "history": history
            }, best_ckpt_path)
            print(f"  [+] Saved new best model checkpoint to {best_ckpt_path.name} (Val AUC: {val_auc*100:.2f}%)", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience and not is_pilot:
                print(f"\n[!] Early stopping triggered: Val AUC has not improved for {patience} consecutive epochs.")
                break

    # Save training history
    prefix = f"{save_prefix}_pilot" if is_pilot else save_prefix
    hist_file = RESULTS_DIR / f"{prefix}_history.json"
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"\n[*] Training Complete. Best Epoch: {best_epoch} | Best Val AUC: {best_val_auc*100:.2f}%")
    return best_ckpt_path, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run 3-epoch pilot run")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs for full run")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()
    
    run_phase1_training(epochs=args.epochs, lr=args.lr, is_pilot=args.pilot)
