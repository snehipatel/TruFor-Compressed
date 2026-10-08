"""
Training and Validation Engine for TruFor Compression Robustness Fine-Tuning.
Optimized for local execution on NVIDIA RTX 5050 Laptop GPU (8GB VRAM):
- Mixed Precision (AMP FP16)
- Gradient Accumulation (accum=4, effective batch=4)
- Class-Balanced Epoch Sampling
- Multi-metric Validation & Best Checkpoint Tracking
"""

import os
import sys
import time
import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from study.dataset import RobustnessDataset, BalancedClassSampler
from study.model_manager import get_trufor_model, MultiTaskLoss

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "study" / "checkpoints"
LOGS_DIR = BASE_DIR / "study" / "logs"


def run_training(
    epochs=5,
    lr=1e-4,
    batch_size=1,
    accum_steps=4,
    samples_per_epoch=400,
    val_samples=100,
    fine_tune_mode="heads_only",
    save_name="best_robust_model.pth",
    use_compression_aug=False
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    print(f"[*] Compression-aware Domain Randomization Augmentation: {'ENABLED' if use_compression_aug else 'DISABLED'}")
    
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Datasets & Loaders
    train_manifest = BASE_DIR / "study" / "data_splits" / "train_manifest.json"
    val_manifest = BASE_DIR / "study" / "data_splits" / "val_manifest.json"
    
    if not train_manifest.exists() or not val_manifest.exists():
        from study.dataset_builder import build_train_val_manifests
        print("[*] Manifests not found. Building now...")
        build_train_val_manifests()
        
    train_dataset = RobustnessDataset(str(train_manifest), is_training=True, use_compression_aug=use_compression_aug)
    val_dataset = RobustnessDataset(str(val_manifest), is_training=False, use_compression_aug=False)
    
    train_sampler = BalancedClassSampler(train_dataset, num_samples_per_epoch=samples_per_epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    
    val_sampler = BalancedClassSampler(val_dataset, num_samples_per_epoch=val_samples)
    val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=0)
    
    # 2. Model & Optimizer
    model, trainable = get_trufor_model(fine_tune_mode=fine_tune_mode, device=device)
    
    if fine_tune_mode in ["heads_only", "detection_only"]:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    elif fine_tune_mode == "differential":
        optimizer = torch.optim.AdamW([
            {"params": trainable[0]["params"], "lr": lr * trainable[0]["lr_scale"]},
            {"params": trainable[1]["params"], "lr": lr * trainable[1]["lr_scale"]},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = MultiTaskLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    best_val_bacc = 0.0
    log_csv = LOGS_DIR / "training_log.csv"
    
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_bacc", "val_auc", "lr"])
        
    print("\n" + "="*70)
    print(f" STARTING PLATFORM ROBUSTNESS FINE-TUNING ({epochs} EPOCHS)")
    print(f" Effective Batch Size: {batch_size * accum_steps} | Steps/Epoch: {len(train_loader)}")
    print("="*70 + "\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        model.dncnn.eval() # Keep frozen
        optimizer.zero_grad()
        
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        
        for step, batch in enumerate(pbar):
            rgb = batch["rgb"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred_loc, pred_conf, pred_det, _ = model(rgb)
                loss, loss_dict = criterion(pred_loc, pred_conf, pred_det, mask, label)
                loss_scaled = loss / accum_steps
                
            scaler.scale(loss_scaled).backward()
            train_losses.append(loss.item())
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            pbar.set_postfix({"loss": f"{np.mean(train_losses[-20:]):.4f}"})
            
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        torch.cuda.empty_cache()
        val_losses = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                rgb = batch["rgb"].to(device)
                mask = batch["mask"].to(device)
                label = batch["label"].to(device)
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    pred_loc, pred_conf, pred_det, _ = model(rgb)
                    loss, _ = criterion(pred_loc, pred_conf, pred_det, mask, label)
                    
                val_losses.append(loss.item())
                det_score = torch.sigmoid(pred_det).item()
                all_scores.append(det_score)
                all_labels.append(int(label.item()))
                
        avg_val_loss = np.mean(val_losses)
        y_true = np.array(all_labels)
        y_scores = np.array(all_scores)
        y_pred = (y_scores >= 0.5).astype(int)
        
        try:
            val_bacc = balanced_accuracy_score(y_true, y_pred)
        except Exception:
            val_bacc = 0.5
        try:
            val_auc = roc_auc_score(y_true, y_scores)
        except Exception:
            val_auc = 0.5
            
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val bACC: {val_bacc*100:.2f}% | Val AUC: {val_auc:.4f} | LR: {curr_lr:.2e}")
        
        # Save to log CSV
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{val_bacc:.4f}", f"{val_auc:.4f}", f"{curr_lr:.2e}"])
            
        # Best model checkpoint
        if val_bacc >= best_val_bacc:
            best_val_bacc = val_bacc
            ckpt_path = CHECKPOINTS_DIR / save_name
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_bacc": val_bacc,
                "val_auc": val_auc,
                "val_loss": avg_val_loss
            }, ckpt_path)
            print(f"[*] NEW BEST CHECKPOINT SAVED -> {ckpt_path} (bACC: {val_bacc*100:.2f}%)")
            
    print("\n[OK] Fine-tuning finished! Best checkpoint stored at:", CHECKPOINTS_DIR / save_name)
    return CHECKPOINTS_DIR / save_name
