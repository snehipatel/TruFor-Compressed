"""
Stage 2 Trainer: Fine-tuning TruFor for compressed-image robustness.

Experiments:
  1. Head-only: freeze everything except detection head (optimized with cached features)
  2. Decoder+head: unfreeze decode_head + decode_head_conf at 1e-6, detection at 1e-5
  
Features:
  - AdamW with weight decay + EWC-style penalty pulling toward pretrained weights
  - Balanced 50/50 Real/Fake sampling per platform (Original, WhatsApp, Instagram, Facebook, Telegram)
  - Early stopping on validation AUC (measured on compressed val images)
  - Checkpoint selection on validation only
  - Score distribution tracking
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
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

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from study.model_manager import get_trufor_model
from study.evaluator import evaluate_single_image
from test_docker.src.models.cmx.layer_utils import weighted_statistics_pooling

CHECKPOINT_DIR = BASE_DIR / "study" / "checkpoints_stage2"
RESULTS_DIR = BASE_DIR / "study" / "results_stage2"
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage2"
CACHE_DIR = BASE_DIR / "study" / "cache_features"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLATFORMS = ["original", "whatsapp", "instagram", "facebook", "telegram"]


def select_balanced_subset(samples, max_per_platform_class=40, seed=42):
    """
    Selects a balanced subset: exactly max_per_platform_class real and fake
    per platform (5 platforms: original, whatsapp, instagram, facebook, telegram).
    """
    import random
    by_plat_class = defaultdict(list)
    for s in samples:
        p = s.get("platform", "original")
        lbl = s["label"]
        by_plat_class[(p, lbl)].append(s)
    
    selected = []
    for (p, lbl), s_list in sorted(by_plat_class.items()):
        selected.extend(s_list[:max_per_platform_class])
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def extract_detection_features(model, samples, device, max_size=512):
    """Extracts deterministic 8-dimensional detection feature vectors from frozen model."""
    model.eval()
    feats = []
    labels = []
    platforms = []
    t0 = time.time()
    for i, s in enumerate(samples):
        img_path = s["image_path"]
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
            out, conf, _, _ = model(t_rgb)
            f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
            f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)).view(out.shape[0], -1)
            feat = torch.cat((f1, f2), -1).squeeze(0).cpu()
        feats.append(feat)
        labels.append(float(s["label"]))
        platforms.append(s.get("platform", "original"))
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"    [{i+1}/{len(samples)}] Extracted features ({time.time()-t0:.1f}s)", flush=True)
    return torch.stack(feats), torch.tensor(labels, dtype=torch.float32), platforms


def score_samples(model, samples, device, max_samples=None):
    """Score a list of manifest samples and return scores, labels."""
    model.eval()
    scores = []
    labels = []
    platforms = []
    
    items = samples if max_samples is None else samples[:max_samples]
    
    with torch.no_grad():
        for s in items:
            try:
                sc, _, _ = evaluate_single_image(model, s["image_path"], device)
                scores.append(sc)
                labels.append(s["label"])
                platforms.append(s.get("platform", "original"))
            except Exception:
                pass
    
    return np.array(scores), np.array(labels), platforms


def ewc_penalty(model, pretrained_state, ewc_lambda=0.1):
    """
    Elastic Weight Consolidation-style penalty: L2 distance from pretrained weights.
    Only applied to trainable (unfrozen) parameters.
    """
    penalty = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in pretrained_state:
            penalty += ((param - pretrained_state[name].to(param.device)) ** 2).sum()
    return ewc_lambda * penalty


def run_stage2_training(
    mode="head_only",  # "head_only" or "decoder_head"
    epochs=15,
    det_lr=1e-5,
    decoder_lr=1e-6,
    weight_decay=1e-4,
    ewc_lambda=0.05,
    patience=4,
    max_train_per_plat_class=40,
    max_val_per_plat_class=10,
    seed=42,
    save_prefix=None,
):
    """
    Stage 2 fine-tuning for compressed image robustness.
    """
    if save_prefix is None:
        save_prefix = f"stage2_{mode}"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("\n" + "=" * 80)
    print(f" STAGE 2 FINE-TUNING: {mode.upper()}")
    print(f" Device: {device} | Max Epochs: {epochs} | Patience: {patience}")
    print(f" Det LR: {det_lr} | Decoder LR: {decoder_lr} | EWC lambda: {ewc_lambda}")
    print("=" * 80)
    
    # Load manifests
    with open(SPLITS_DIR / "train_manifest.json", "r") as f:
        full_train = json.load(f)
    with open(SPLITS_DIR / "val_manifest.json", "r") as f:
        full_val = json.load(f)
    
    # Select balanced subsets
    train_samples = select_balanced_subset(full_train, max_per_platform_class=max_train_per_plat_class, seed=seed)
    val_samples = select_balanced_subset(full_val, max_per_platform_class=max_val_per_plat_class, seed=seed)
    
    print(f"\n[*] Balanced subsets: Train={len(train_samples)} ({max_train_per_plat_class}r+{max_train_per_plat_class}f x 5 plats) | "
          f"Val={len(val_samples)} ({max_val_per_plat_class}r+{max_val_per_plat_class}f x 5 plats)")
    
    # Load model
    if mode == "head_only":
        model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    elif mode == "decoder_head":
        model, _ = get_trufor_model(fine_tune_mode="heads_only", device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Save pretrained state for EWC penalty
    pretrained_state = {name: param.clone().detach() 
                        for name, param in model.named_parameters() 
                        if param.requires_grad}
    
    # -------------------------------------------------------------
    # FAST PATH FOR HEAD-ONLY: Feature Caching
    # -------------------------------------------------------------
    if mode == "head_only":
        print("\n[*] Mode is head_only: Extracting / caching 8-dim features for instant training...")
        train_cache = CACHE_DIR / f"stage2_train_feats_{max_train_per_plat_class}.pt"
        val_cache = CACHE_DIR / f"stage2_val_feats_{max_val_per_plat_class}.pt"
        
        if train_cache.exists():
            print(f"  [+] Loading cached train features from {train_cache.name}")
            t_data = torch.load(train_cache, weights_only=False)
            train_feats, train_labels, train_plats = t_data["feats"], t_data["labels"], t_data["plats"]
        else:
            print("  [*] Extracting train features...")
            train_feats, train_labels, train_plats = extract_detection_features(model, train_samples, device)
            torch.save({"feats": train_feats, "labels": train_labels, "plats": train_plats}, train_cache)
            print(f"  [+] Saved {len(train_feats)} features to {train_cache.name}")
        
        if val_cache.exists():
            print(f"  [+] Loading cached val features from {val_cache.name}")
            v_data = torch.load(val_cache, weights_only=False)
            val_feats, val_labels, val_plats = v_data["feats"], v_data["labels"], v_data["plats"]
        else:
            print("  [*] Extracting val features...")
            val_feats, val_labels, val_plats = extract_detection_features(model, val_samples, device)
            torch.save({"feats": val_feats, "labels": val_labels, "plats": val_plats}, val_cache)
            print(f"  [+] Saved {len(val_feats)} features to {val_cache.name}")
        
        # Train head directly
        trainable = [p for p in model.detection.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=det_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        criterion = nn.BCEWithLogitsLoss()
        
        best_comp_auc = -1.0
        best_epoch = 0
        best_ckpt_path = None
        no_improve = 0
        history = []
        
        val_feats_dev = val_feats.to(device)
        train_feats_dev = train_feats.to(device)
        train_labels_dev = train_labels.to(device)
        
        for epoch in range(1, epochs + 1):
            model.detection.train()
            optimizer.zero_grad()
            
            det_logits = model.detection(train_feats_dev).squeeze(-1)
            loss = criterion(det_logits, train_labels_dev)
            
            if ewc_lambda > 0:
                ewc = ewc_penalty(model, pretrained_state, ewc_lambda)
                loss = loss + ewc
            else:
                ewc = torch.tensor(0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Validation
            model.detection.eval()
            with torch.no_grad():
                val_logits = model.detection(val_feats_dev).squeeze(-1)
                val_scores = torch.sigmoid(val_logits).cpu().numpy()
            
            y_val = val_labels.numpy()
            val_auc = roc_auc_score(y_val, val_scores)
            val_bacc = balanced_accuracy_score(y_val, (val_scores >= 0.5).astype(int))
            
            # Per-platform val metrics
            plat_aucs = {}
            for plat in PLATFORMS:
                mask = np.array([p == plat for p in val_plats])
                if mask.sum() > 5 and len(np.unique(y_val[mask])) >= 2:
                    plat_aucs[plat] = roc_auc_score(y_val[mask], val_scores[mask]) * 100
            
            # Compressed-only val AUC
            comp_mask = np.array([p != "original" for p in val_plats])
            comp_auc = roc_auc_score(y_val[comp_mask], val_scores[comp_mask])
            
            real_sc = val_scores[y_val == 0]
            fake_sc = val_scores[y_val == 1]
            
            plat_str = " | ".join(f"{p}:{a:.1f}" for p, a in plat_aucs.items())
            print(f"[Epoch {epoch:02d}/{epochs}] Loss: {loss.item():.4f} | EWC: {ewc.item():.4f} | "
                  f"Val AUC: {val_auc*100:.2f}% | Comp AUC: {comp_auc*100:.2f}% | bACC: {val_bacc*100:.1f}%\n"
                  f"  Platform AUCs: {plat_str}\n"
                  f"  Real: {real_sc.mean():.4f}+-{real_sc.std():.4f} | Fake: {fake_sc.mean():.4f}+-{fake_sc.std():.4f}",
                  flush=True)
            
            record = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "ewc_loss": ewc.item(),
                "val_auc": float(val_auc * 100),
                "comp_val_auc": float(comp_auc * 100),
                "val_bacc": float(val_bacc * 100),
                "plat_aucs": plat_aucs,
            }
            history.append(record)
            
            if comp_auc > best_comp_auc:
                best_comp_auc = comp_auc
                best_epoch = epoch
                no_improve = 0
                best_ckpt_path = CHECKPOINT_DIR / f"{save_prefix}_best.pth"
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_auc": float(val_auc * 100),
                    "comp_val_auc": float(comp_auc * 100),
                    "history": history,
                }, best_ckpt_path)
                print(f"  [+] New best checkpoint (comp AUC: {comp_auc*100:.2f}%)")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n[!] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
        
        hist_file = RESULTS_DIR / f"{save_prefix}_history.json"
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"\n[*] Training complete. Best epoch: {best_epoch} | Best comp val AUC: {best_comp_auc*100:.2f}%")
        return best_ckpt_path, history

    # -------------------------------------------------------------
    # DECODER + HEAD END-TO-END TRAINING
    # -------------------------------------------------------------
    det_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "detection" in name:
            det_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": decoder_params, "lr": decoder_lr},
        {"params": det_params, "lr": det_lr},
    ], weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    
    best_comp_auc = -1.0
    best_epoch = 0
    best_ckpt_path = None
    no_improve = 0
    history = []
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        
        if hasattr(model, 'dncnn'):
            model.dncnn.eval()
        if hasattr(model, 'backbone'):
            model.backbone.eval()
        if hasattr(model, 'backbone_conf') and model.backbone_conf is not None:
            model.backbone_conf.eval()
        
        indices = np.random.permutation(len(train_samples))
        running_loss = 0.0
        running_ewc = 0.0
        n_processed = 0
        
        for i_step, idx in enumerate(indices):
            s = train_samples[idx]
            try:
                img_path = s["image_path"]
                label = float(s["label"])
                
                pil_img = Image.open(img_path).convert("RGB")
                w, h = pil_img.size
                max_size = 512
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
                
                pred_loc, pred_conf, pred_det, _ = model(t_rgb)
                
                target = torch.tensor([label], device=device)
                det_logit = pred_det.squeeze()
                loss = criterion(det_logit.unsqueeze(0), target)
                
                if ewc_lambda > 0:
                    ewc = ewc_penalty(model, pretrained_state, ewc_lambda)
                    loss = loss + ewc
                    running_ewc += ewc.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item()
                n_processed += 1
                
                if (i_step + 1) % 50 == 0:
                    print(f"    Epoch {epoch} [{i_step+1}/{len(indices)}] loss={loss.item():.4f}", flush=True)
            except Exception:
                continue
        
        scheduler.step()
        avg_loss = running_loss / max(n_processed, 1)
        avg_ewc = running_ewc / max(n_processed, 1)
        
        # Validation
        val_scores, val_labels, val_plats = score_samples(model, val_samples, device)
        val_auc = roc_auc_score(val_labels, val_scores)
        val_bacc = balanced_accuracy_score(val_labels, (val_scores >= 0.5).astype(int))
        
        comp_mask = np.array([p != "original" for p in val_plats])
        comp_auc = roc_auc_score(val_labels[comp_mask], val_scores[comp_mask])
        
        plat_aucs = {}
        for plat in PLATFORMS:
            mask = np.array([p == plat for p in val_plats])
            if mask.sum() > 5 and len(np.unique(val_labels[mask])) >= 2:
                plat_aucs[plat] = roc_auc_score(val_labels[mask], val_scores[mask]) * 100
        
        elapsed = time.time() - t0
        plat_str = " | ".join(f"{p}:{a:.1f}" for p, a in plat_aucs.items())
        print(f"[Epoch {epoch:02d}/{epochs}] ({elapsed:.1f}s) | Loss: {avg_loss:.4f} | EWC: {avg_ewc:.4f} | "
              f"Val AUC: {val_auc*100:.2f}% | Comp AUC: {comp_auc*100:.2f}% | bACC: {val_bacc*100:.1f}%\n"
              f"  Platform AUCs: {plat_str}",
              flush=True)
        
        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "ewc_loss": avg_ewc,
            "val_auc": float(val_auc * 100),
            "comp_val_auc": float(comp_auc * 100),
            "val_bacc": float(val_bacc * 100),
            "plat_aucs": plat_aucs,
            "elapsed": elapsed,
        }
        history.append(record)
        
        if comp_auc > best_comp_auc:
            best_comp_auc = comp_auc
            best_epoch = epoch
            no_improve = 0
            best_ckpt_path = CHECKPOINT_DIR / f"{save_prefix}_best.pth"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_auc": float(val_auc * 100),
                "comp_val_auc": float(comp_auc * 100),
                "history": history,
            }, best_ckpt_path)
            print(f"  [+] New best checkpoint (comp AUC: {comp_auc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[!] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    
    hist_file = RESULTS_DIR / f"{save_prefix}_history.json"
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[*] Training complete. Best epoch: {best_epoch} | Best comp val AUC: {best_comp_auc*100:.2f}%")
    return best_ckpt_path, history
