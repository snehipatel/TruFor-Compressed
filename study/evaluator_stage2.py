"""
Stage 2 Evaluator: Comprehensive evaluation of TruFor on compressed test sets.
Evaluates pretrained baseline and fine-tuned models on:
  - CocoGlide, Columbia, CASIA 1.0
  - x5 conditions: original, whatsapp, instagram, facebook, telegram
With:
  - AUC + 95% bootstrap CI
  - Balanced accuracy at tau=0.50 and validation-calibrated tau
  - TPR, TNR
  - Score distributions (mean, median, std)
  - Robustness gap analysis
  - Per-platform threshold calibration
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from PIL import Image
import torch

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

DATASET_DIR = BASE_DIR / "DATASET"
RESULTS_DIR = BASE_DIR / "study" / "results_stage2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage2"

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}
PLATFORMS = ["original", "whatsapp", "instagram", "facebook", "telegram"]


def bootstrap_auc_ci(y_true, scores, n_bootstrap=2000, ci=0.95, seed=42):
    """Compute AUC with bootstrap 95% confidence interval."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    scores = np.array(scores)
    
    # Point estimate
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0
    
    point_auc = roc_auc_score(y_true, scores)
    
    boot_aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        s_b = scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, s_b))
    
    boot_aucs = np.array(boot_aucs)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_aucs, alpha * 100)
    hi = np.percentile(boot_aucs, (1 - alpha) * 100)
    
    return point_auc, lo, hi


def get_youden_threshold(scores, labels):
    """Compute optimal threshold via Youden's J statistic."""
    taus = np.linspace(0.01, 0.99, 200)
    best_j = -1.0
    best_tau = 0.50
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    for tau in taus:
        preds = (scores >= tau).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = tpr + tnr - 1.0
        
        if j > best_j:
            best_j = j
            best_tau = tau
    
    return best_tau


def collect_compressed_test_suite(max_per_class=50):
    """
    Collect test suites across all platforms.
    If max_per_class is set, limits to that many real and fake images per condition
    (default: 50, exactly matching Stage 1 protocol for verified baseline reproducibility).
    Returns: {dataset: {platform: {"real": [...], "fake": [...]}}}
    """
    suite = {}
    
    # === CocoGlide ===
    cg = {}
    cg_dir = DATASET_DIR / "CocoGlide"
    # Original
    r = sorted([str(f) for f in (cg_dir / "real").glob("*") if f.suffix.lower() in IMG_EXTS])
    f = sorted([str(f) for f in (cg_dir / "fake").glob("*") if f.suffix.lower() in IMG_EXTS])
    if max_per_class:
        r = r[:max_per_class]
        f = f[:max_per_class]
    cg["original"] = {"real": r, "fake": f}
    # Compressed
    for p in ["whatsapp", "instagram", "facebook", "telegram"]:
        r_dir = cg_dir / p / "real"
        f_dir = cg_dir / p / "fake"
        if r_dir.exists() and f_dir.exists():
            r = sorted([str(x) for x in r_dir.glob("*") if x.suffix.lower() in IMG_EXTS])
            f = sorted([str(x) for x in f_dir.glob("*") if x.suffix.lower() in IMG_EXTS])
            if max_per_class:
                r = r[:max_per_class]
                f = f[:max_per_class]
            cg[p] = {"real": r, "fake": f}
        else:
            cg[p] = {"real": [], "fake": []}
    suite["CocoGlide"] = cg
    
    # === Columbia ===
    col = {}
    col_dir = DATASET_DIR / "Columbia"
    r = sorted([str(f) for f in (col_dir / "4cam_auth").glob("*") 
                if f.suffix.lower() in IMG_EXTS and f.is_file()])
    f = sorted([str(f) for f in (col_dir / "4cam_splc").glob("*") 
                if f.suffix.lower() in IMG_EXTS and f.is_file()])
    if max_per_class:
        r = r[:max_per_class]
        f = f[:max_per_class]
    col["original"] = {"real": r, "fake": f}
    for p in ["whatsapp", "instagram", "facebook", "telegram"]:
        r_dir = col_dir / p / "4cam_auth"
        f_dir = col_dir / p / "4cam_splc"
        if r_dir.exists() and f_dir.exists():
            r = sorted([str(x) for x in r_dir.glob("*") if x.suffix.lower() in IMG_EXTS])
            f = sorted([str(x) for x in f_dir.glob("*") if x.suffix.lower() in IMG_EXTS])
            if max_per_class:
                r = r[:max_per_class]
                f = f[:max_per_class]
            col[p] = {"real": r, "fake": f}
        else:
            col[p] = {"real": [], "fake": []}
    suite["Columbia"] = col
    
    # === CASIA 1.0 ===
    c1 = {}
    c1_base = DATASET_DIR / "CASIA1.0" / "CASIA 1.0 dataset"
    r = sorted([str(f) for f in (c1_base / "Au" / "Au").glob("*") if f.suffix.lower() in IMG_EXTS])
    f_cm = sorted([str(f) for f in (c1_base / "Modified Tp" / "Tp" / "CM").glob("*") if f.suffix.lower() in IMG_EXTS])
    f_sp = sorted([str(f) for f in (c1_base / "Modified Tp" / "Tp" / "Sp").glob("*") if f.suffix.lower() in IMG_EXTS])
    f_all = f_cm + f_sp
    if max_per_class:
        r = r[:max_per_class]
        f_all = f_all[:max_per_class]
    c1["original"] = {"real": r, "fake": f_all}
    for p in ["whatsapp", "instagram", "facebook", "telegram"]:
        r_dir = DATASET_DIR / "CASIA1.0" / p / "Au"
        f_dir = DATASET_DIR / "CASIA1.0" / p / "Modified Tp"
        if r_dir.exists() and f_dir.exists():
            r = sorted([str(x) for x in r_dir.glob("*") if x.suffix.lower() in IMG_EXTS])
            f = sorted([str(x) for x in f_dir.rglob("*") if x.suffix.lower() in IMG_EXTS])
            if max_per_class:
                r = r[:max_per_class]
                f = f[:max_per_class]
            c1[p] = {"real": r, "fake": f}
        else:
            c1[p] = {"real": [], "fake": []}
    suite["CASIA1.0"] = c1
    
    return suite


def evaluate_suite(model, suite, device, val_tau=0.50, model_name="Pretrained"):
    """Evaluate model on entire test suite with CIs."""
    results = {}
    
    for ds_name, platforms in suite.items():
        ds_results = {}
        for plat, pair in platforms.items():
            real_imgs = pair["real"]
            fake_imgs = pair["fake"]
            
            if not real_imgs or not fake_imgs:
                print(f"  [{ds_name}/{plat}] Empty — skipping")
                continue
            
            y_true = []
            scores = []
            
            for p in real_imgs:
                try:
                    s, _, _ = evaluate_single_image(model, p, device)
                    scores.append(s)
                    y_true.append(0)
                except Exception as e:
                    pass
            
            for p in fake_imgs:
                try:
                    s, _, _ = evaluate_single_image(model, p, device)
                    scores.append(s)
                    y_true.append(1)
                except Exception as e:
                    pass
            
            y_true = np.array(y_true)
            scores_arr = np.array(scores)
            
            if len(np.unique(y_true)) < 2:
                print(f"  [{ds_name}/{plat}] Only one class — skipping")
                continue
            
            # AUC with CI
            auc, auc_lo, auc_hi = bootstrap_auc_ci(y_true, scores_arr)
            
            # Metrics at tau=0.50
            preds_50 = (scores_arr >= 0.50).astype(int)
            bacc_50 = balanced_accuracy_score(y_true, preds_50)
            tp50 = np.sum((preds_50 == 1) & (y_true == 1))
            fn50 = np.sum((preds_50 == 0) & (y_true == 1))
            tn50 = np.sum((preds_50 == 0) & (y_true == 0))
            fp50 = np.sum((preds_50 == 1) & (y_true == 0))
            tpr_50 = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
            tnr_50 = tn50 / (tn50 + fp50) if (tn50 + fp50) > 0 else 0.0
            
            # Metrics at val_tau
            preds_vt = (scores_arr >= val_tau).astype(int)
            bacc_vt = balanced_accuracy_score(y_true, preds_vt)
            tp_vt = np.sum((preds_vt == 1) & (y_true == 1))
            fn_vt = np.sum((preds_vt == 0) & (y_true == 1))
            tn_vt = np.sum((preds_vt == 0) & (y_true == 0))
            fp_vt = np.sum((preds_vt == 1) & (y_true == 0))
            tpr_vt = tp_vt / (tp_vt + fn_vt) if (tp_vt + fn_vt) > 0 else 0.0
            tnr_vt = tn_vt / (tn_vt + fp_vt) if (tn_vt + fp_vt) > 0 else 0.0
            
            # Score distributions
            real_sc = scores_arr[y_true == 0]
            fake_sc = scores_arr[y_true == 1]
            
            ds_results[plat] = {
                "n_real": len(real_sc),
                "n_fake": len(fake_sc),
                "AUC": float(auc * 100),
                "AUC_lo": float(auc_lo * 100),
                "AUC_hi": float(auc_hi * 100),
                "bACC_050": float(bacc_50 * 100),
                "TPR_050": float(tpr_50 * 100),
                "TNR_050": float(tnr_50 * 100),
                "val_tau": float(val_tau),
                "bACC_vtau": float(bacc_vt * 100),
                "TPR_vtau": float(tpr_vt * 100),
                "TNR_vtau": float(tnr_vt * 100),
                "real_mean": float(real_sc.mean()),
                "real_median": float(np.median(real_sc)),
                "real_std": float(real_sc.std()),
                "fake_mean": float(fake_sc.mean()),
                "fake_median": float(np.median(fake_sc)),
                "fake_std": float(fake_sc.std()),
                "scores": [float(x) for x in scores_arr],
                "y_true": [int(x) for x in y_true],
            }
            
            print(f"  [{ds_name}/{plat}] AUC={auc*100:.2f}% [{auc_lo*100:.1f}-{auc_hi*100:.1f}] | "
                  f"bACC(0.5)={bacc_50*100:.1f}% | n={len(real_sc)}r+{len(fake_sc)}f", flush=True)
        
        results[ds_name] = ds_results
    
    return results


def calibrate_per_platform_thresholds(model, val_manifest_path, device, max_per_platform=50):
    """
    Calibrate per-platform thresholds on validation set.
    Samples up to max_per_platform real and fake per platform for balanced, efficient calibration.
    Returns: {platform: optimal_tau}
    """
    with open(val_manifest_path, "r") as f:
        val_samples = json.load(f)
    
    # Filter to balanced subsets per platform
    from collections import defaultdict
    by_plat_label = defaultdict(list)
    for s in val_samples:
        p = s.get("platform", "original")
        lbl = s["label"]
        by_plat_label[(p, lbl)].append(s)
    
    selected_samples = []
    for (p, lbl), s_list in by_plat_label.items():
        selected_samples.extend(s_list[:max_per_platform])
    
    print(f"[*] Calibrating on {len(selected_samples)} validation samples ({max_per_platform}r+{max_per_platform}f per platform)...")
    
    plat_scores = {p: {"scores": [], "labels": []} for p in PLATFORMS}
    for s in selected_samples:
        p = s.get("platform", "original")
        if p not in plat_scores:
            p = "original"
        try:
            sc, _, _ = evaluate_single_image(model, s["image_path"], device)
            plat_scores[p]["scores"].append(sc)
            plat_scores[p]["labels"].append(s["label"])
        except Exception:
            pass
    
    thresholds = {}
    for plat, data in plat_scores.items():
        if len(data["scores"]) > 10 and len(set(data["labels"])) == 2:
            tau = get_youden_threshold(data["scores"], data["labels"])
            auc = roc_auc_score(data["labels"], data["scores"]) * 100
            thresholds[plat] = tau
            print(f"  [{plat}] tau*={tau:.3f} | val AUC={auc:.1f}% | n={len(data['scores'])}", flush=True)
        else:
            thresholds[plat] = 0.50
            print(f"  [{plat}] Insufficient data, using tau=0.50", flush=True)
    
    return thresholds


def evaluate_with_per_platform_thresholds(model, suite, device, plat_thresholds, precomputed_results=None):
    """
    Evaluate using per-platform calibrated thresholds.
    If precomputed_results is provided, reuses existing scores and labels instantly!
    """
    results = {}
    
    for ds_name, platforms in suite.items():
        ds_results = {}
        for plat, pair in platforms.items():
            if precomputed_results and ds_name in precomputed_results and plat in precomputed_results[ds_name]:
                # Instant computation using precomputed scores
                prec = precomputed_results[ds_name][plat]
                y_true = np.array(prec.get("y_true", []))
                scores_arr = np.array(prec.get("scores", []))
            else:
                real_imgs = pair["real"]
                fake_imgs = pair["fake"]
                if not real_imgs or not fake_imgs:
                    continue
                
                y_true = []
                scores = []
                for p in real_imgs:
                    try:
                        s, _, _ = evaluate_single_image(model, p, device)
                        scores.append(s)
                        y_true.append(0)
                    except:
                        pass
                for p in fake_imgs:
                    try:
                        s, _, _ = evaluate_single_image(model, p, device)
                        scores.append(s)
                        y_true.append(1)
                    except:
                        pass
                
                y_true = np.array(y_true)
                scores_arr = np.array(scores)
            
            if len(np.unique(y_true)) < 2:
                continue
            
            tau = plat_thresholds.get(plat, 0.50)
            preds = (scores_arr >= tau).astype(int)
            bacc = balanced_accuracy_score(y_true, preds) * 100
            auc = roc_auc_score(y_true, scores_arr) * 100
            
            tp = np.sum((preds == 1) & (y_true == 1))
            fn = np.sum((preds == 0) & (y_true == 1))
            tn = np.sum((preds == 0) & (y_true == 0))
            fp = np.sum((preds == 1) & (y_true == 0))
            tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
            
            ds_results[plat] = {
                "tau": float(tau),
                "AUC": float(auc),
                "bACC": float(bacc),
                "TPR": float(tpr),
                "TNR": float(tnr),
            }
            
            print(f"  [{ds_name}/{plat}] tau={tau:.3f} | AUC={auc:.1f}% | bACC={bacc:.1f}%", flush=True)
        
        results[ds_name] = ds_results
    
    return results


def run_full_evaluation(checkpoint_path=None, model_name="Pretrained", val_tau=0.50):
    """Run full evaluation pipeline."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 80)
    print(f" STAGE 2 EVALUATION: {model_name.upper()}")
    print("=" * 80)
    
    # Load model
    if checkpoint_path is None:
        model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    else:
        model, _ = get_trufor_model(weights_path=checkpoint_path, fine_tune_mode="detection_only", device=device)
    model.eval()
    
    # Collect test suite
    print("\n[*] Collecting test suites...")
    suite = collect_compressed_test_suite()
    for ds, plats in suite.items():
        for p, pair in plats.items():
            print(f"  {ds}/{p}: {len(pair['real'])} real, {len(pair['fake'])} fake")
    
    # Evaluate
    print(f"\n[*] Evaluating {model_name}...")
    t0 = time.time()
    results = evaluate_suite(model, suite, device, val_tau=val_tau, model_name=model_name)
    elapsed = time.time() - t0
    print(f"[*] Evaluation took {elapsed:.1f}s")
    
    # Save results
    out_file = RESULTS_DIR / f"eval_{model_name.lower().replace(' ', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "val_tau": val_tau,
            "results": results,
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"[*] Results saved to {out_file}")
    
    return results


if __name__ == "__main__":
    run_full_evaluation()
