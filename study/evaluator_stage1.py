"""
Stage 1 Comprehensive Evaluator for Uncompressed Test Datasets.
Evaluates:
- Pretrained TruFor Baseline
- Phase 1 (Detection-Only Fine-Tuned)
- Phase 2 (Multi-Head Fine-Tuned)

Strictly on Uncompressed Test Sets:
- CocoGlide (512 Real / 512 Fake)
- Columbia (183 Real / 180 Fake)
- CASIA 1.0 (800 Real / 921 Fake)

Thresholds Evaluated:
- Author Default: tau = 0.50
- Validation-Calibrated Optimal: tau* (Youden's J on uncompressed validation set ONLY)
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import torch
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
from study.dataset_builder_stage1 import is_compressed_path

DATASET_DIR = BASE_DIR / "DATASET"
RESULTS_DIR = BASE_DIR / "study" / "results_stage1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage1"


def get_validation_threshold(val_scores, val_labels):
    """
    Computes optimal threshold tau* on the validation set using Youden's J statistic:
    J = TPR + TNR - 1
    """
    taus = np.linspace(0.01, 0.99, 100)
    best_j = -1.0
    best_tau = 0.50
    
    for tau in taus:
        preds = (val_scores >= tau).astype(int)
        tp = np.sum((preds == 1) & (val_labels == 1))
        fn = np.sum((preds == 0) & (val_labels == 1))
        tn = np.sum((preds == 0) & (val_labels == 0))
        fp = np.sum((preds == 1) & (val_labels == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = tpr + tnr - 1.0
        
        if j > best_j:
            best_j = j
            best_tau = tau
            
    return best_tau


def load_uncompressed_test_suites(max_per_class=100):
    """
    Loads uncompressed test suites strictly isolated from training/validation.
    """
    suites = {}
    
    # 1. CocoGlide
    cg_dir = DATASET_DIR / "CocoGlide"
    r_cg = sorted(list((cg_dir / "real").glob("*.png")) + list((cg_dir / "real").glob("*.jpg")))[:max_per_class]
    f_cg = sorted(list((cg_dir / "fake").glob("*.png")) + list((cg_dir / "fake").glob("*.jpg")))[:max_per_class]
    suites["CocoGlide"] = {"real": [str(x) for x in r_cg], "fake": [str(x) for x in f_cg]}
    
    # 2. Columbia
    col_dir = DATASET_DIR / "Columbia"
    r_col = sorted(list((col_dir / "4cam_auth").glob("*.tif")))[:max_per_class]
    f_col = sorted(list((col_dir / "4cam_splc").glob("*.tif")))[:max_per_class]
    suites["Columbia"] = {"real": [str(x) for x in r_col], "fake": [str(x) for x in f_col]}
    
    # 3. CASIA 1.0
    c1_dir = DATASET_DIR / "CASIA1.0" / "CASIA 1.0 dataset"
    r_c1 = sorted(list((c1_dir / "Au" / "Au").glob("*.jpg")) + list((c1_dir / "Au").glob("*.jpg")))[:max_per_class]
    f_c1 = sorted(list((c1_dir / "Modified Tp" / "Tp" / "Sp").glob("*.jpg")) + list((c1_dir / "Modified Tp" / "Tp" / "CM").glob("*.jpg")))[:max_per_class]
    suites["CASIA1.0"] = {"real": [str(x) for x in r_c1], "fake": [str(x) for x in f_c1]}
    
    return suites


def evaluate_model_on_suite(model, suites, device, val_tau=0.50):
    results = {}
    
    for ds_name, pairs in suites.items():
        y_true = []
        scores = []
        
        for p in pairs["real"]:
            s, _, _ = evaluate_single_image(model, p, device)
            scores.append(s)
            y_true.append(0)
            
        for p in pairs["fake"]:
            s, _, _ = evaluate_single_image(model, p, device)
            scores.append(s)
            y_true.append(1)
            
        y_true = np.array(y_true)
        scores = np.array(scores)
        
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
        bacc_val, tpr_val, tnr_val = calc_metrics(val_tau)
        
        real_sc = scores[y_true == 0]
        fake_sc = scores[y_true == 1]
        
        results[ds_name] = {
            "AUC": auc,
            "bACC_050": bacc_50,
            "TPR_050": tpr_50,
            "TNR_050": tnr_50,
            "val_tau": val_tau,
            "bACC_val_tau": bacc_val,
            "TPR_val_tau": tpr_val,
            "TNR_val_tau": tnr_val,
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
        
    return results


def run_full_stage1_evaluation(checkpoint_path=None, model_name="Fine-Tuned"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f" EVALUATING {model_name.upper()} ON UNCOMPRESSED BENCHMARK")
    print("=" * 80)
    
    # 1. Load model
    if checkpoint_path is None:
        model, _ = get_trufor_model(fine_tune_mode="heads_only", device=device)
    else:
        model, _ = get_trufor_model(weights_path=checkpoint_path, fine_tune_mode="heads_only", device=device)
    model.eval()
    
    # 2. Get optimal threshold tau* on validation set
    with open(SPLITS_DIR / "val_manifest.json", "r") as f:
        val_samples = json.load(f)
        
    val_scores = []
    val_labels = []
    for s in val_samples:
        img_p = str(BASE_DIR / s["image_path"])
        sc, _, _ = evaluate_single_image(model, img_p, device)
        val_scores.append(sc)
        val_labels.append(s["label"])
        
    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    
    val_tau = get_validation_threshold(val_scores, val_labels)
    val_auc = roc_auc_score(val_labels, val_scores) * 100
    val_bacc_50 = balanced_accuracy_score(val_labels, (val_scores >= 0.50).astype(int)) * 100
    val_bacc_tau = balanced_accuracy_score(val_labels, (val_scores >= val_tau).astype(int)) * 100
    
    print(f"[*] Validation Performance: AUC={val_auc:.2f}% | bACC(0.50)={val_bacc_50:.2f}% | Optimal tau*={val_tau:.2f} | bACC(tau*)={val_bacc_tau:.2f}%")
    
    # 3. Evaluate on test suites
    suites = load_uncompressed_test_suites(max_per_class=50)
    test_results = evaluate_model_on_suite(model, suites, device, val_tau=val_tau)
    
    # Print clean summary
    print("\n" + "=" * 80)
    print(f" {model_name} RESULTS SUMMARY ON UNCOMPRESSED TEST SUITES")
    print("=" * 80)
    print(f"Validation Calibrated Threshold: tau* = {val_tau:.2f}\n")
    print("| Dataset | AUC (%) | bACC (0.50) | bACC (tau*) | TPR (tau*) | TNR (tau*) | Margin (Med) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for ds, r in test_results.items():
        print(f"| {ds:<10} | {r['AUC']:6.2f}% | {r['bACC_050']:6.2f}% | {r['bACC_val_tau']:6.2f}% | {r['TPR_val_tau']:6.2f}% | {r['TNR_val_tau']:6.2f}% | {r['margin']:+6.4f} |")
        
    # Save results json
    out_file = RESULTS_DIR / f"eval_{model_name.lower().replace(' ', '_')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "val_performance": {
                "AUC": val_auc,
                "bACC_050": val_bacc_50,
                "val_tau": val_tau,
                "bACC_tau": val_bacc_tau,
                "val_real_mean": float(val_scores[val_labels==0].mean()),
                "val_fake_mean": float(val_scores[val_labels==1].mean()),
            },
            "test_performance": test_results
        }, f, indent=2)
        
    return test_results
