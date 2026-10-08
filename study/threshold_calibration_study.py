"""
Validation-Calibrated Threshold Search and Score Distribution Analysis
for TruFor Platform Compression Robustness Study.

Methodology:
1. Evaluate 4 models:
   - Baseline (Pretrained)
   - Approach 2 (No-Aug)
   - Approach 2 (Aggressive-Aug)
   - Approach 2 (Mild-Aug)
2. Validation Set: Balanced across (Real/Tampered) and all 5 transmission channels.
3. Generate score distributions (Mean, Std, Median, IQR, Min, Max) for Real vs Tampered.
4. Sweep decision thresholds from 0.10 to 0.90 (step 0.05).
5. Identify optimal threshold tau* based STRICTLY on the VALIDATION SET.
6. Freeze tau* and evaluate on UNSEEN TEST DATASETS (CocoGlide, Columbia, CASIA 1.0 across 5 platforms).
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Add root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from study.model_manager import get_trufor_model
from study.evaluator import evaluate_single_image, collect_test_dataset_pairs

RESULTS_DIR = BASE_DIR / "study" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_balanced_val_suite(manifest_path, per_platform=60):
    """
    Builds a balanced validation set from val_manifest.json:
    per_platform real and per_platform fake for each of 5 platforms.
    Total = 10 * per_platform (e.g. 600 samples).
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
        
    platforms = ["original", "whatsapp", "instagram", "facebook", "telegram"]
    buckets = defaultdict(list)
    for s in samples:
        buckets[(s["label"], s["platform"])].append(s)
        
    val_set = []
    for p in platforms:
        reals = buckets[(0, p)][:per_platform]
        fakes = buckets[(1, p)][:per_platform]
        val_set.extend(reals)
        val_set.extend(fakes)
        
    return val_set


def run_or_load_predictions(model_name, weights_path, val_samples, test_suite, device):
    cache_file = RESULTS_DIR / f"predictions_{model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}.json"
    
    if cache_file.exists():
        print(f"[*] Loading cached predictions for {model_name} from {cache_file.name}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print(f"\n[*] Running inference for: {model_name}...")
    model, _ = get_trufor_model(weights_path=str(weights_path), device=device)
    model.eval()
    
    # 1. Validation predictions
    print(f"    Evaluating {len(val_samples)} validation images...")
    val_preds = []
    t0 = time.time()
    for s in val_samples:
        img_path = str(BASE_DIR / s["image_path"])
        try:
            score, anom_area, peak_anom = evaluate_single_image(model, img_path, device)
            val_preds.append({
                "score": float(score),
                "anom_area": float(anom_area),
                "peak_anom": float(peak_anom),
                "label": int(s["label"]),
                "platform": s["platform"],
                "dataset": s["dataset"]
            })
        except Exception as e:
            print(f"[!] Error on {img_path}: {e}")
    dt_val = time.time() - t0
    print(f"    Validation inference completed in {dt_val:.1f}s ({len(val_preds)/max(1e-3, dt_val):.1f} img/s)")
    
    # 2. Test predictions
    print("    Evaluating test benchmark suite (1500 images)...")
    test_preds = []
    t0 = time.time()
    for ds_name, p_dict in test_suite.items():
        for p, pair in p_dict.items():
            for pth in pair["real"]:
                try:
                    score, anom_area, peak_anom = evaluate_single_image(model, pth, device)
                    test_preds.append({
                        "score": float(score),
                        "anom_area": float(anom_area),
                        "peak_anom": float(peak_anom),
                        "label": 0,
                        "platform": p,
                        "dataset": ds_name
                    })
                except Exception:
                    pass
            for pth in pair["fake"]:
                try:
                    score, anom_area, peak_anom = evaluate_single_image(model, pth, device)
                    test_preds.append({
                        "score": float(score),
                        "anom_area": float(anom_area),
                        "peak_anom": float(peak_anom),
                        "label": 1,
                        "platform": p,
                        "dataset": ds_name
                    })
                except Exception:
                    pass
    dt_test = time.time() - t0
    print(f"    Test inference completed in {dt_test:.1f}s ({len(test_preds)/max(1e-3, dt_test):.1f} img/s)")
    
    del model
    torch.cuda.empty_cache()
    
    result = {
        "model_name": model_name,
        "val_preds": val_preds,
        "test_preds": test_preds
    }
    
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    return result


def compute_score_stats(scores):
    if len(scores) == 0:
        return {}
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def evaluate_threshold_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": threshold,
        "bACC": bacc,
        "F1": f1,
        "TPR": tpr,
        "TNR": tnr,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Threshold Calibration Study on {device}")
    
    val_manifest = BASE_DIR / "study" / "data_splits" / "val_manifest.json"
    val_samples = build_balanced_val_suite(val_manifest, per_platform=60)
    print(f"[*] Prepared balanced validation set: {len(val_samples)} images (300 real, 300 fake)")
    
    test_suite = collect_test_dataset_pairs(max_per_class=50)
    total_test = sum(len(p["real"]) + len(p["fake"]) for ds in test_suite.values() for p in ds.values())
    print(f"[*] Prepared unseen test suite: {total_test} images (750 real, 750 fake)")
    
    models = {
        "Baseline (Pretrained)": BASE_DIR / "pretrained_models" / "trufor.pth.tar",
        "Approach 2 (No-Aug)": BASE_DIR / "study" / "checkpoints" / "exp2_without_aug.pth",
        "Approach 2 (Aggressive-Aug)": BASE_DIR / "study" / "checkpoints" / "exp2_with_aug.pth",
        "Approach 2 (Mild-Aug)": BASE_DIR / "study" / "checkpoints" / "exp2_mild_aug.pth",
    }
    
    all_data = {}
    for m_name, m_path in models.items():
        if not Path(m_path).exists():
            print(f"[!] Warning: weights not found for {m_name}: {m_path}")
            continue
        all_data[m_name] = run_or_load_predictions(m_name, m_path, val_samples, test_suite, device)
        
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]
    
    summary_report = {
        "score_distributions": {},
        "val_threshold_sweep": {},
        "optimal_val_thresholds": {},
        "test_results_at_frozen_threshold": {},
        "test_results_at_050": {},
        "test_results_at_030": {}
    }
    
    # -------------------------------------------------------------
    # 1. SCORE DISTRIBUTIONS (REAL VS TAMPERED)
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print(" 1. SCORE DISTRIBUTIONS: REAL VS TAMPERED (VALIDATION SET)")
    print("="*85)
    print(f"{'MODEL':<28} {'CLASS':<10} {'MEAN':<7} {'STD':<7} {'MEDIAN':<7} {'IQR (25-75%)':<14} {'RANGE':<12}")
    print("-" * 85)
    
    for m_name, d in all_data.items():
        val_preds = d["val_preds"]
        real_scores = [p["score"] for p in val_preds if p["label"] == 0]
        fake_scores = [p["score"] for p in val_preds if p["label"] == 1]
        
        real_stats = compute_score_stats(real_scores)
        fake_stats = compute_score_stats(fake_scores)
        
        summary_report["score_distributions"][m_name] = {
            "Real": real_stats,
            "Tampered": fake_stats
        }
        
        iqr_r = f"{real_stats['p25']:.3f}-{real_stats['p75']:.3f}"
        range_r = f"{real_stats['min']:.2f}-{real_stats['max']:.2f}"
        print(f"{m_name:<28} {'Real':<10} {real_stats['mean']:.4f}  {real_stats['std']:.4f}  {real_stats['median']:.4f}  {iqr_r:<14} {range_r:<12}")
        
        iqr_f = f"{fake_stats['p25']:.3f}-{fake_stats['p75']:.3f}"
        range_f = f"{fake_stats['min']:.2f}-{fake_stats['max']:.2f}"
        print(f"{'':<28} {'Tampered':<10} {fake_stats['mean']:.4f}  {fake_stats['std']:.4f}  {fake_stats['median']:.4f}  {iqr_f:<14} {range_f:<12}")
        print("-" * 85)
        
    # -------------------------------------------------------------
    # 2. VALIDATION THRESHOLD SWEEP (0.10 to 0.90)
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print(" 2. VALIDATION THRESHOLD SWEEP (0.10 - 0.90) & OPTIMAL SELECTION")
    print(" (Threshold selected using ONLY the validation set, strictly zero test data)")
    print("="*85)
    
    best_thresholds = {}
    
    for m_name, d in all_data.items():
        val_preds = d["val_preds"]
        y_true = np.array([p["label"] for p in val_preds])
        scores = np.array([p["score"] for p in val_preds])
        
        sweep_results = []
        best_bacc = -1.0
        best_tau = 0.50
        
        for tau in thresholds:
            m = evaluate_threshold_metrics(y_true, scores, tau)
            sweep_results.append(m)
            if m["bACC"] > best_bacc:
                best_bacc = m["bACC"]
                best_tau = tau
                
        best_thresholds[m_name] = best_tau
        summary_report["val_threshold_sweep"][m_name] = sweep_results
        summary_report["optimal_val_thresholds"][m_name] = {
            "best_threshold": best_tau,
            "best_val_bacc": best_bacc
        }
        
        print(f"\n---> Model: {m_name}")
        print(f"     Optimal Frozen Validation Threshold: tau* = {best_tau:.2f} (Val bACC = {best_bacc*100:.2f}%)")
        print(f"     {'tau':<6} {'bACC':<9} {'F1':<9} {'TPR':<9} {'TNR':<9}")
        print("     " + "-"*40)
        for r in sweep_results:
            is_best = " <-- BEST VAL tau*" if r["threshold"] == best_tau else ""
            print(f"     {r['threshold']:<6.2f} {r['bACC']*100:>6.2f}%   {r['F1']*100:>6.2f}%   {r['TPR']*100:>6.2f}%   {r['TNR']*100:>6.2f}%{is_best}")
            
    # -------------------------------------------------------------
    # 3. REPORT ON UNSEEN TEST SETS AT FROZEN THRESHOLDS
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print(" 3. UNSEEN TEST SET PERFORMANCE AT FROZEN VALIDATION THRESHOLD (tau*)")
    print(" (Zero Leakage: tau* was selected exclusively on Val and frozen for Unseen Test Suite)")
    print("="*95)
    
    test_eval_summary = []
    
    for m_name, d in all_data.items():
        test_preds = d["test_preds"]
        y_true = np.array([p["label"] for p in test_preds])
        scores = np.array([p["score"] for p in test_preds])
        
        frozen_tau = best_thresholds[m_name]
        
        # Unseen Test ROC-AUC
        test_auc = roc_auc_score(y_true, scores)
        
        # Test Metrics at Frozen Threshold tau*
        m_frozen = evaluate_threshold_metrics(y_true, scores, frozen_tau)
        m_050 = evaluate_threshold_metrics(y_true, scores, 0.50)
        m_030 = evaluate_threshold_metrics(y_true, scores, 0.30)
        
        # Per dataset breakdown at frozen tau*
        ds_breakdown = {}
        for ds in ["CocoGlide", "Columbia", "CASIA1.0"]:
            idx = [i for i, p in enumerate(test_preds) if p["dataset"] == ds]
            ds_y = y_true[idx]
            ds_s = scores[idx]
            ds_auc = roc_auc_score(ds_y, ds_s)
            ds_metrics = evaluate_threshold_metrics(ds_y, ds_s, frozen_tau)
            ds_breakdown[ds] = {
                "AUC": ds_auc,
                "bACC": ds_metrics["bACC"],
                "F1": ds_metrics["F1"],
                "TPR": ds_metrics["TPR"],
                "TNR": ds_metrics["TNR"]
            }
            
        test_eval_summary.append({
            "model": m_name,
            "frozen_tau": frozen_tau,
            "test_auc": test_auc,
            "frozen_bacc": m_frozen["bACC"],
            "frozen_f1": m_frozen["F1"],
            "frozen_tpr": m_frozen["TPR"],
            "frozen_tnr": m_frozen["TNR"],
            "bacc_050": m_050["bACC"],
            "bacc_030": m_030["bACC"],
            "ds_breakdown": ds_breakdown
        })
        
    print(f"{'MODEL':<28} {'FROZEN tau*':<12} {'TEST AUC':<10} {'TEST bACC(tau*)':<16} {'bACC(0.50)':<12} {'bACC(0.30)':<12} {'TEST F1(tau*)':<12}")
    print("-" * 105)
    for r in test_eval_summary:
        print(f"{r['model']:<28} {r['frozen_tau']:<12.2f} {r['test_auc']*100:>6.2f}%    {r['frozen_bacc']*100:>6.2f}%         {r['bacc_050']*100:>6.2f}%      {r['bacc_030']*100:>6.2f}%      {r['frozen_f1']*100:>6.2f}%")
        
    print("\n" + "="*95)
    print(" DATASET-BY-DATASET BREAKDOWN AT FROZEN VALIDATION THRESHOLD (tau*)")
    print("="*95)
    for ds in ["CocoGlide", "Columbia", "CASIA1.0"]:
        print(f"\n--- Dataset: {ds} ---")
        print(f"{'MODEL':<28} {'FROZEN tau*':<12} {'AUC':<10} {'bACC(tau*)':<14} {'TPR(tau*)':<12} {'TNR(tau*)':<12} {'F1(tau*)':<12}")
        print("-" * 95)
        for r in test_eval_summary:
            m = r["ds_breakdown"][ds]
            print(f"{r['model']:<28} {r['frozen_tau']:<12.2f} {m['AUC']*100:>6.2f}%    {m['bACC']*100:>6.2f}%        {m['TPR']*100:>6.2f}%      {m['TNR']*100:>6.2f}%      {m['F1']*100:>6.2f}%")
            
    # Save full JSON report
    out_json = RESULTS_DIR / "threshold_calibration_study.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "test_summary": test_eval_summary,
            "full_report": summary_report
        }, f, indent=2)
    print(f"\n[OK] Complete Threshold Calibration Study saved to: {out_json}")


if __name__ == "__main__":
    main()
