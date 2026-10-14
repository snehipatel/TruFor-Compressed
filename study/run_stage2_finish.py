"""
Stage 2 Completion Script: Executes remaining steps:
  Step 7: Full Test Evaluation of Head-Only and Decoder+Head Fine-Tuned Models
  Step 8: Leave-One-Platform-Out (LOPO) Generalization Test
  Step 9: Simulator Realism Audit
  Step 10: Final Synthesis & Comprehensive Report Generation
"""

import os
import sys
import json
import time
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
from study.evaluator_stage2 import (
    collect_compressed_test_suite,
    evaluate_suite,
    calibrate_per_platform_thresholds,
    evaluate_with_per_platform_thresholds,
    bootstrap_auc_ci,
)
from study.run_stage2 import generate_final_report, step9_simulator_realism_audit

RESULTS_DIR = BASE_DIR / "study" / "results_stage2"
CHECKPOINTS_DIR = BASE_DIR / "study" / "checkpoints_stage2"
SPLITS_DIR = BASE_DIR / "study" / "data_splits_stage2"
CACHE_DIR = BASE_DIR / "study" / "cache_features"

device = torch.device("cpu")


def run_finish():
    print("=" * 80)
    print(" STAGE 2 PIPELINE COMPLETION (STEPS 7 -> 10)")
    print("=" * 80)
    
    # Load Pretrained Baseline Results
    bl_file = RESULTS_DIR / "eval_pretrained_baseline.json"
    with open(bl_file, "r") as f:
        bl_data = json.load(f)
    baseline_results = bl_data["results"]
    print("[+] Loaded Pretrained Baseline results (15 conditions)")
    
    # Load Calibration Results
    cal_file = RESULTS_DIR / "per_platform_calibration.json"
    with open(cal_file, "r") as f:
        calibration_results = json.load(f)
    thresholds = calibration_results["thresholds"]
    print(f"[+] Loaded Calibrated Thresholds: {thresholds}")
    
    # -------------------------------------------------------------------------
    # STEP 7A: Evaluate Head-Only Fine-Tuned Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 7A: EVALUATING HEAD-ONLY FINE-TUNED MODEL")
    print("=" * 80)
    
    head_ckpt = CHECKPOINTS_DIR / "stage2_head_only_best.pth"
    head_model, _ = get_trufor_model(weights_path=str(head_ckpt), fine_tune_mode="detection_only", device=device)
    head_model.eval()
    
    # Calibrate thresholds for head-only model on val
    val_manifest = SPLITS_DIR / "val_manifest.json"
    head_thresh = calibrate_per_platform_thresholds(head_model, val_manifest, device, max_per_platform=50)
    
    # Fast evaluation using baseline features or evaluate suite
    suite = collect_compressed_test_suite(max_per_class=50)
    print("\n[*] Evaluating Head-Only model on all 15 conditions...")
    head_results = evaluate_suite(head_model, suite, device, val_tau=0.50, model_name="HeadOnly_FineTuned")
    
    head_plat_results = evaluate_with_per_platform_thresholds(
        head_model, suite, device, head_thresh, precomputed_results=head_results
    )
    
    with open(RESULTS_DIR / "eval_head_only_finetuned.json", "w", encoding="utf-8") as f:
        json.dump({"results": head_results, "per_platform_results": head_plat_results, "thresholds": head_thresh}, f, indent=2)
    print("[+] Saved Head-Only evaluation results")

    # -------------------------------------------------------------------------
    # STEP 7B: Evaluate Decoder+Head Fine-Tuned Model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 7B: EVALUATING DECODER+HEAD FINE-TUNED MODEL (MAIN EXPERIMENT)")
    print("=" * 80)
    
    dec_ckpt = CHECKPOINTS_DIR / "stage2_decoder_head_best.pth"
    dec_model, _ = get_trufor_model(weights_path=str(dec_ckpt), fine_tune_mode="heads_only", device=device)
    dec_model.eval()
    
    dec_thresh = calibrate_per_platform_thresholds(dec_model, val_manifest, device, max_per_platform=50)
    
    print("\n[*] Evaluating Decoder+Head model on all 15 conditions...")
    dec_results = evaluate_suite(dec_model, suite, device, val_tau=0.50, model_name="DecoderHead_FineTuned")
    
    dec_plat_results = evaluate_with_per_platform_thresholds(
        dec_model, suite, device, dec_thresh, precomputed_results=dec_results
    )
    
    with open(RESULTS_DIR / "eval_decoder_head_finetuned.json", "w", encoding="utf-8") as f:
        json.dump({"results": dec_results, "per_platform_results": dec_plat_results, "thresholds": dec_thresh}, f, indent=2)
    print("[+] Saved Decoder+Head evaluation results")

    # -------------------------------------------------------------------------
    # STEP 8: Leave-One-Platform-Out (LOPO) Test
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 8: LEAVE-ONE-PLATFORM-OUT (LOPO) GENERALIZATION TEST")
    print("=" * 80)
    
    lopo_cache = CACHE_DIR / "stage2_train_feats_40.pt"
    platforms = ["whatsapp", "instagram", "facebook", "telegram"]
    lopo_results = {}
    
    if lopo_cache.exists():
        t_data = torch.load(lopo_cache, weights_only=False)
        t_feats, t_labels, t_plats = t_data["feats"], t_data["labels"], t_data["plats"]
        
        for held_out in platforms:
            print(f"\n--- LOPO: Holding out {held_out.upper()} ---")
            keep_mask = torch.tensor([p != held_out for p in t_plats], dtype=torch.bool)
            
            lopo_m, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
            opt = torch.optim.AdamW(lopo_m.detection.parameters(), lr=1e-5, weight_decay=1e-4)
            crit = torch.nn.BCEWithLogitsLoss()
            
            lf = t_feats[keep_mask].to(device)
            ll = t_labels[keep_mask].to(device)
            
            lopo_m.detection.train()
            for _ in range(15):
                opt.zero_grad()
                out = lopo_m.detection(lf).squeeze(-1)
                loss = crit(out, ll)
                loss.backward()
                opt.step()
            lopo_m.detection.eval()
            print(f"  [+] Trained LOPO model on {keep_mask.sum().item()} samples (omitted {held_out})")
            
            # Evaluate on held-out platform test sets
            lopo_results[held_out] = {}
            for ds_name in ["CocoGlide", "Columbia", "CASIA1.0"]:
                pair = suite[ds_name][held_out]
                real_imgs = pair["real"]
                fake_imgs = pair["fake"]
                
                scores = []
                y_true = []
                from study.evaluator import evaluate_single_image
                for p in real_imgs:
                    try:
                        s, _, _ = evaluate_single_image(lopo_m, p, device)
                        scores.append(s)
                        y_true.append(0)
                    except Exception:
                        pass
                for p in fake_imgs:
                    try:
                        s, _, _ = evaluate_single_image(lopo_m, p, device)
                        scores.append(s)
                        y_true.append(1)
                    except Exception:
                        pass
                
                y_true = np.array(y_true)
                scores = np.array(scores)
                if len(np.unique(y_true)) >= 2:
                    auc, lo, hi = bootstrap_auc_ci(y_true, scores)
                    bacc = balanced_accuracy_score(y_true, (scores >= 0.5).astype(int)) * 100
                    print(f"  {ds_name}/{held_out}: AUC={auc*100:.2f}% [{lo*100:.1f}-{hi*100:.1f}] | bACC={bacc:.1f}%")
                    lopo_results[held_out][ds_name] = {
                        "AUC": float(auc * 100),
                        "AUC_lo": float(lo * 100),
                        "AUC_hi": float(hi * 100),
                        "bACC": float(bacc),
                    }
        
        with open(RESULTS_DIR / "lopo_results.json", "w", encoding="utf-8") as f:
            json.dump(lopo_results, f, indent=2)
    
    # -------------------------------------------------------------------------
    # STEP 9: Simulator Realism Audit
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 9: SIMULATOR REALISM AUDIT")
    print("=" * 80)
    audit_results = step9_simulator_realism_audit()
    
    # -------------------------------------------------------------------------
    # STEP 10: Final Synthesis & Report Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 10: GENERATING FINAL REPORT")
    print("=" * 80)
    best_results = dec_results if dec_results else head_results
    report_file = generate_final_report(
        baseline_results=baseline_results,
        calibration_results=calibration_results,
        finetuned_results=best_results,
        head_only_results=head_results,
        lopo_results=lopo_results,
        audit_results=audit_results,
    )
    print(f"\n[+] Final Report successfully created at {report_file}")


if __name__ == "__main__":
    run_finish()
