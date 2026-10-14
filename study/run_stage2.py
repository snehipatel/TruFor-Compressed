"""
Stage 2 Master Orchestrator: Runs the full compressed-image fine-tuning pipeline.

Steps:
  1. Generate compressed test sets (if not already done)
  2. Build Stage 2 manifests (train/val with original + compressed versions)
  3. Evaluate pretrained baseline on compressed test sets (full, with CIs)
  4. Per-platform threshold calibration (no-training improvement)
  5. Fine-tune: head-only variant
  6. Fine-tune: decoder+head variant (main experiment)
  7. Evaluate best fine-tuned model on all test sets
  8. Leave-one-platform-out generalization test
  9. Simulator realism audit
  10. Generate final report
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

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

RESULTS_DIR = BASE_DIR / "study" / "results_stage2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import torch


def step1_generate_compressed_tests():
    """Generate compressed test sets using the simulator."""
    print("\n" + "=" * 80)
    print(" STEP 1: GENERATE COMPRESSED TEST SETS")
    print("=" * 80)
    
    from study.generate_compressed_test_sets import main as gen_main
    gen_main()


def step2_build_manifests():
    """Build Stage 2 train/val manifests."""
    print("\n" + "=" * 80)
    print(" STEP 2: BUILD STAGE 2 MANIFESTS")
    print("=" * 80)
    
    from study.dataset_builder_stage2 import build_stage2_manifests
    train_samples, val_samples = build_stage2_manifests()
    return train_samples, val_samples


def step3_baseline_evaluation():
    """Evaluate pretrained TruFor on compressed test sets."""
    print("\n" + "=" * 80)
    print(" STEP 3: PRETRAINED BASELINE EVALUATION")
    print("=" * 80)
    
    from study.evaluator_stage2 import run_full_evaluation
    results = run_full_evaluation(
        checkpoint_path=None, 
        model_name="Pretrained_Baseline",
        val_tau=0.50
    )
    return results


def step4_per_platform_calibration(baseline_results=None):
    """Per-platform threshold calibration on pretrained model."""
    print("\n" + "=" * 80)
    print(" STEP 4: PER-PLATFORM THRESHOLD CALIBRATION")
    print("=" * 80)
    
    from study.evaluator_stage2 import (
        calibrate_per_platform_thresholds, 
        evaluate_with_per_platform_thresholds,
        collect_compressed_test_suite
    )
    from study.model_manager import get_trufor_model
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    model.eval()
    
    val_manifest = BASE_DIR / "study" / "data_splits_stage2" / "val_manifest.json"
    print("\n[*] Calibrating per-platform thresholds on validation set...")
    thresholds = calibrate_per_platform_thresholds(model, val_manifest, device)
    
    print("\n[*] Evaluating with per-platform thresholds on test sets...")
    suite = collect_compressed_test_suite()
    results = evaluate_with_per_platform_thresholds(model, suite, device, thresholds, precomputed_results=baseline_results)
    
    # Save
    out = {"thresholds": {k: float(v) for k, v in thresholds.items()}, "results": results}
    with open(RESULTS_DIR / "per_platform_calibration.json", "w") as f:
        json.dump(out, f, indent=2)
    
    return thresholds, results


def step5_train_head_only():
    """Fine-tune head-only variant."""
    print("\n" + "=" * 80)
    print(" STEP 5: HEAD-ONLY FINE-TUNING")
    print("=" * 80)
    
    from study.trainer_stage2 import run_stage2_training
    ckpt, history = run_stage2_training(
        mode="head_only",
        epochs=10,
        det_lr=1e-5,
        ewc_lambda=0.05,
        patience=4,
        save_prefix="stage2_head_only",
    )
    return ckpt


def step6_train_decoder_head():
    """Fine-tune decoder+head variant (main experiment)."""
    print("\n" + "=" * 80)
    print(" STEP 6: DECODER+HEAD FINE-TUNING (MAIN EXPERIMENT)")
    print("=" * 80)
    
    from study.trainer_stage2 import run_stage2_training
    ckpt, history = run_stage2_training(
        mode="decoder_head",
        epochs=15,
        det_lr=1e-5,
        decoder_lr=1e-6,
        ewc_lambda=0.05,
        patience=5,
        save_prefix="stage2_decoder_head",
    )
    return ckpt


def step7_evaluate_finetuned(ckpt_path, model_name="FineTuned"):
    """Evaluate the best fine-tuned model."""
    print("\n" + "=" * 80)
    print(f" STEP 7: EVALUATE {model_name.upper()}")
    print("=" * 80)
    
    if ckpt_path is None or not os.path.exists(str(ckpt_path)):
        print("[!] No checkpoint found — skipping")
        return None
    
    from study.evaluator_stage2 import run_full_evaluation
    
    # Get optimal threshold from validation
    from study.evaluator_stage2 import (
        calibrate_per_platform_thresholds, 
        evaluate_with_per_platform_thresholds,
        collect_compressed_test_suite
    )
    from study.model_manager import get_trufor_model
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ft_mode = "heads_only" if "decoder" in model_name.lower() else "detection_only"
    model, _ = get_trufor_model(weights_path=str(ckpt_path), fine_tune_mode=ft_mode, device=device)
    model.eval()
    
    # Calibrate threshold on val
    val_manifest = BASE_DIR / "study" / "data_splits_stage2" / "val_manifest.json"
    thresholds = calibrate_per_platform_thresholds(model, val_manifest, device)
    
    # Use overall threshold for the main evaluation
    overall_tau = np.mean(list(thresholds.values()))
    
    results = run_full_evaluation(
        checkpoint_path=str(ckpt_path),
        model_name=model_name,
        val_tau=overall_tau,
    )
    
    # Also evaluate with per-platform thresholds
    suite = collect_compressed_test_suite()
    plat_results = evaluate_with_per_platform_thresholds(model, suite, device, thresholds, precomputed_results=results)
    
    with open(RESULTS_DIR / f"{model_name.lower()}_per_platform.json", "w") as f:
        json.dump({"thresholds": {k: float(v) for k, v in thresholds.items()}, 
                    "results": plat_results}, f, indent=2)
    
    return results


def step8_leave_one_platform_out(best_ckpt_mode="decoder_head"):
    """Leave-one-platform-out generalization test."""
    print("\n" + "=" * 80)
    print(" STEP 8: LEAVE-ONE-PLATFORM-OUT TEST")
    print("=" * 80)
    
    from study.trainer_stage2 import run_stage2_training, score_samples
    from study.evaluator_stage2 import collect_compressed_test_suite, bootstrap_auc_ci
    from study.model_manager import get_trufor_model
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    platforms = ["whatsapp", "instagram", "facebook", "telegram"]
    splits_dir = BASE_DIR / "study" / "data_splits_stage2"
    
    with open(splits_dir / "train_manifest.json", "r") as f:
        full_train = json.load(f)
    with open(splits_dir / "val_manifest.json", "r") as f:
        full_val = json.load(f)
    
    lopo_results = {}
    
    for held_out in platforms:
        print(f"\n--- Holding out: {held_out.upper()} ---")
        
        # Filter train/val: remove held-out platform entries
        train_filtered = [s for s in full_train if s.get("platform", "original") != held_out]
        val_filtered = [s for s in full_val if s.get("platform", "original") != held_out]
        
        # Save temporary manifests
        lopo_dir = splits_dir / f"lopo_{held_out}"
        lopo_dir.mkdir(exist_ok=True)
        with open(lopo_dir / "train_manifest.json", "w") as f:
            json.dump(train_filtered, f)
        with open(lopo_dir / "val_manifest.json", "w") as f:
            json.dump(val_filtered, f)
        
        print(f"  Train: {len(train_filtered)} | Val: {len(val_filtered)}")
        
        # Train detection head holding out the current platform
        cache_file = BASE_DIR / "study" / "cache_features" / "stage2_train_feats_40.pt"
        model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
        
        if cache_file.exists():
            t_data = torch.load(cache_file, weights_only=False)
            t_feats, t_labels, t_plats = t_data["feats"], t_data["labels"], t_data["plats"]
            keep_mask = torch.tensor([p != held_out for p in t_plats], dtype=torch.bool)
            
            lopo_feats = t_feats[keep_mask].to(device)
            lopo_labels = t_labels[keep_mask].to(device)
            
            # Train head on remaining platforms
            opt = torch.optim.AdamW(model.detection.parameters(), lr=1e-5, weight_decay=1e-4)
            crit = torch.nn.BCEWithLogitsLoss()
            model.detection.train()
            for _ in range(15):
                opt.zero_grad()
                out = model.detection(lopo_feats).squeeze(-1)
                l = crit(out, lopo_labels)
                l.backward()
                opt.step()
            model.detection.eval()
            print(f"  [+] Trained LOPO model on {keep_mask.sum().item()} samples (excluding {held_out})")
        else:
            model.eval()
        
        # Evaluate LOPO model on held-out platform test sets
        suite = collect_compressed_test_suite()
        
        for ds_name, plat_data in suite.items():
            if held_out in plat_data and plat_data[held_out]["real"] and plat_data[held_out]["fake"]:
                y_true = []
                scores = []
                for p in plat_data[held_out]["real"]:
                    try:
                        from study.evaluator import evaluate_single_image
                        s, _, _ = evaluate_single_image(model, p, device)
                        scores.append(s)
                        y_true.append(0)
                    except:
                        pass
                for p in plat_data[held_out]["fake"]:
                    try:
                        s, _, _ = evaluate_single_image(model, p, device)
                        scores.append(s)
                        y_true.append(1)
                    except:
                        pass
                
                y_true = np.array(y_true)
                scores = np.array(scores)
                if len(np.unique(y_true)) >= 2:
                    auc, lo, hi = bootstrap_auc_ci(y_true, scores)
                    bacc = balanced_accuracy_score(y_true, (scores >= 0.5).astype(int)) * 100
                    print(f"  {ds_name}/{held_out}: AUC={auc*100:.2f}% [{lo*100:.1f}-{hi*100:.1f}] | bACC={bacc:.1f}%")
                    
                    if held_out not in lopo_results:
                        lopo_results[held_out] = {}
                    lopo_results[held_out][ds_name] = {
                        "AUC": float(auc * 100),
                        "AUC_lo": float(lo * 100),
                        "AUC_hi": float(hi * 100),
                        "bACC": float(bacc),
                    }
    
    with open(RESULTS_DIR / "lopo_results.json", "w") as f:
        json.dump(lopo_results, f, indent=2)
    
    return lopo_results


def step9_simulator_realism_audit():
    """Check if any real platform-processed images exist and compare with simulator."""
    print("\n" + "=" * 80)
    print(" STEP 9: SIMULATOR REALISM AUDIT")
    print("=" * 80)
    
    from PIL import Image
    import struct
    
    data_dir = BASE_DIR / "DATA"
    sml = Path("D:/sample_images_ml")
    
    # Real platform images
    real_platform_dirs = {
        "telegram": [sml / "telegram_real", sml / "telegram_fake"],
        "facebook": [data_dir / "FACEBOOK" / "facebook_real"],
        "instagram": [data_dir / "INSTAGRAM" / "Instagram_Real"],
        "whatsapp": [data_dir / "WHATSAPP" / "whatsapp_real"],
    }
    
    # Simulator-compressed images (e.g., from raise_whatsapp, IMD2020_instagram)
    sim_dirs = {
        "telegram": [BASE_DIR / "IMD2020_telegram"],
        "facebook": [BASE_DIR / "IMD2020_facebook"],
        "instagram": [BASE_DIR / "IMD2020_instagram"],
        "whatsapp": [BASE_DIR / "IMD2020_whatsapp"],
    }
    
    audit = {}
    
    for plat in ["whatsapp", "instagram", "facebook", "telegram"]:
        print(f"\n--- {plat.upper()} ---")
        audit[plat] = {"real": {}, "simulator": {}}
        
        # Sample real platform images
        for d in real_platform_dirs.get(plat, []):
            if d.exists():
                files = sorted([f for f in d.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])[:5]
                for fp in files:
                    try:
                        with Image.open(fp) as im:
                            w, h = im.size
                            fmt = im.format
                        size = fp.stat().st_size
                        print(f"  REAL: {fp.name} | {w}x{h} | {fmt} | {size/1024:.1f}KB")
                        audit[plat]["real"][fp.name] = {
                            "resolution": f"{w}x{h}",
                            "format": fmt,
                            "size_kb": round(size / 1024, 1),
                        }
                    except Exception as e:
                        print(f"  REAL: {fp.name} | ERROR: {e}")
        
        # Sample simulator images
        for d in sim_dirs.get(plat, []):
            if d.exists():
                # Get a few sample files
                for sub in d.iterdir():
                    if sub.is_dir():
                        files = sorted([f for f in sub.iterdir() 
                                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
                                       and '_mask' not in f.name])[:2]
                        for fp in files:
                            try:
                                with Image.open(fp) as im:
                                    w, h = im.size
                                    fmt = im.format
                                size = fp.stat().st_size
                                print(f"  SIM:  {fp.name} | {w}x{h} | {fmt} | {size/1024:.1f}KB")
                                audit[plat]["simulator"][fp.name] = {
                                    "resolution": f"{w}x{h}",
                                    "format": fmt,
                                    "size_kb": round(size / 1024, 1),
                                }
                            except Exception as e:
                                print(f"  SIM:  {fp.name} | ERROR: {e}")
                        break  # Just check first subfolder
    
    with open(RESULTS_DIR / "simulator_realism_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    
    return audit


def generate_final_report(
    baseline_results, 
    calibration_results, 
    finetuned_results,
    head_only_results,
    lopo_results,
    audit_results,
):
    """Generate the comprehensive markdown report."""
    print("\n" + "=" * 80)
    print(" GENERATING FINAL REPORT")
    print("=" * 80)
    
    report_path = RESULTS_DIR / "STAGE2_COMPRESSED_REPORT.md"
    
    # Load all result files if not passed
    if baseline_results is None:
        bl_file = RESULTS_DIR / "eval_pretrained_baseline.json"
        if bl_file.exists():
            with open(bl_file) as f:
                baseline_results = json.load(f).get("results", {})
    
    if calibration_results is None:
        cal_file = RESULTS_DIR / "per_platform_calibration.json"
        if cal_file.exists():
            with open(cal_file) as f:
                calibration_results = json.load(f)
    
    # Build report
    lines = []
    lines.append("# Stage 2: Compressed-Image TruFor Fine-Tuning Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
    
    lines.append("## 1. Objective\n")
    lines.append("Improve TruFor tampering detection on images compressed by WhatsApp, Instagram, ")
    lines.append("Facebook, and Telegram, while maintaining original-image performance.\n")
    
    lines.append("## 2. Data Summary\n")
    
    # Load manifest counts
    splits_dir = BASE_DIR / "study" / "data_splits_stage2"
    for split in ["train", "val"]:
        mf = splits_dir / f"{split}_manifest.json"
        if mf.exists():
            with open(mf) as f:
                data = json.load(f)
            n_real = sum(1 for e in data if e["label"] == 0)
            n_fake = sum(1 for e in data if e["label"] == 1)
            lines.append(f"**{split.upper()}**: {len(data)} entries ({n_real} real, {n_fake} fake)\n")
            
            # Platform breakdown
            from collections import Counter
            plat_counts = Counter(e.get("platform", "original") for e in data)
            ds_counts = Counter(e["dataset"] for e in data)
            n_inpaint = sum(1 for e in data if "inpainting" in e.get("manipulation", ""))
            
            lines.append(f"- Platform breakdown: {dict(plat_counts)}\n")
            lines.append(f"- Dataset breakdown: {dict(ds_counts)}\n")
            lines.append(f"- Inpainting images: {n_inpaint}\n")
    
    lines.append("\n## 3. Results\n")
    
    # Build comparison table
    lines.append("### 3.1 AUC Results (with 95% Bootstrap CI)\n")
    lines.append("| Dataset | Platform | Pretrained AUC | Pretrained CI | Fine-Tuned AUC | Fine-Tuned CI | Δ AUC |\n")
    lines.append("|:--------|:---------|:--------------:|:-------------:|:--------------:|:-------------:|:-----:|\n")
    
    datasets = ["CocoGlide", "Columbia", "CASIA1.0"]
    platforms = ["original", "whatsapp", "instagram", "facebook", "telegram"]
    
    if baseline_results and finetuned_results:
        for ds in datasets:
            bl_ds = baseline_results.get(ds, {})
            ft_ds = finetuned_results.get(ds, {}) if finetuned_results else {}
            for plat in platforms:
                bl = bl_ds.get(plat, {})
                ft = ft_ds.get(plat, {})
                bl_auc = bl.get("AUC", "-")
                bl_ci = f"[{bl.get('AUC_lo', '-'):.1f}-{bl.get('AUC_hi', '-'):.1f}]" if isinstance(bl.get("AUC_lo"), (int, float)) else "-"
                ft_auc = ft.get("AUC", "-")
                ft_ci = f"[{ft.get('AUC_lo', '-'):.1f}-{ft.get('AUC_hi', '-'):.1f}]" if isinstance(ft.get("AUC_lo"), (int, float)) else "-"
                
                if isinstance(bl_auc, (int, float)) and isinstance(ft_auc, (int, float)):
                    delta = ft_auc - bl_auc
                    delta_str = f"{delta:+.2f}"
                else:
                    delta_str = "-"
                
                bl_str = f"{bl_auc:.2f}%" if isinstance(bl_auc, (int, float)) else bl_auc
                ft_str = f"{ft_auc:.2f}%" if isinstance(ft_auc, (int, float)) else ft_auc
                
                lines.append(f"| {ds} | {plat} | {bl_str} | {bl_ci} | {ft_str} | {ft_ci} | {delta_str} |\n")
    
    # Robustness gap
    lines.append("\n### 3.2 Robustness Gap (Original AUC - Platform AUC)\n")
    lines.append("| Dataset | Platform | Pretrained Gap | Fine-Tuned Gap | Gap Reduction |\n")
    lines.append("|:--------|:---------|:--------------:|:--------------:|:-------------:|\n")
    
    if baseline_results and finetuned_results:
        for ds in datasets:
            bl_ds = baseline_results.get(ds, {})
            ft_ds = finetuned_results.get(ds, {}) if finetuned_results else {}
            bl_orig = bl_ds.get("original", {}).get("AUC", None)
            ft_orig = ft_ds.get("original", {}).get("AUC", None)
            
            for plat in ["whatsapp", "instagram", "facebook", "telegram"]:
                bl_plat = bl_ds.get(plat, {}).get("AUC", None)
                ft_plat = ft_ds.get(plat, {}).get("AUC", None)
                
                bl_gap = f"{bl_orig - bl_plat:.2f}" if isinstance(bl_orig, (int,float)) and isinstance(bl_plat, (int,float)) else "-"
                ft_gap = f"{ft_orig - ft_plat:.2f}" if isinstance(ft_orig, (int,float)) and isinstance(ft_plat, (int,float)) else "-"
                
                if bl_gap != "-" and ft_gap != "-":
                    reduction = float(bl_gap) - float(ft_gap)
                    red_str = f"{reduction:+.2f}"
                else:
                    red_str = "-"
                
                lines.append(f"| {ds} | {plat} | {bl_gap} | {ft_gap} | {red_str} |\n")
    
    # Per-platform calibration
    if calibration_results:
        lines.append("\n### 3.3 Per-Platform Threshold Calibration (Pretrained Model)\n")
        thresholds = calibration_results.get("thresholds", {})
        lines.append(f"Calibrated thresholds: {json.dumps(thresholds, indent=2)}\n\n")
    
    # LOPO results
    if lopo_results:
        lines.append("\n### 3.4 Leave-One-Platform-Out Results\n")
        lines.append("| Held-Out | Dataset | AUC | CI |\n")
        lines.append("|:---------|:--------|:---:|:--:|\n")
        for plat, ds_res in lopo_results.items():
            for ds, metrics in ds_res.items():
                auc = metrics.get("AUC", "-")
                ci = f"[{metrics.get('AUC_lo', '-'):.1f}-{metrics.get('AUC_hi', '-'):.1f}]"
                lines.append(f"| {plat} | {ds} | {auc:.2f}% | {ci} |\n")
    
    # Score distributions
    lines.append("\n### 3.5 Score Distributions\n")
    if baseline_results:
        lines.append("\n**Pretrained Baseline:**\n")
        lines.append("| Dataset | Platform | Real Mean | Real Med | Real Std | Fake Mean | Fake Med | Fake Std |\n")
        lines.append("|:--------|:---------|:---------:|:--------:|:--------:|:---------:|:--------:|:--------:|\n")
        for ds in datasets:
            for plat in platforms:
                r = baseline_results.get(ds, {}).get(plat, {})
                if r:
                    lines.append(f"| {ds} | {plat} | {r.get('real_mean', 0):.4f} | {r.get('real_median', 0):.4f} | {r.get('real_std', 0):.4f} | "
                                f"{r.get('fake_mean', 0):.4f} | {r.get('fake_median', 0):.4f} | {r.get('fake_std', 0):.4f} |\n")
    
    # Simulator audit
    if audit_results:
        lines.append("\n### 3.6 Simulator Realism Audit\n")
        for plat, data in audit_results.items():
            lines.append(f"\n**{plat.upper()}:**\n")
            if data.get("real"):
                lines.append(f"- Real platform images found: {len(data['real'])}\n")
                for fname, info in list(data["real"].items())[:3]:
                    lines.append(f"  - `{fname}`: {info['resolution']}, {info['format']}, {info['size_kb']}KB\n")
            else:
                lines.append(f"- No real platform images found\n")
            if data.get("simulator"):
                lines.append(f"- Simulator images sampled: {len(data['simulator'])}\n")
                for fname, info in list(data["simulator"].items())[:3]:
                    lines.append(f"  - `{fname}`: {info['resolution']}, {info['format']}, {info['size_kb']}KB\n")
    
    # Conclusion
    lines.append("\n## 4. Conclusion\n")
    lines.append("See main text for analysis.\n")
    
    # Limitations
    lines.append("\n## 5. Limitations\n")
    lines.append("- CPU-only training (slow, limits dataset scale)\n")
    lines.append("- Scarce inpainting training data (~100 images)\n")
    lines.append("- CASIA images are JPEG-compressed even in 'original' form\n")
    lines.append("- Simulator may not perfectly replicate real platform compression\n")
    lines.append("- Small platform-specific real-image datasets (25 images each)\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    print(f"[*] Report saved to {report_path}")
    return report_path


def main():
    """Run the full Stage 2 pipeline."""
    t_start = time.time()
    
    print("=" * 80)
    print(" STAGE 2: COMPRESSED-IMAGE TRUFOR FINE-TUNING PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate compressed test sets
    step1_generate_compressed_tests()
    
    # Step 2: Build manifests
    train_samples, val_samples = step2_build_manifests()
    
    # Step 3: Baseline evaluation
    baseline_results = step3_baseline_evaluation()
    
    # Step 4: Per-platform calibration
    thresholds, calibration_results = step4_per_platform_calibration(baseline_results)
    
    # Step 5: Head-only training
    head_ckpt = step5_train_head_only()
    
    # Step 6: Decoder+head training
    decoder_ckpt = step6_train_decoder_head()
    
    # Step 7: Evaluate best models
    head_results = None
    if head_ckpt and Path(str(head_ckpt)).exists():
        head_results = step7_evaluate_finetuned(head_ckpt, "HeadOnly_FineTuned")
    
    decoder_results = None
    if decoder_ckpt and Path(str(decoder_ckpt)).exists():
        decoder_results = step7_evaluate_finetuned(decoder_ckpt, "DecoderHead_FineTuned")
    
    # Use the better model for remaining experiments
    best_results = decoder_results if decoder_results else head_results
    
    # Step 8: LOPO test
    lopo_results = step8_leave_one_platform_out()
    
    # Step 9: Simulator audit
    audit_results = step9_simulator_realism_audit()
    
    # Step 10: Final report
    report_path = generate_final_report(
        baseline_results=baseline_results,
        calibration_results={"thresholds": thresholds, "results": calibration_results} if thresholds else None,
        finetuned_results=best_results,
        head_only_results=head_results,
        lopo_results=lopo_results,
        audit_results=audit_results,
    )
    
    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f" STAGE 2 PIPELINE COMPLETE in {total_elapsed/3600:.1f} hours")
    print(f" Report: {report_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
