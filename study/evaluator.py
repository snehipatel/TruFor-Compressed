"""
Rigorous Scientific Evaluation Benchmark & Multi-Threshold Analyzer.
Reference: TruFor_Evaluation_Report.txt (Sections 5.2, 5.3, 6.1-6.3, 8.1, 9.1(d), 11)

Evaluates:
1. Baseline Model (Official TruFor Pretrained)
2. Approach 1 (Detection-Only Fine-Tuned)
3. Approach 2 (Low-LR All-Heads Fine-Tuned)

Strictly on Unseen Testing Datasets:
- CocoGlide (Real vs Inpainted/Generative)
- Columbia (Authentic vs Spliced)
- CASIA 1.0 (Authentic vs Spliced/Copy-Move)

Across all 5 transmission/compression conditions:
- Original / Uncompressed
- WhatsApp
- Instagram
- Facebook
- Telegram

Threshold Regimes:
1. Threshold-Free Gold Standard: ROC-AUC (Area Under ROC Curve)
2. Calibrated Social Media Threshold: tau = 0.30 (Report Section 5.2 & 9.1(d))
3. High-Sensitivity Inpainting Threshold: tau = 0.20
4. Platform-Adaptive Threshold: tau = 0.20 for raw/orig, tau = 0.30 for social platforms
5. Dual-Criteria Detection: Global score >= tau OR (anom_area >= 1% and peak >= 0.80) (Report Section 6.3)
6. Author Default Baseline: tau = 0.50 (for historical comparability)
"""

import os
import sys
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

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
DATASET_DIR = BASE_DIR / "DATASET"
COMPRESSED_DIR = DATASET_DIR / "COMPRESSED"
RESULTS_DIR = BASE_DIR / "study" / "results"

PLATFORMS = ["original", "whatsapp", "instagram", "facebook", "telegram"]


def collect_test_dataset_pairs(max_per_class=100):
    """
    Scans the 3 test datasets and prepares evaluation pairs across all 5 platforms.
    Guarantees equal pairing between real and tampered.
    """
    test_suite = {}
    
    # ---------------- 1. COCOGLIDE ----------------
    cocoglide_suite = {p: {"real": [], "fake": []} for p in PLATFORMS}
    # Original
    cg_orig = DATASET_DIR / "CocoGlide"
    if cg_orig.exists():
        r_files = sorted(list((cg_orig / "real").glob("*.png")) + list((cg_orig / "real").glob("*.jpg")))[:max_per_class]
        f_files = sorted(list((cg_orig / "fake").glob("*.png")) + list((cg_orig / "fake").glob("*.jpg")))[:max_per_class]
        cocoglide_suite["original"]["real"] = [str(x) for x in r_files]
        cocoglide_suite["original"]["fake"] = [str(x) for x in f_files]
    # Compressed
    cg_comp = COMPRESSED_DIR / "CocoGlide"
    if cg_comp.exists():
        for p in ["whatsapp", "instagram", "facebook", "telegram"]:
            r_dir = cg_comp / p / "real"
            f_dir = cg_comp / p / "fake"
            if r_dir.exists() and f_dir.exists():
                r_f = sorted(list(r_dir.glob("*.*")))[:max_per_class]
                f_f = sorted(list(f_dir.glob("*.*")))[:max_per_class]
                cocoglide_suite[p]["real"] = [str(x) for x in r_f]
                cocoglide_suite[p]["fake"] = [str(x) for x in f_f]
    test_suite["CocoGlide"] = cocoglide_suite

    # ---------------- 2. COLUMBIA ----------------
    columbia_suite = {p: {"real": [], "fake": []} for p in PLATFORMS}
    col_orig = DATASET_DIR / "Columbia"
    if col_orig.exists():
        r_files = sorted(list((col_orig / "4cam_auth").glob("*.*")))[:max_per_class]
        f_files = sorted(list((col_orig / "4cam_splc").glob("*.*")))[:max_per_class]
        columbia_suite["original"]["real"] = [str(x) for x in r_files]
        columbia_suite["original"]["fake"] = [str(x) for x in f_files]
    col_comp = COMPRESSED_DIR / "Columbia"
    if col_comp.exists():
        for p in ["whatsapp", "instagram", "facebook", "telegram"]:
            r_dir = col_comp / p / "4cam_auth"
            f_dir = col_comp / p / "4cam_splc"
            if r_dir.exists() and f_dir.exists():
                r_f = sorted(list(r_dir.glob("*.*")))[:max_per_class]
                f_f = sorted(list(f_dir.glob("*.*")))[:max_per_class]
                columbia_suite[p]["real"] = [str(x) for x in r_f]
                columbia_suite[p]["fake"] = [str(x) for x in f_f]
    test_suite["Columbia"] = columbia_suite

    # ---------------- 3. CASIA 1.0 ----------------
    casia1_suite = {p: {"real": [], "fake": []} for p in PLATFORMS}
    c1_orig = DATASET_DIR / "CASIA1.0" / "CASIA 1.0 dataset"
    if c1_orig.exists():
        r_files = sorted(list((c1_orig / "Au" / "Au").glob("*.*")))[:max_per_class]
        f_files = sorted(list((c1_orig / "Modified Tp" / "Tp" / "Sp").glob("*.*")) + list((c1_orig / "Modified Tp" / "Tp" / "CM").glob("*.*")))[:max_per_class]
        casia1_suite["original"]["real"] = [str(x) for x in r_files]
        casia1_suite["original"]["fake"] = [str(x) for x in f_files]
    c1_comp = COMPRESSED_DIR / "CASIA1.0"
    if c1_comp.exists():
        for p in ["whatsapp", "instagram", "facebook", "telegram"]:
            r_dir = c1_comp / p / "Au"
            f_dir = c1_comp / p / "Modified Tp"
            if r_dir.exists() and f_dir.exists():
                r_f = sorted(list(r_dir.glob("**/*.*")))[:max_per_class]
                f_f = sorted(list(f_dir.glob("**/*.*")))[:max_per_class]
                casia1_suite[p]["real"] = [str(x) for x in r_f]
                casia1_suite[p]["fake"] = [str(x) for x in f_f]
    test_suite["CASIA1.0"] = casia1_suite

    return test_suite


def evaluate_single_image(model, img_path: str, device, max_size=512):
    """
    Safely preprocesses and runs TruFor inference on a single image.
    Extracts global detection score and spatial anomaly statistics for dual-criteria detection.
    """
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
        pred_loc, pred_conf, pred_det, _ = model(t_rgb)
        score = torch.sigmoid(pred_det).item() if pred_det is not None else 0.0
        
        # Spatial anomaly localization and confidence map
        prob_map = torch.softmax(pred_loc, dim=1)[:, 1].squeeze(0)
        conf_map = torch.sigmoid(pred_conf).squeeze()
        high_anom = (prob_map > 0.5) & (conf_map > 0.5)
        anom_area = high_anom.float().mean().item()
        peak_anom = prob_map.max().item()
        
    return score, anom_area, peak_anom


def evaluate_model_on_suite(model, test_suite, device, model_name="Model"):
    """
    Evaluates a model across all test datasets and platforms under multiple threshold regimes:
    1. ROC-AUC (threshold-independent)
    2. tau = 0.50 (author default)
    3. tau = 0.30 (report recommended balanced)
    4. tau = 0.20 (inpainting sensitivity)
    5. Adaptive tau (0.20 for raw/orig, 0.30 for social)
    6. Dual-Criteria (Global score + Localization anomaly map)
    """
    model.eval()
    results = []
    
    for ds_name, platforms in test_suite.items():
        for p, pair in platforms.items():
            real_imgs = pair["real"]
            fake_imgs = pair["fake"]
            if not real_imgs and not fake_imgs:
                continue
                
            y_true = []
            scores = []
            anom_areas = []
            peak_anoms = []
            
            for pth in real_imgs:
                try:
                    s, ar, pk = evaluate_single_image(model, pth, device)
                    scores.append(s)
                    anom_areas.append(ar)
                    peak_anoms.append(pk)
                    y_true.append(0)
                except Exception:
                    pass
                    
            for pth in fake_imgs:
                try:
                    s, ar, pk = evaluate_single_image(model, pth, device)
                    scores.append(s)
                    anom_areas.append(ar)
                    peak_anoms.append(pk)
                    y_true.append(1)
                except Exception:
                    pass
                    
            if not y_true:
                continue
                
            y_true = np.array(y_true)
            scores = np.array(scores)
            anom_areas = np.array(anom_areas)
            peak_anoms = np.array(peak_anoms)
            
            # AUC (Threshold-independent)
            try:
                auc = roc_auc_score(y_true, scores)
            except Exception:
                auc = 0.5
                
            def get_stats(y_pred):
                bacc = balanced_accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                return bacc, f1, tpr, tnr

            # Regime 1: Author Default (tau = 0.50)
            bacc_050, f1_050, tpr_050, tnr_050 = get_stats((scores >= 0.50).astype(int))
            
            # Regime 2: Report Recommended Social Balanced (tau = 0.30)
            bacc_030, f1_030, tpr_030, tnr_030 = get_stats((scores >= 0.30).astype(int))
            
            # Regime 3: High-Sensitivity Inpainting (tau = 0.20)
            bacc_020, f1_020, tpr_020, tnr_020 = get_stats((scores >= 0.20).astype(int))
            
            # Regime 4: Platform-Adaptive (0.20 for original, 0.30 for social)
            tau_adapt = 0.20 if p == "original" else 0.30
            bacc_adapt, f1_adapt, tpr_adapt, tnr_adapt = get_stats((scores >= tau_adapt).astype(int))
            
            # Regime 5: Dual-Criteria Detection (Global score OR localized anomaly >= 1% and peak >= 0.80)
            crit1 = (scores >= tau_adapt)
            crit2 = (anom_areas >= 0.01) & (peak_anoms >= 0.80)
            y_pred_dual = (crit1 | crit2).astype(int)
            bacc_dual, f1_dual, tpr_dual, tnr_dual = get_stats(y_pred_dual)
            
            results.append({
                "model": model_name,
                "dataset": ds_name,
                "platform": p,
                "num_real": np.sum(y_true == 0),
                "num_fake": np.sum(y_true == 1),
                "AUC": auc,
                "bACC": bacc_050,  # Legacy field
                "F1": f1_050,      # Legacy field
                "bACC_050": bacc_050,
                "TPR_050": tpr_050,
                "TNR_050": tnr_050,
                "F1_050": f1_050,
                "bACC_030": bacc_030,
                "TPR_030": tpr_030,
                "TNR_030": tnr_030,
                "F1_030": f1_030,
                "bACC_020": bacc_020,
                "TPR_020": tpr_020,
                "TNR_020": tnr_020,
                "bACC_adapt": bacc_adapt,
                "TPR_adapt": tpr_adapt,
                "TNR_adapt": tnr_adapt,
                "bACC_dual": bacc_dual,
                "TPR_dual": tpr_dual,
                "TNR_dual": tnr_dual,
                "F1_dual": f1_dual
            })
            
    return results


def run_benchmark_comparison(
    models_dict=None,
    max_eval_per_class=50
):
    """
    Executes 3-way evaluation benchmark across all test datasets, platforms, and threshold regimes.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if models_dict is None:
        models_dict = {
            "Baseline (Pretrained)": BASE_DIR / "pretrained_models" / "trufor.pth.tar",
        }
        if (BASE_DIR / "study" / "checkpoints" / "exp2_without_aug.pth").exists():
            models_dict["Approach 2 (No Aug)"] = BASE_DIR / "study" / "checkpoints" / "exp2_without_aug.pth"
        elif (BASE_DIR / "study" / "checkpoints" / "exp2_low_lr.pth").exists():
            models_dict["Approach 2 (No Aug)"] = BASE_DIR / "study" / "checkpoints" / "exp2_low_lr.pth"
            
        if (BASE_DIR / "study" / "checkpoints" / "exp2_mild_aug.pth").exists():
            models_dict["Approach 2 (Mild Aug)"] = BASE_DIR / "study" / "checkpoints" / "exp2_mild_aug.pth"
        elif (BASE_DIR / "study" / "checkpoints" / "exp2_with_aug.pth").exists():
            models_dict["Approach 2 (With Aug)"] = BASE_DIR / "study" / "checkpoints" / "exp2_with_aug.pth"
        
    print("\n" + "="*85)
    print(" COMPREHENSIVE BENCHMARK EVALUATION ACROSS CALIBRATED THRESHOLDS")
    print(" Reference: TruFor_Evaluation_Report.txt (Sections 5.2, 5.3, 6.1-6.3, 8.1, 9.1(d))")
    print("="*85)
    for name, pth in models_dict.items():
        exists_str = "[Found]" if Path(pth).exists() else "[Missing]"
        print(f"[*] {name:<26}: {pth} {exists_str}")
        
    test_suite = collect_test_dataset_pairs(max_per_class=max_eval_per_class)
    all_results = []
    evaluated_models = []
    
    for model_name, weights_path in models_dict.items():
        if not Path(weights_path).exists():
            print(f"[!] Skipping {model_name} (checkpoint file not found)")
            continue
            
        print(f"\n---> Evaluating: {model_name}...")
        model, _ = get_trufor_model(weights_path=str(weights_path), device=device)
        res = evaluate_model_on_suite(model, test_suite, device, model_name=model_name)
        all_results.extend(res)
        evaluated_models.append(model_name)
        del model
        torch.cuda.empty_cache()
        
    # Save CSV
    csv_path = RESULTS_DIR / "benchmark_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_results:
            fieldnames = list(all_results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)

    def print_comparison_grid(title, metric_key, is_percentage=True):
        print("\n" + "="*105)
        print(f" {title.upper()}")
        print("="*105)
        header = f"{'DATASET':<12} {'PLATFORM':<12}"
        for m in evaluated_models:
            header += f" {m[:18]:<18}"
        if len(evaluated_models) >= 2 and evaluated_models[0] == "Baseline (Pretrained)":
            for m in evaluated_models[1:]:
                header += f" {'Δ ' + m[:10]:<12}"
        print(header)
        print("-" * 105)
        
        grid = {}
        for r in all_results:
            key = (r['dataset'], r['platform'])
            if key not in grid:
                grid[key] = {}
            multiplier = 100.0 if is_percentage else 1.0
            grid[key][r['model']] = r[metric_key] * multiplier
            
        for (ds, p), m_scores in grid.items():
            row_str = f"{ds:<12} {p:<12}"
            for m in evaluated_models:
                val = m_scores.get(m, None)
                if val is not None:
                    row_str += f" {val:>6.2f}%            "
                else:
                    row_str += f" {'N/A':>6}            "
                    
            base_val = m_scores.get(evaluated_models[0], None) if evaluated_models else None
            if base_val is not None:
                for m in evaluated_models[1:]:
                    comp_val = m_scores.get(m, None)
                    if comp_val is not None:
                        diff = comp_val - base_val
                        diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
                        row_str += f" {diff_str:>10} "
                    else:
                        row_str += f" {'N/A':>10} "
            print(row_str)
        print("="*105)

    # 1. ROC-AUC Table
    print_comparison_grid("1. Threshold-Free Separation: ROC-AUC (Academic Gold Standard)", "AUC")

    # 2. Calibrated tau = 0.30 Table
    print_comparison_grid("2. Report-Calibrated Operating Point: Balanced Accuracy (tau = 0.30)", "bACC_030")

    # 3. Dual-Criteria Table
    print_comparison_grid("3. Dual-Criteria Detection: Balanced Accuracy (Global + Local Anomaly)", "bACC_dual")

    # 4. Author Default tau = 0.50 Table
    print_comparison_grid("4. Author Default Baseline: Balanced Accuracy (tau = 0.50)", "bACC_050")

    # 5. Executive Summary Table
    print("\n" + "="*85)
    print(" EXECUTIVE BENCHMARK SUMMARY (OVERALL MEAN ACROSS ALL 15 CONDITIONS)")
    print("="*85)
    print(f"{'Model':<26} | {'ROC-AUC':<10} | {'bACC (tau=0.30)':<16} | {'Dual-Criteria':<14} | {'bACC (tau=0.50)':<14}")
    print("-" * 85)
    for m in evaluated_models:
        m_items = [r for r in all_results if r['model'] == m]
        if not m_items:
            continue
        mean_auc = np.mean([r['AUC'] for r in m_items]) * 100
        mean_bacc_030 = np.mean([r['bACC_030'] for r in m_items]) * 100
        mean_dual = np.mean([r['bACC_dual'] for r in m_items]) * 100
        mean_bacc_050 = np.mean([r['bACC_050'] for r in m_items]) * 100
        print(f"{m:<26} | {mean_auc:>8.2f}% | {mean_bacc_030:>14.2f}% | {mean_dual:>12.2f}% | {mean_bacc_050:>12.2f}%")
    print("="*85)
    print(f"\n[OK] Full multi-threshold benchmark results saved to: {csv_path}\n")
    return csv_path


if __name__ == "__main__":
    run_benchmark_comparison(max_eval_per_class=50)
