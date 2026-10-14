"""
run_v3_baseline_evaluation.py
Re-establish TruFor baseline on Simulator v3 compressed test sets and recalibrate thresholds.
=============================================================================================
1. Evaluates pretrained TruFor model (native resolution) on results/simulated_v3/
   for CocoGlide, Columbia, and CASIA 1.0 across:
   [original, whatsapp, instagram, facebook, telegram] (15 conditions, 50r + 50f each).
2. Saves scores to disk per image in scores_cache.json.
3. Pre-compresses validation set with Simulator v3 and recalibrates per-platform thresholds via Youden's J.
4. Evaluates balanced accuracy at tau=0.50, old tau, and new tau.
5. Produces comparison table against old Stage 2 v1 baseline.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

BASE_DIR = Path("D:/ml_project")
sys.path.insert(0, str(BASE_DIR))

# Ensure simulator_v3 is on path for validation compression
sys.path.insert(0, str(BASE_DIR / "simulator_v3"))
import server as sim_v3

# TruFor model loader
sys.path.insert(0, str(BASE_DIR / "test_docker" / "src"))
from config import update_config, _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

from study.evaluator_stage2 import collect_compressed_test_suite

RESULTS_DIR = BASE_DIR / "study" / "results_v3_baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = RESULTS_DIR / "scores_cache.json"

OLD_BASELINE_FILE = BASE_DIR / "study" / "results_stage2" / "eval_pretrained_baseline.json"
VAL_MANIFEST_PATH = BASE_DIR / "study" / "data_splits_stage2" / "val_manifest.json"
SIMULATED_V3_DIR = BASE_DIR / "results" / "simulated_v3"
VAL_V3_DIR = SIMULATED_V3_DIR / "validation"
VAL_V3_DIR.mkdir(parents=True, exist_ok=True)

PLATFORMS = ["original", "whatsapp", "instagram", "facebook", "telegram"]
DATASETS = ["CocoGlide", "Columbia", "CASIA1.0"]

OLD_THRESHOLDS = {
    "original": 0.16266,
    "whatsapp": 0.37935,
    "instagram": 0.44829,
    "facebook": 0.37935,
    "telegram": 0.36457,
}


def load_trufor_model(num_threads=10):
    torch.set_num_threads(num_threads)
    class DummyArgs: opts = None
    update_config(config, DummyArgs())
    print("[*] Loading TruFor model on CPU...", flush=True)
    t0 = time.time()
    model = confcmx(cfg=config)
    weights_path = BASE_DIR / "pretrained_models" / "trufor.pth.tar"
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[*] TruFor model loaded in {time.time()-t0:.2f}s.", flush=True)
    return model


def native_preprocess(img_path):
    """
    Official native-resolution preprocessing from test_docker/src/data_core.py:
    Reads RGB, converts to float tensor divided by 256.0, NO resizing or downscaling.
    """
    pil_img = Image.open(img_path).convert("RGB")
    arr = np.array(pil_img)
    tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 256.0
    return tensor


def bootstrap_auc_ci(y_true, scores, n_bootstrap=2000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    scores = np.array(scores)
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
    return float(best_tau)


def score_image_with_cache(model, img_path, cache_dict, key):
    if key in cache_dict:
        return cache_dict[key]["score"]
    
    tensor = native_preprocess(img_path)
    with torch.no_grad():
        _, _, det, _ = model(tensor)
        score = float(torch.sigmoid(det).item())
        
    cache_dict[key] = {
        "score": score,
        "path": str(img_path)
    }
    # Save cache periodically or let caller flush
    return score


def save_cache(cache_dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_dict, f, indent=2)


def prepare_validation_v3():
    """
    Find original sources of validation set and compress with v3 for each platform.
    Returns: list of dicts with {platform, label, eval_path, id}
    """
    print("[*] Preparing validation set compressed with Simulator v3...", flush=True)
    with open(VAL_MANIFEST_PATH, "r") as f:
        val_samples = json.load(f)
        
    by_plat_label = defaultdict(list)
    for s in val_samples:
        p = s.get("platform", "original")
        lbl = s["label"]
        by_plat_label[(p, lbl)].append(s)
        
    selected = []
    for (p, lbl), s_list in by_plat_label.items():
        selected.extend(s_list[:50])  # 50 real, 50 fake per platform
        
    val_items = []
    for item in selected:
        plat = item.get("platform", "original")
        lbl = item["label"]
        orig_p = item["image_path"]
        
        # Locate uncompressed source
        actual_orig = None
        if plat == "original":
            actual_orig = orig_p if os.path.exists(orig_p) else None
        elif "IMD2020_" in orig_p:
            base = orig_p.replace(f"IMD2020_{plat}", "IMD2020")
            stem, _ = os.path.splitext(base)
            for ext in [".jpg", ".png", ".jpeg", ".tif"]:
                if os.path.exists(stem + ext):
                    actual_orig = stem + ext
                    break
        elif os.path.exists(orig_p):
            actual_orig = orig_p
            
        if not actual_orig:
            continue
            
        if plat == "original":
            val_items.append({
                "platform": "original",
                "label": lbl,
                "eval_path": actual_orig,
                "id": item["id"]
            })
        else:
            # Compress with v3 if not already compressed
            ext = ".webp" if plat == "instagram" else ".jpg"
            out_p = VAL_V3_DIR / plat / f"{Path(actual_orig).stem}{ext}"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if not out_p.exists():
                try:
                    with Image.open(actual_orig) as im:
                        comp_bytes, _, _, _ = sim_v3.compress_image(im, sim_v3.PLATFORM_PROFILES[plat]["standard"])
                        with open(out_p, "wb") as f_out:
                            f_out.write(comp_bytes)
                except Exception as e:
                    print(f"  [Error compressing val {actual_orig}]: {e}")
                    continue
            val_items.append({
                "platform": plat,
                "label": lbl,
                "eval_path": str(out_p),
                "id": item["id"]
            })
            
    print(f"[*] Prepared {len(val_items)} validation items across platforms.", flush=True)
    return val_items


def main():
    print("=" * 80)
    print("RE-ESTABLISHING TRUFOR BASELINE ON SIMULATOR v3 (NATIVE RESOLUTION)")
    print("=" * 80)
    
    # Load cache if exists
    cache_dict = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache_dict = json.load(f)
            print(f"[*] Loaded {len(cache_dict)} cached scores from disk.", flush=True)
        except Exception:
            cache_dict = {}
            
    # Load model
    model = load_trufor_model(num_threads=10)
    
    # ─── PART 1: TEST SUITE EVALUATION ───
    print("\n[*] Collecting test suite (50 real + 50 fake per condition)...", flush=True)
    suite = collect_compressed_test_suite(max_per_class=50)
    
    test_results = {}
    total_scored_this_run = 0
    
    for ds_name in DATASETS:
        test_results[ds_name] = {}
        print(f"\n--- DATASET: {ds_name} ---")
        for plat in PLATFORMS:
            ext = ".webp" if plat == "instagram" else ".jpg"
            reals = suite[ds_name]["original"]["real"]
            fakes = suite[ds_name]["original"]["fake"]
            
            y_true = []
            scores = []
            
            t0 = time.time()
            # Process 50 reals
            for r_p in reals:
                stem = Path(r_p).stem
                if plat == "original":
                    target_p = r_p
                else:
                    matches = list((SIMULATED_V3_DIR / ds_name / plat).rglob(f"{stem}{ext}"))
                    target_p = str(matches[0]) if matches else None
                    
                if not target_p or not os.path.exists(target_p):
                    print(f"  [MISSING] {ds_name}/{plat} real {stem}")
                    continue
                    
                key = f"test_{ds_name}_{plat}_real_{stem}"
                s = score_image_with_cache(model, target_p, cache_dict, key)
                scores.append(s)
                y_true.append(0)
                total_scored_this_run += 1
                if total_scored_this_run % 20 == 0:
                    save_cache(cache_dict)
                    
            # Process 50 fakes
            for f_p in fakes:
                stem = Path(f_p).stem
                if plat == "original":
                    target_p = f_p
                else:
                    matches = list((SIMULATED_V3_DIR / ds_name / plat).rglob(f"{stem}{ext}"))
                    target_p = str(matches[0]) if matches else None
                    
                if not target_p or not os.path.exists(target_p):
                    print(f"  [MISSING] {ds_name}/{plat} fake {stem}")
                    continue
                    
                key = f"test_{ds_name}_{plat}_fake_{stem}"
                s = score_image_with_cache(model, target_p, cache_dict, key)
                scores.append(s)
                y_true.append(1)
                total_scored_this_run += 1
                if total_scored_this_run % 20 == 0:
                    save_cache(cache_dict)
                    
            save_cache(cache_dict)
            
            auc, lo, hi = bootstrap_auc_ci(y_true, scores)
            dur = time.time() - t0
            
            test_results[ds_name][plat] = {
                "n_real": int(np.sum(np.array(y_true) == 0)),
                "n_fake": int(np.sum(np.array(y_true) == 1)),
                "AUC": float(auc * 100.0),
                "AUC_lo": float(lo * 100.0),
                "AUC_hi": float(hi * 100.0),
                "y_true": y_true,
                "scores": scores,
                "dur_sec": dur
            }
            print(f"  [{plat:10s}] AUC={auc*100:.2f}% [{lo*100:.1f}-{hi*100:.1f}] ({dur:.1f}s)", flush=True)

    # ─── PART 2: THRESHOLD RECALIBRATION ON VALIDATION SET ───
    print("\n" + "=" * 80)
    print("RECALIBRATING PER-PLATFORM THRESHOLDS ON v3 VALIDATION SET")
    print("=" * 80)
    val_items = prepare_validation_v3()
    
    val_scores_by_plat = defaultdict(lambda: {"scores": [], "labels": []})
    for v_item in val_items:
        plat = v_item["platform"]
        lbl = v_item["label"]
        p = v_item["eval_path"]
        key = f"val_{plat}_{lbl}_{Path(p).stem}"
        s = score_image_with_cache(model, p, cache_dict, key)
        val_scores_by_plat[plat]["scores"].append(s)
        val_scores_by_plat[plat]["labels"].append(lbl)
        
    save_cache(cache_dict)
    
    new_thresholds = {}
    for plat in PLATFORMS:
        data = val_scores_by_plat[plat]
        if len(data["scores"]) > 10 and len(set(data["labels"])) == 2:
            tau_star = get_youden_threshold(data["scores"], data["labels"])
            val_auc = roc_auc_score(data["labels"], data["scores"]) * 100.0
            new_thresholds[plat] = tau_star
            print(f"  [{plat:10s}] Old tau={OLD_THRESHOLDS[plat]:.4f} | New tau*={tau_star:.4f} | Val AUC={val_auc:.1f}% (n={len(data['scores'])})")
        else:
            new_thresholds[plat] = 0.50
            print(f"  [{plat:10s}] Insufficient val data, using 0.50")

    # ─── PART 3: BALANCED ACCURACY AT tau=0.5, old tau, new tau ───
    print("\n" + "=" * 80)
    print("BALANCED ACCURACY EVALUATION ON TEST SETS")
    print("=" * 80)
    
    for ds_name in DATASETS:
        for plat in PLATFORMS:
            info = test_results[ds_name][plat]
            y_t = np.array(info["y_true"])
            sc = np.array(info["scores"])
            
            tau_50 = 0.50
            tau_old = OLD_THRESHOLDS[plat]
            tau_new = new_thresholds[plat]
            
            bacc_50 = balanced_accuracy_score(y_t, (sc >= tau_50).astype(int)) * 100.0
            bacc_old = balanced_accuracy_score(y_t, (sc >= tau_old).astype(int)) * 100.0
            bacc_new = balanced_accuracy_score(y_t, (sc >= tau_new).astype(int)) * 100.0
            
            info["bacc_050"] = bacc_50
            info["bacc_old_tau"] = bacc_old
            info["bacc_new_tau"] = bacc_new
            info["tau_old"] = tau_old
            info["tau_new"] = tau_new

    # ─── PART 4: LOAD OLD STAGE 2 BASELINE & COMPARE ───
    with open(OLD_BASELINE_FILE) as f:
        old_baseline = json.load(f)["results"]

    comparison_rows = []
    print("\n" + "=" * 80)
    print("TRUFOR BASELINE COMPARISON: OLD (v1) vs NEW (v3)")
    print("=" * 80)
    print(f"{'Dataset':10s} | {'Platform':10s} | {'Old AUC (95% CI)':20s} | {'New AUC (95% CI)':20s} | {'Delta':7s} | {'CI Width':8s} | {'|Delta| > CI Width?'}")
    print("-" * 88)
    
    for ds_name in DATASETS:
        for plat in PLATFORMS:
            old_info = old_baseline[ds_name][plat]
            new_info = test_results[ds_name][plat]
            
            old_auc = old_info["AUC"]
            old_lo = old_info["AUC_lo"]
            old_hi = old_info["AUC_hi"]
            old_ci_width = old_hi - old_lo
            
            new_auc = new_info["AUC"]
            new_lo = new_info["AUC_lo"]
            new_hi = new_info["AUC_hi"]
            
            delta = new_auc - old_auc
            changed_more = abs(delta) > old_ci_width
            
            old_ci_str = f"{old_auc:5.2f}% [{old_lo:4.1f}-{old_hi:4.1f}]"
            new_ci_str = f"{new_auc:5.2f}% [{new_lo:4.1f}-{new_hi:4.1f}]"
            changed_str = "YES (Significant)" if changed_more else "No (Within CI)"
            
            print(f"{ds_name:10s} | {plat:10s} | {old_ci_str:20s} | {new_ci_str:20s} | {delta:+6.2f}% | {old_ci_width:6.2f}% | {changed_str}")
            
            comparison_rows.append({
                "dataset": ds_name,
                "platform": plat,
                "old_auc": old_auc,
                "old_ci": [old_lo, old_hi],
                "old_ci_width": old_ci_width,
                "new_auc": new_auc,
                "new_ci": [new_lo, new_hi],
                "delta": delta,
                "changed_more_than_ci": changed_more,
                "bacc_050": new_info["bacc_050"],
                "bacc_old_tau": new_info["bacc_old_tau"],
                "bacc_new_tau": new_info["bacc_new_tau"],
                "tau_old": new_info["tau_old"],
                "tau_new": new_info["tau_new"],
            })

    # Save summary JSON
    summary_data = {
        "comparison_rows": comparison_rows,
        "new_thresholds": new_thresholds,
        "old_thresholds": OLD_THRESHOLDS,
        "test_results": {
            ds: {
                p: {k: v for k, v in met.items() if k not in ["y_true", "scores"]}
                for p, met in test_results[ds].items()
            } for ds in test_results
        }
    }
    with open(RESULTS_DIR / "v3_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n[*] Full results saved to study/results_v3_baseline/v3_baseline_results.json")
    return summary_data


if __name__ == "__main__":
    main()
