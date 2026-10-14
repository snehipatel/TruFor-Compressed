"""
run_step6_trufor.py
Step 6: TruFor Downstream Forensics Evaluation
- Paired comparison on 25 originals across 4 platforms (Real posted vs v1 vs v3)
- Real-world threshold check on genuine platform images from DATA
- Incremental JSON caching after every single image evaluation
- Bootstrap 95% confidence intervals
"""

import os
import io
import sys
import json
import time
import math
import random
import statistics
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from PIL import Image, ImageOps

PROJECT_ROOT = Path(r"D:\ml_project")
DATA_DIR = PROJECT_ROOT / "DATA"
V3_DIR = PROJECT_ROOT / "simulator_v3"
RESULTS_DIR = V3_DIR / "audit_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = PROJECT_ROOT / "simulator_audit" / "PROGRESS.md"
CACHE_FILE = RESULTS_DIR / "trufor_scores_cache.json"

# TruFor imports
SRC_DIR = PROJECT_ROOT / "test_docker" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
from config import _C as config, update_config


def log_progress(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## {ts} - v3 Step 6 TruFor: {msg}\n")
    print(f"[{ts}] {msg}", flush=True)


# ─── MODULE LOADERS ───
import importlib.machinery
import importlib.util
import types

_v1_mod = None
_v3_mod = None

def get_v1():
    global _v1_mod
    if _v1_mod is None:
        p = str(PROJECT_ROOT / "SIMULATOR" / "server.py")
        loader = importlib.machinery.SourceFileLoader("sim_v1", p)
        mod = types.ModuleType("sim_v1")
        mod.__file__ = p
        mod.__spec__ = importlib.util.spec_from_file_location("sim_v1", p)
        loader.exec_module(mod)
        _v1_mod = mod
    return _v1_mod

def get_v3():
    global _v3_mod
    if _v3_mod is None:
        p = str(V3_DIR / "server.py")
        loader = importlib.machinery.SourceFileLoader("sim_v3", p)
        mod = types.ModuleType("sim_v3")
        mod.__file__ = p
        mod.__spec__ = importlib.util.spec_from_file_location("sim_v3", p)
        loader.exec_module(mod)
        _v3_mod = mod
    return _v3_mod


# ─── TRUFOR MODEL & INFERENCE ───
class TruForEvaluator:
    def __init__(self, cache_path=CACHE_FILE):
        self.cache_path = cache_path
        self.cache = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"[*] Loaded {len(self.cache)} cached TruFor scores.", flush=True)
            except Exception:
                self.cache = {}

        class DummyArgs:
            opts = None
        update_config(config, DummyArgs())

        print("[*] Loading TruFor model on CPU...", flush=True)
        t0 = time.time()
        self.model = confcmx(cfg=config)
        weights_path = PROJECT_ROOT / "pretrained_models" / "trufor.pth.tar"
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(f"[*] TruFor model loaded in {time.time()-t0:.2f}s.", flush=True)

    def prepare_tensor(self, pil_img, max_size=384):
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if max(w, h) > max_size:
            scale = float(max_size) / float(max(w, h))
            new_w = max(8, (int(round(w * scale)) // 8) * 8)
            new_h = max(8, (int(round(h * scale)) // 8) * 8)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w = max(8, (w // 8) * 8)
            new_h = max(8, (h // 8) * 8)
            if (new_w, new_h) != (w, h):
                pil_img = pil_img.crop((0, 0, new_w, new_h))
        arr = np.array(pil_img).transpose(2, 0, 1) / 256.0
        return torch.tensor(arr, dtype=torch.float).unsqueeze(0)

    def score_image(self, key, pil_img):
        if key in self.cache:
            return self.cache[key]
        t = self.prepare_tensor(pil_img, max_size=384)
        with torch.no_grad():
            _, _, det, _ = self.model(t)
            score = float(torch.sigmoid(det).item())
        self.cache[key] = score
        # Immediately save cache to disk
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)
        return score


# ─── STATISTICAL HELPERS ───
def compute_auc(y_true, y_scores):
    # Standard Wilcoxon-Mann-Whitney AUC
    pos = [s for y, s in zip(y_true, y_scores) if y == 1]
    neg = [s for y, s in zip(y_true, y_scores) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def compute_balanced_acc(y_true, y_scores, tau):
    tp = sum(1 for y, s in zip(y_true, y_scores) if y == 1 and s >= tau)
    fn = sum(1 for y, s in zip(y_true, y_scores) if y == 1 and s < tau)
    tn = sum(1 for y, s in zip(y_true, y_scores) if y == 0 and s < tau)
    fp = sum(1 for y, s in zip(y_true, y_scores) if y == 0 and s >= tau)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (tpr + tnr) / 2.0


def bootstrap_ci(data_tuples, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    random.seed(seed)
    n = len(data_tuples)
    boot_vals = []
    for _ in range(n_boot):
        sample = [random.choice(data_tuples) for _ in range(n)]
        val = metric_fn(sample)
        boot_vals.append(val)
    boot_vals.sort()
    low = boot_vals[int(alpha / 2 * n_boot)]
    high = boot_vals[int((1 - alpha / 2) * n_boot)]
    return round(low, 4), round(high, 4)


def pearson_corr(x, y):
    if len(x) < 2:
        return 0.0
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = statistics.stdev(x)
    sy = statistics.stdev(y)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) - 1)
    return cov / (sx * sy)


# ─── STEP 6 PART (i): PAIRED COMPARISON ───
def run_paired_check(evaluator):
    print("\n" + "=" * 70, flush=True)
    print("STEP 6 (i): PAIRED TRUFOR SCORE COMPARISON (25 ORIGINALS)", flush=True)
    print("=" * 70, flush=True)

    v1 = get_v1()
    v3 = get_v3()

    platforms = ["whatsapp", "facebook", "telegram", "instagram"]
    paired_summary = {}

    for plat in platforms:
        pair_file = RESULTS_DIR / f"step1_pairing_{plat}_real.json"
        with open(pair_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)["pairs"]

        print(f"\n[{plat.upper()}] Scoring {len(pairs)} paired images...", flush=True)
        prof1 = v1.PLATFORM_PROFILES[plat]["standard"]
        prof3 = v3.PLATFORM_PROFILES[plat]["standard"]

        records = []
        for i, p in enumerate(pairs):
            orig_p = p["orig_path"]
            real_p = p["posted_path"]
            stem = Path(orig_p).stem

            # 1. Real posted copy
            real_im = ImageOps.exif_transpose(Image.open(real_p)).convert("RGB")
            score_real = evaluator.score_image(f"step6_real_{plat}_{stem}", real_im)

            # 2. v1 simulator
            orig_im = ImageOps.exif_transpose(Image.open(orig_p)).convert("RGB")
            c1, _, _, _ = v1.compress_image(orig_im, prof1)
            sim1_im = Image.open(io.BytesIO(c1))
            score_v1 = evaluator.score_image(f"step6_v1_{plat}_{stem}", sim1_im)

            # 3. v3 simulator
            c3, _, _, _ = v3.compress_image(orig_im, prof3)
            sim3_im = Image.open(io.BytesIO(c3))
            score_v3 = evaluator.score_image(f"step6_v3_{plat}_{stem}", sim3_im)

            rec = {
                "pair_idx": i + 1,
                "orig": stem,
                "score_real": round(score_real, 4),
                "score_v1": round(score_v1, 4),
                "score_v3": round(score_v3, 4),
                "abs_diff_v1": round(abs(score_v1 - score_real), 4),
                "abs_diff_v3": round(abs(score_v3 - score_real), 4),
            }
            records.append(rec)
            print(f"  [{i+1:2d}/25] {stem:15s}: Real={score_real:.4f} | v1={score_v1:.4f} (diff={rec['abs_diff_v1']:.4f}) | v3={score_v3:.4f} (diff={rec['abs_diff_v3']:.4f})", flush=True)

        scores_real = [r["score_real"] for r in records]
        scores_v1 = [r["score_v1"] for r in records]
        scores_v3 = [r["score_v3"] for r in records]

        mae_v1 = round(statistics.mean([r["abs_diff_v1"] for r in records]), 4)
        mae_v3 = round(statistics.mean([r["abs_diff_v3"] for r in records]), 4)
        corr_v1 = round(pearson_corr(scores_v1, scores_real), 4)
        corr_v3 = round(pearson_corr(scores_v3, scores_real), 4)

        plat_res = {
            "platform": plat,
            "num_pairs": len(records),
            "mae_v1": mae_v1,
            "mae_v3": mae_v3,
            "v3_closer": (mae_v3 <= mae_v1),
            "mae_reduction": round(mae_v1 - mae_v3, 4),
            "corr_v1": corr_v1,
            "corr_v3": corr_v3,
            "mean_score_real": round(statistics.mean(scores_real), 4),
            "mean_score_v1": round(statistics.mean(scores_v1), 4),
            "mean_score_v3": round(statistics.mean(scores_v3), 4),
            "per_image": records,
        }
        paired_summary[plat] = plat_res
        print(f"  [{plat.upper()} SUMMARY] MAE: v1={mae_v1:.4f} -> v3={mae_v3:.4f} (Closer: {plat_res['v3_closer']}) | Corr: v1={corr_v1:.4f} -> v3={corr_v3:.4f}", flush=True)

    out_file = RESULTS_DIR / "step6_trufor_paired_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(paired_summary, f, indent=2)
    print(f"\n[Saved] {out_file.name}", flush=True)
    log_progress("Step 6 (i) Paired TruFor check complete.")
    return paired_summary


# ─── STEP 6 PART (ii): REAL-WORLD THRESHOLD CHECK ───
def run_threshold_check(evaluator):
    print("\n" + "=" * 70, flush=True)
    print("STEP 6 (ii): REAL-WORLD THRESHOLD CHECK ON DATASET", flush=True)
    print("=" * 70, flush=True)

    calibrated_thresholds = {
        "whatsapp": 0.379,
        "instagram": 0.448,
        "facebook": 0.379,
        "telegram": 0.365,
    }

    platforms = ["whatsapp", "facebook", "telegram", "instagram"]
    threshold_summary = {}

    for plat in platforms:
        # Real platform real images: 25 training images
        real_dir = RESULTS_DIR.parent / ".." / "DATA"
        # Find platform real images
        plat_real_file = RESULTS_DIR / f"step1_pairing_{plat}_real.json"
        with open(plat_real_file, "r", encoding="utf-8") as f:
            real_pairs = json.load(f)["pairs"]

        real_posted_paths = [p["posted_path"] for p in real_pairs]

        # Fake platform images: 10 test_fake images
        fake_test_file = RESULTS_DIR / f"step1_pairing_{plat}_test_fake.json"
        with open(fake_test_file, "r", encoding="utf-8") as f:
            fake_pairs = json.load(f)["pairs"]

        fake_posted_paths = [p["posted_path"] for p in fake_pairs]

        print(f"\n[{plat.upper()}] Scoring {len(real_posted_paths)} real + {len(fake_posted_paths)} fake posted images...", flush=True)

        y_true = []
        y_scores = []

        # Real images (y=0)
        for p in real_posted_paths:
            stem = Path(p).stem
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            score = evaluator.score_image(f"step6_real_{plat}_{stem}", im)
            y_true.append(0)
            y_scores.append(score)

        # Fake images (y=1)
        for p in fake_posted_paths:
            stem = Path(p).stem
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            score = evaluator.score_image(f"step6_fake_{plat}_{stem}", im)
            y_true.append(1)
            y_scores.append(score)

        tau_calib = calibrated_thresholds[plat]

        auc = round(compute_auc(y_true, y_scores), 4)
        bacc_default = round(compute_balanced_acc(y_true, y_scores, tau=0.5), 4)
        bacc_calib = round(compute_balanced_acc(y_true, y_scores, tau=tau_calib), 4)

        # Bootstrap CIs
        data_tuples = list(zip(y_true, y_scores))
        auc_ci = bootstrap_ci(data_tuples, lambda d: compute_auc([x[0] for x in d], [x[1] for x in d]))
        bacc_default_ci = bootstrap_ci(data_tuples, lambda d: compute_balanced_acc([x[0] for x in d], [x[1] for x in d], tau=0.5))
        bacc_calib_ci = bootstrap_ci(data_tuples, lambda d: compute_balanced_acc([x[0] for x in d], [x[1] for x in d], tau=tau_calib))

        plat_stats = {
            "platform": plat,
            "sample_size": {
                "n_real": len(real_posted_paths),
                "n_fake": len(fake_posted_paths),
                "n_total": len(y_true),
            },
            "calibrated_threshold": tau_calib,
            "auc": auc,
            "auc_95_ci": list(auc_ci),
            "balanced_acc_tau_0_5": bacc_default,
            "balanced_acc_tau_0_5_95_ci": list(bacc_default_ci),
            "balanced_acc_calibrated": bacc_calib,
            "balanced_acc_calibrated_95_ci": list(bacc_calib_ci),
            "real_scores_mean": round(statistics.mean([s for y, s in zip(y_true, y_scores) if y == 0]), 4),
            "fake_scores_mean": round(statistics.mean([s for y, s in zip(y_true, y_scores) if y == 1]), 4),
        }
        threshold_summary[plat] = plat_stats
        print(f"  [{plat.upper()}] N_real={len(real_posted_paths)}, N_fake={len(fake_posted_paths)} | AUC={auc} (95% CI: {auc_ci})", flush=True)
        print(f"    BAcc (tau=0.5):        {bacc_default} (95% CI: {bacc_default_ci})", flush=True)
        print(f"    BAcc (tau={tau_calib}): {bacc_calib} (95% CI: {bacc_calib_ci})", flush=True)

    out_file = RESULTS_DIR / "step6_trufor_threshold_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(threshold_summary, f, indent=2)
    print(f"\n[Saved] {out_file.name}", flush=True)
    log_progress("Step 6 (ii) Real-world threshold check complete.")
    return threshold_summary


def main():
    print("\n" + "=" * 70, flush=True)
    print("STARTING STEP 6: TRUFOR DOWNSTREAM FORENSICS CHECK", flush=True)
    print("=" * 70, flush=True)
    t_start = time.time()

    evaluator = TruForEvaluator()

    # Part (i): Paired check
    paired_results = run_paired_check(evaluator)

    # Part (ii): Threshold check
    threshold_results = run_threshold_check(evaluator)

    print("\n" + "=" * 70, flush=True)
    print(f"STEP 6 COMPLETED in {time.time() - t_start:.1f}s", flush=True)
    print("=" * 70, flush=True)
    log_progress("TruFor Step 6 downstream evaluation finished successfully.")


if __name__ == "__main__":
    main()
