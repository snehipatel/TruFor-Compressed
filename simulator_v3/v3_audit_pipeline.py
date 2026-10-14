"""
v3 Audit Pipeline — Phase 2 (Fix & Complete)
=============================================
Fixes from Phase 1 run:
  1. v1 import: set __file__ before exec_module
  2. Instagram pairing: center-crop originals to 1:1 before hashing
  3. Display: handle None values gracefully
  4. Instagram BPP: also check if sim needs center-crop to match real IG behavior

Re-uses cached step1 results for WA/FB/TG. Redoes IG pairing + all step2.
"""

import os
import io
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps

# ─── CONFIGURATION ───
PROJECT_ROOT = Path(r"D:\ml_project")
DATA_DIR = PROJECT_ROOT / "DATA"
V3_DIR = PROJECT_ROOT / "simulator_v3"
RESULTS_DIR = V3_DIR / "audit_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = PROJECT_ROOT / "simulator_audit" / "PROGRESS.md"

PLATFORM_DIRS = {
    "whatsapp": {
        "real_posted": DATA_DIR / "WHATSAPP" / "whatsapp_real",
        "fake_posted": DATA_DIR / "WHATSAPP" / "whatsapp_fake",
        "test_real_posted": DATA_DIR / "WHATSAPP" / "whatsapp_real_test",
        "test_fake_posted": DATA_DIR / "WHATSAPP" / "whatsapp_fake_test",
    },
    "facebook": {
        "real_posted": DATA_DIR / "FACEBOOK" / "facebook_real",
        "fake_posted": DATA_DIR / "FACEBOOK" / "facebook_fake",
        "test_real_posted": DATA_DIR / "FACEBOOK" / "facebook_test_real",
        "test_fake_posted": DATA_DIR / "FACEBOOK" / "facebook_test_fake",
    },
    "telegram": {
        "real_posted": DATA_DIR / "TELEGRAM" / "telegram_real",
        "fake_posted": DATA_DIR / "TELEGRAM" / "telegram_fake",
        "test_real_posted": DATA_DIR / "TELEGRAM" / "telegram_real_test",
        "test_fake_posted": DATA_DIR / "TELEGRAM" / "telegram_fake_test",
    },
    "instagram": {
        "real_posted": DATA_DIR / "INSTAGRAM" / "Instagram_Real",
        "fake_posted": DATA_DIR / "INSTAGRAM" / "Instagram_Fake",
        "test_real_posted": DATA_DIR / "INSTAGRAM" / "Instagram_real_test",
        "test_fake_posted": DATA_DIR / "INSTAGRAM" / "Instagram_fake_test",
    },
}

ORIGINALS = {
    "real": DATA_DIR / "REAL",
    "fake": DATA_DIR / "FAKE",
    "test_real": DATA_DIR / "REAL" / "real_test",
    "test_fake": DATA_DIR / "FAKE" / "fake_test",
}


def log_progress(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## {ts} - v3: {msg}\n")
    print(f"[{ts}] {msg}")


def save_result(step_name, platform, data):
    outpath = RESULTS_DIR / f"{step_name}_{platform}.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  -> Saved {outpath.name}")
    return outpath


def load_result(step_name, platform):
    outpath = RESULTS_DIR / f"{step_name}_{platform}.json"
    if outpath.exists():
        with open(outpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_images(directory):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if not directory.exists():
        return []
    return sorted([
        directory / f for f in os.listdir(directory)
        if f.lower().endswith(tuple(exts)) and not f.startswith(".")
    ])


# ─── v1 SIMULATOR LOADING (fixed __file__) ───
_v1_module = None

def get_v1_module():
    global _v1_module
    if _v1_module is None:
        import importlib, importlib.machinery, types
        v1_path = str(PROJECT_ROOT / "SIMULATOR" / "server.py")
        loader = importlib.machinery.SourceFileLoader("sim_v1", v1_path)
        mod = types.ModuleType("sim_v1")
        mod.__file__ = v1_path  # Fix: v1 server.py uses __file__
        mod.__loader__ = loader
        mod.__spec__ = importlib.util.spec_from_file_location("sim_v1", v1_path)
        loader.exec_module(mod)
        _v1_module = mod
    return _v1_module


# ─── v3 SIMULATOR LOADING ───
_v3_module = None

def get_v3_module():
    global _v3_module
    if _v3_module is None:
        import importlib, importlib.machinery, types
        v3_path = str(V3_DIR / "server.py")
        loader = importlib.machinery.SourceFileLoader("sim_v3", v3_path)
        mod = types.ModuleType("sim_v3")
        mod.__file__ = v3_path
        mod.__loader__ = loader
        mod.__spec__ = importlib.util.spec_from_file_location("sim_v3", v3_path)
        loader.exec_module(mod)
        _v3_module = mod
    return _v3_module


def reload_v3_module():
    """Force reload v3 after modifying server.py"""
    global _v3_module
    _v3_module = None
    return get_v3_module()


# ─── PERCEPTUAL HASH + SSIM ───
def compute_dhash(img, hash_size=16):
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(small)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def hamming_distance(h1, h2):
    return int(np.sum(h1 != h2))


def compute_ssim_simple(img1, img2, target_size=(256, 256)):
    a = np.array(img1.convert("L").resize(target_size, Image.LANCZOS), dtype=np.float64)
    b = np.array(img2.convert("L").resize(target_size, Image.LANCZOS), dtype=np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.std(), b.std()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)) / \
           ((mu_a**2 + mu_b**2 + C1) * (sig_a**2 + sig_b**2 + C2))
    return float(ssim)


def center_crop_square(img):
    """Center-crop to square (Instagram-style)"""
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


# ─── STEP 1: PAIRING ───
def step1_pair_images(platform, orig_type, posted_dir, force=False):
    step_name = f"step1_pairing_{orig_type}"
    if not force:
        existing = load_result(step_name, platform)
        if existing is not None:
            n_high = existing.get("high_confidence_count", 0)
            n_pairs = existing.get("num_pairs", 0)
            if n_high > 0:  # Only use cache if we got good results
                print(f"  Step 1 ({platform}/{orig_type}): cached ({n_high}/{n_pairs} HIGH)")
                return existing

    originals = list_images(ORIGINALS[orig_type])
    posted = list_images(posted_dir)

    if not originals or not posted:
        result = {"pairs": [], "num_pairs": 0, "high_confidence_count": 0,
                  "medium_confidence_count": 0, "low_confidence_count": 0,
                  "error": f"No images: {len(originals)} orig, {len(posted)} posted"}
        save_result(step_name, platform, result)
        return result

    print(f"  Pairing {len(originals)} originals -> {len(posted)} posted ({platform}/{orig_type})...")

    # Check if posted images are square (Instagram center-crops)
    sample_posted = Image.open(posted[0])
    sample_posted = ImageOps.exif_transpose(sample_posted)
    posted_is_square = abs(sample_posted.size[0] - sample_posted.size[1]) < 5

    orig_data = []
    for p in originals:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)
        # If posted images are square, center-crop originals before hashing
        if posted_is_square and abs(img.size[0] - img.size[1]) > 50:
            img_for_hash = center_crop_square(img)
        else:
            img_for_hash = img
        dhash = compute_dhash(img_for_hash)
        orig_data.append({"path": str(p), "dhash": dhash, "img_for_ssim": img_for_hash})

    posted_data = []
    for p in posted:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)
        dhash = compute_dhash(img)
        posted_data.append({"path": str(p), "dhash": dhash, "img_for_ssim": img})

    n_orig = len(orig_data)
    n_post = len(posted_data)

    # Build distance matrix
    dist_matrix = np.zeros((n_orig, n_post), dtype=int)
    for i, od in enumerate(orig_data):
        for j, pd in enumerate(posted_data):
            dist_matrix[i, j] = hamming_distance(od["dhash"], pd["dhash"])

    # Greedy best-match
    pairs = []
    remaining_orig = set(range(n_orig))
    remaining_post = set(range(n_post))

    while remaining_orig and remaining_post:
        best_dist = 999
        best_i, best_j = -1, -1
        for i in remaining_orig:
            for j in remaining_post:
                if dist_matrix[i, j] < best_dist:
                    best_dist = dist_matrix[i, j]
                    best_i, best_j = i, j
        if best_i < 0:
            break

        # SSIM verification
        ssim = compute_ssim_simple(orig_data[best_i]["img_for_ssim"],
                                    posted_data[best_j]["img_for_ssim"])

        confidence = "HIGH" if best_dist < 30 and ssim > 0.8 else \
                     "MEDIUM" if best_dist < 60 and ssim > 0.6 else "LOW"

        pairs.append({
            "orig_path": str(orig_data[best_i]["path"]),
            "posted_path": str(posted_data[best_j]["path"]),
            "dhash_distance": best_dist,
            "ssim": round(ssim, 4),
            "confidence": confidence,
        })

        remaining_orig.discard(best_i)
        remaining_post.discard(best_j)

    result = {
        "platform": platform, "orig_type": orig_type,
        "num_originals": n_orig, "num_posted": n_post, "num_pairs": len(pairs),
        "pairs": pairs,
        "posted_is_square": posted_is_square,
        "high_confidence_count": sum(1 for p in pairs if p["confidence"] == "HIGH"),
        "medium_confidence_count": sum(1 for p in pairs if p["confidence"] == "MEDIUM"),
        "low_confidence_count": sum(1 for p in pairs if p["confidence"] == "LOW"),
    }
    save_result(step_name, platform, result)
    return result


# ─── STEP 2: PAIRED BPP ───
def get_image_properties(filepath):
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    file_size = os.path.getsize(filepath)
    bpp = (file_size * 8) / (w * h)
    props = {
        "path": str(filepath), "filename": os.path.basename(filepath),
        "width": w, "height": h, "max_dim": max(w, h),
        "file_size": file_size, "bpp": round(bpp, 6), "format": img.format,
    }
    if img.format == "JPEG":
        props["progressive"] = bool(img.info.get("progressive", False))
        qtables = img.quantization
        if qtables:
            props["qtable_0"] = list(qtables.get(0, []))
            props["qtable_1"] = list(qtables.get(1, []))
    return props


def simulate_v1(orig_path, platform):
    try:
        v1 = get_v1_module()
        img = Image.open(orig_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        profile = v1.PLATFORM_PROFILES[platform]["standard"]
        compressed_bytes, new_w, new_h, codec = v1.compress_image(img, profile)
        bpp = (len(compressed_bytes) * 8) / (new_w * new_h)
        return {"bpp": round(bpp, 6), "file_size": len(compressed_bytes),
                "width": new_w, "height": new_h, "codec": codec}
    except Exception as e:
        return {"bpp": None, "error": str(e)}


def simulate_v3(orig_path, platform):
    try:
        v3 = get_v3_module()
        img = Image.open(orig_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        profile = v3.PLATFORM_PROFILES[platform]["standard"]
        compressed_bytes, new_w, new_h, codec = v3.compress_image(img, profile)
        bpp = (len(compressed_bytes) * 8) / (new_w * new_h)
        return {"bpp": round(bpp, 6), "file_size": len(compressed_bytes),
                "width": new_w, "height": new_h, "codec": codec}
    except Exception as e:
        return {"bpp": None, "error": str(e)}


def step2_paired_bpp(platform, pairing_result, orig_type="real", force=False):
    step_name = f"step2_paired_bpp_{orig_type}"
    if not force:
        existing = load_result(step_name, platform)
        if existing is not None and len(existing.get("per_image", [])) > 0:
            print(f"  Step 2 ({platform}/{orig_type}): cached ({len(existing['per_image'])} pairs)")
            return existing

    pairs = pairing_result.get("pairs", [])
    # Use HIGH+MEDIUM pairs; if none, use all (with warning)
    good_pairs = [p for p in pairs if p["confidence"] in ("HIGH", "MEDIUM")]
    if not good_pairs and pairs:
        print(f"  WARNING: No HIGH/MEDIUM pairs for {platform}/{orig_type}, using LOW")
        good_pairs = pairs  # Use LOW as fallback
    if not good_pairs:
        result = {"platform": platform, "orig_type": orig_type,
                  "summary": {"num_pairs": 0}, "per_image": []}
        save_result(step_name, platform, result)
        return result

    print(f"  Measuring {len(good_pairs)} paired images ({platform}/{orig_type})...")

    per_image = []
    for p in good_pairs:
        orig_path = p["orig_path"]
        posted_path = p["posted_path"]
        real_props = get_image_properties(posted_path)
        v1_sim = simulate_v1(orig_path, platform)
        v3_sim = simulate_v3(orig_path, platform)

        record = {
            "orig_file": os.path.basename(orig_path),
            "posted_file": os.path.basename(posted_path),
            "pairing_confidence": p["confidence"],
            "pairing_ssim": p["ssim"],
            "real_bpp": real_props["bpp"],
            "real_file_size": real_props["file_size"],
            "real_dims": f"{real_props['width']}x{real_props['height']}",
            "real_max_dim": real_props["max_dim"],
        }

        if v1_sim.get("bpp") is not None:
            record["v1_bpp"] = v1_sim["bpp"]
            record["v1_ratio"] = round(v1_sim["bpp"] / real_props["bpp"], 4) if real_props["bpp"] > 0 else None
            record["v1_dims"] = f"{v1_sim['width']}x{v1_sim['height']}"
        else:
            record["v1_error"] = v1_sim.get("error", "unknown")

        if v3_sim.get("bpp") is not None:
            record["v3_bpp"] = v3_sim["bpp"]
            record["v3_ratio"] = round(v3_sim["bpp"] / real_props["bpp"], 4) if real_props["bpp"] > 0 else None
            record["v3_dims"] = f"{v3_sim['width']}x{v3_sim['height']}"
        else:
            record["v3_error"] = v3_sim.get("error", "unknown")

        per_image.append(record)
        v1_s = f"v1={v1_sim.get('bpp', 'ERR')}"
        v3_s = f"v3={v3_sim.get('bpp', 'ERR')}"
        print(f"    {record['orig_file'][:30]:30s}: real={real_props['bpp']:.4f}, {v1_s}, {v3_s}")

    # Summary stats
    v1_ratios = [r["v1_ratio"] for r in per_image if r.get("v1_ratio") is not None]
    v3_ratios = [r["v3_ratio"] for r in per_image if r.get("v3_ratio") is not None]
    real_bpps = [r["real_bpp"] for r in per_image]

    summary = {"num_pairs": len(per_image)}
    if real_bpps:
        summary["real_bpp_median"] = round(statistics.median(real_bpps), 6)
        summary["real_bpp_mean"] = round(statistics.mean(real_bpps), 6)
    if v1_ratios:
        summary["v1_ratio_median"] = round(statistics.median(v1_ratios), 4)
        summary["v1_ratio_mean"] = round(statistics.mean(v1_ratios), 4)
        summary["v1_ratio_std"] = round(statistics.stdev(v1_ratios), 4) if len(v1_ratios) > 1 else 0
    if v3_ratios:
        summary["v3_ratio_median"] = round(statistics.median(v3_ratios), 4)
        summary["v3_ratio_mean"] = round(statistics.mean(v3_ratios), 4)
        summary["v3_ratio_std"] = round(statistics.stdev(v3_ratios), 4) if len(v3_ratios) > 1 else 0

    result = {"platform": platform, "orig_type": orig_type,
              "summary": summary, "per_image": per_image}
    save_result(step_name, platform, result)
    return result


# ─── STEP 3: INSTAGRAM CALIBRATION ───
def calibrate_instagram_quality(pairing_result):
    step_name = "step3_ig_calibration"
    existing = load_result(step_name, "instagram")
    if existing is not None and "best_quality" in existing:
        print(f"  Step 3 (Instagram calibration): cached, best_q={existing['best_quality']}")
        return existing

    pairs = [p for p in pairing_result.get("pairs", [])
             if p["confidence"] in ("HIGH", "MEDIUM")]
    if not pairs:
        # Fallback to all pairs
        pairs = pairing_result.get("pairs", [])
    if not pairs:
        return {"error": "No pairs"}

    print(f"  Calibrating Instagram WebP quality on {len(pairs)} paired images...")

    quality_range = list(range(60, 96))
    quality_results = {}

    for q in quality_range:
        ratios = []
        for p in pairs:
            real_props = get_image_properties(p["posted_path"])
            real_bpp = real_props["bpp"]

            img = Image.open(p["orig_path"])
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Instagram center-crops to square then resizes to max_dim=1440
            w, h = img.size
            if abs(w - h) > 50:
                img = center_crop_square(img)
                w, h = img.size
            if max(w, h) > 1440:
                scale = 1440 / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=q, method=4)
            sim_bpp = (len(buf.getvalue()) * 8) / (img.size[0] * img.size[1])

            ratio = sim_bpp / real_bpp if real_bpp > 0 else None
            if ratio is not None:
                ratios.append(ratio)

        if ratios:
            median_ratio = statistics.median(ratios)
            abs_dev = abs(median_ratio - 1.0)
            quality_results[q] = {
                "median_ratio": round(median_ratio, 4),
                "mean_ratio": round(statistics.mean(ratios), 4),
                "median_abs_deviation": round(abs_dev, 4),
                "std": round(statistics.stdev(ratios), 4) if len(ratios) > 1 else 0,
            }
            if q % 5 == 0 or abs_dev < 0.05:
                print(f"    q={q}: median_ratio={median_ratio:.4f}, dev={abs_dev:.4f}")

    best_q = min(quality_results.keys(),
                 key=lambda q: quality_results[q]["median_abs_deviation"])

    # LOO-CV
    print(f"  Best q={best_q}. Running LOO cross validation...")
    loo_best_qs = []
    for leave_out_idx in range(len(pairs)):
        cv_pairs = [p for i, p in enumerate(pairs) if i != leave_out_idx]
        cv_best_q, cv_best_dev = best_q, float("inf")
        for q in range(max(60, best_q - 5), min(96, best_q + 6)):
            ratios = []
            for p in cv_pairs:
                real_props = get_image_properties(p["posted_path"])
                real_bpp = real_props["bpp"]
                img = Image.open(p["orig_path"])
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if abs(w - h) > 50:
                    img = center_crop_square(img)
                    w, h = img.size
                if max(w, h) > 1440:
                    scale = 1440 / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=q, method=4)
                sim_bpp = (len(buf.getvalue()) * 8) / (img.size[0] * img.size[1])
                ratio = sim_bpp / real_bpp if real_bpp > 0 else None
                if ratio is not None:
                    ratios.append(ratio)
            if ratios:
                dev = abs(statistics.median(ratios) - 1.0)
                if dev < cv_best_dev:
                    cv_best_dev = dev
                    cv_best_q = q
        loo_best_qs.append(cv_best_q)

    result = {
        "best_quality": best_q,
        "best_median_ratio": quality_results[best_q]["median_ratio"],
        "best_std": quality_results[best_q]["std"],
        "quality_scan": {str(k): v for k, v in quality_results.items()},
        "loo_cv_best_qs": loo_best_qs,
        "loo_cv_q_median": int(statistics.median(loo_best_qs)),
        "loo_cv_q_range": [min(loo_best_qs), max(loo_best_qs)],
        "loo_cv_stable": max(loo_best_qs) - min(loo_best_qs) <= 3,
    }
    save_result(step_name, "instagram", result)
    log_progress(f"Step 3 IG cal done. best_q={best_q}, LOO range={result['loo_cv_q_range']}")
    return result


# ─── STEP 4: Q-TABLE VERIFICATION ───
def step4_verify_qtables(platform, pairing_result):
    step_name = "step4_qtables"
    existing = load_result(step_name, platform)
    if existing is not None:
        print(f"  Step 4 ({platform}): cached")
        return existing

    if platform == "instagram":
        result = {"platform": platform, "note": "WebP, no Q-tables"}
        save_result(step_name, platform, result)
        return result

    pairs = [p for p in pairing_result.get("pairs", []) if p["confidence"] in ("HIGH", "MEDIUM")]
    if not pairs:
        result = {"error": "No pairs"}
        save_result(step_name, platform, result)
        return result

    print(f"  Verifying Q-tables for {platform} ({len(pairs)} pairs)...")
    v3 = get_v3_module()

    results_list = []
    for p in pairs:
        real_props = get_image_properties(p["posted_path"])

        img = Image.open(p["orig_path"])
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        profile = v3.PLATFORM_PROFILES[platform]["standard"]
        compressed_bytes, new_w, new_h, codec = v3.compress_image(img, profile)

        sim_img = Image.open(io.BytesIO(compressed_bytes))
        record = {
            "orig_file": os.path.basename(p["orig_path"]),
            "real_dims": f"{real_props['width']}x{real_props['height']}",
            "sim_dims": f"{sim_img.size[0]}x{sim_img.size[1]}",
            "dims_match": sim_img.size[0] == real_props["width"] and sim_img.size[1] == real_props["height"],
        }

        if real_props.get("progressive") is not None:
            sim_prog = bool(sim_img.info.get("progressive", False))
            record["progressive_match"] = sim_prog == real_props["progressive"]

        if "qtable_0" in real_props and sim_img.quantization:
            sim_qt0 = list(sim_img.quantization.get(0, []))
            record["qt0_match"] = real_props["qtable_0"] == sim_qt0
            if not record["qt0_match"]:
                diffs = [(i, real_props["qtable_0"][i], sim_qt0[i])
                         for i in range(min(len(real_props["qtable_0"]), len(sim_qt0)))
                         if real_props["qtable_0"][i] != sim_qt0[i]]
                record["qt0_diffs_count"] = len(diffs)

        if "qtable_1" in real_props and sim_img.quantization:
            sim_qt1 = list(sim_img.quantization.get(1, []))
            record["qt1_match"] = real_props["qtable_1"] == sim_qt1
            if not record["qt1_match"]:
                diffs = [(i, real_props["qtable_1"][i], sim_qt1[i])
                         for i in range(min(len(real_props["qtable_1"]), len(sim_qt1)))
                         if real_props["qtable_1"][i] != sim_qt1[i]]
                record["qt1_diffs_count"] = len(diffs)
                record["qt1_diffs_sample"] = diffs[:3]

        results_list.append(record)

    result = {
        "platform": platform, "num_checked": len(results_list),
        "all_dims_match": all(r.get("dims_match", False) for r in results_list),
        "all_progressive_match": all(r.get("progressive_match", True) for r in results_list),
        "all_qt0_match": all(r.get("qt0_match", True) for r in results_list),
        "all_qt1_match": all(r.get("qt1_match", True) for r in results_list),
        "per_image": results_list,
    }
    save_result(step_name, platform, result)
    return result


# ─── STEP 5: HELD-OUT TEST ───
def step5_held_out(platform):
    step_name = "step5_test"
    existing = load_result(step_name, platform)
    if existing is not None:
        print(f"  Step 5 ({platform}): cached")
        return existing

    dirs = PLATFORM_DIRS[platform]
    results = {}

    for test_type, orig_key in [("test_real", "test_real"), ("test_fake", "test_fake")]:
        posted_dir = dirs.get(f"{test_type}_posted")
        if posted_dir is None or not posted_dir.exists():
            results[test_type] = {"error": f"No dir: {posted_dir}"}
            continue

        pairing = step1_pair_images(platform, orig_key, posted_dir, force=True)
        bpp = step2_paired_bpp(platform, pairing, orig_type=orig_key, force=True)
        results[test_type] = bpp.get("summary", {})
        results[test_type + "_detail"] = bpp.get("per_image", [])

    save_result(step_name, platform, results)
    return results


# ─── MAIN ───
def run_pipeline():
    print("=" * 70)
    print("v3 AUDIT PIPELINE - PHASE 2 (FIXED)")
    print("=" * 70)
    log_progress("Starting v3 Phase 2 pipeline")

    # Step 1: Pairing (reuse cache for WA/FB/TG; redo IG with center-crop)
    print("\n" + "-" * 50)
    print("STEP 1: PERCEPTUAL HASH + SSIM PAIRING")
    print("-" * 50)

    pairings = {}
    for platform in ["whatsapp", "facebook", "telegram", "instagram"]:
        dirs = PLATFORM_DIRS[platform]
        print(f"\n[{platform.upper()}]")
        force_ig = (platform == "instagram")  # Force redo for Instagram
        pairings[(platform, "real")] = step1_pair_images(
            platform, "real", dirs["real_posted"], force=force_ig)
        pairings[(platform, "fake")] = step1_pair_images(
            platform, "fake", dirs["fake_posted"], force=force_ig)

    # Print pairing summary
    print("\n\nSTEP 1 SUMMARY:")
    for (plat, ot), pr in pairings.items():
        h = pr.get("high_confidence_count", 0)
        m = pr.get("medium_confidence_count", 0)
        lo = pr.get("low_confidence_count", 0)
        n = pr.get("num_pairs", 0)
        print(f"  {plat:12s}/{ot:6s}: {n} pairs (H={h}, M={m}, L={lo})")

    log_progress("Step 1 pairing done")

    # Step 2: Paired BPP (force redo for all to get v1 data)
    print("\n" + "-" * 50)
    print("STEP 2: PAIRED BPP COMPARISON")
    print("-" * 50)

    bpp_results = {}
    for platform in ["whatsapp", "facebook", "telegram", "instagram"]:
        print(f"\n[{platform.upper()}]")
        for orig_type in ["real", "fake"]:
            bpp_results[(platform, orig_type)] = step2_paired_bpp(
                platform, pairings[(platform, orig_type)],
                orig_type=orig_type, force=True)

    log_progress("Step 2 paired BPP done")

    # Print BPP summary
    print("\n\nSTEP 2 SUMMARY (paired BPP ratios):")
    print(f"  {'Platform/Type':18s} | {'Real BPP':>9s} | {'v1 Ratio':>9s} | {'v3 Ratio':>9s} | Verdict")
    print("  " + "-" * 70)
    for (plat, ot), br in bpp_results.items():
        s = br.get("summary", {})
        real_bpp = s.get("real_bpp_median")
        v1_r = s.get("v1_ratio_median")
        v3_r = s.get("v3_ratio_median")
        real_s = f"{real_bpp:.4f}" if real_bpp else "N/A"
        v1_s = f"{v1_r:.4f}" if v1_r else "N/A"
        v3_s = f"{v3_r:.4f}" if v3_r else "N/A"
        verdict = "OK" if v3_r and 0.85 <= v3_r <= 1.15 else "NEEDS FIX" if v3_r else "NO DATA"
        print(f"  {plat:12s}/{ot:4s} | {real_s:>9s} | {v1_s:>9s} | {v3_s:>9s} | {verdict}")

    # Step 3: Instagram calibration
    print("\n" + "-" * 50)
    print("STEP 3: INSTAGRAM QUALITY CALIBRATION")
    print("-" * 50)
    ig_cal = calibrate_instagram_quality(pairings[("instagram", "real")])
    print(f"  Best quality: {ig_cal.get('best_quality', '?')}")
    print(f"  LOO-CV stable: {ig_cal.get('loo_cv_stable', '?')}")
    print(f"  LOO-CV range: {ig_cal.get('loo_cv_q_range', '?')}")

    # Step 4: Q-table verification
    print("\n" + "-" * 50)
    print("STEP 4: Q-TABLE VERIFICATION")
    print("-" * 50)
    for platform in ["whatsapp", "facebook", "telegram"]:
        print(f"\n[{platform.upper()}]")
        qt = step4_verify_qtables(platform, pairings[(platform, "real")])
        print(f"  Dims: {qt.get('all_dims_match', '?')}, "
              f"Prog: {qt.get('all_progressive_match', '?')}, "
              f"QT0: {qt.get('all_qt0_match', '?')}, "
              f"QT1: {qt.get('all_qt1_match', '?')}")

    log_progress("Steps 3-4 done")

    # Step 5: Held-out test
    print("\n" + "-" * 50)
    print("STEP 5: HELD-OUT TEST VALIDATION")
    print("-" * 50)
    for platform in ["whatsapp", "facebook", "telegram", "instagram"]:
        print(f"\n[{platform.upper()}]")
        test = step5_held_out(platform)
        for tt in ["test_real", "test_fake"]:
            s = test.get(tt, {})
            v3r = s.get("v3_ratio_median")
            print(f"  {tt}: v3_ratio={v3r:.4f}" if v3r else f"  {tt}: N/A")

    log_progress("Step 5 held-out done")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    log_progress("v3 Phase 2 pipeline complete")


if __name__ == "__main__":
    run_pipeline()
