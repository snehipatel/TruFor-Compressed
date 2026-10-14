import os, sys, json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("D:/ml_project")
sys.path.insert(0, str(BASE_DIR))

from study.evaluator_stage2 import collect_compressed_test_suite

# 1. Test Suite
print("Collecting test suite...")
suite = collect_compressed_test_suite(max_per_class=50)

v3_base = BASE_DIR / "results" / "simulated_v3"

test_items = []
for ds_name in ["CocoGlide", "Columbia", "CASIA1.0"]:
    for plat in ["original", "whatsapp", "instagram", "facebook", "telegram"]:
        ext = ".webp" if plat == "instagram" else ".jpg"
        
        # Real
        reals = suite[ds_name]["original"]["real"]
        fakes = suite[ds_name]["original"]["fake"]
        
        for r_p in reals:
            stem = Path(r_p).stem
            if plat == "original":
                target_p = r_p
            else:
                matches = list((v3_base / ds_name / plat).rglob(f"{stem}{ext}"))
                target_p = str(matches[0]) if matches else None
            test_items.append({
                "dataset": ds_name,
                "platform": plat,
                "label": 0,
                "orig_path": r_p,
                "eval_path": target_p
            })
            
        for f_p in fakes:
            stem = Path(f_p).stem
            if plat == "original":
                target_p = f_p
            else:
                matches = list((v3_base / ds_name / plat).rglob(f"{stem}{ext}"))
                target_p = str(matches[0]) if matches else None
            test_items.append({
                "dataset": ds_name,
                "platform": plat,
                "label": 1,
                "orig_path": f_p,
                "eval_path": target_p
            })

print(f"Total test items: {len(test_items)} (Expected: 15 conditions * 100 = 1500)")
missing_test = [x for x in test_items if x["eval_path"] is None or not os.path.exists(x["eval_path"])]
print(f"Missing test files: {len(missing_test)}")

# 2. Validation Suite
print("\nCollecting validation suite...")
val_manifest_path = BASE_DIR / "study" / "data_splits_stage2" / "val_manifest.json"
with open(val_manifest_path) as f:
    val_data = json.load(f)

# Group by (platform, label)
by_plat_label = defaultdict(list)
for s in val_data:
    p = s.get("platform", "original")
    lbl = s["label"]
    by_plat_label[(p, lbl)].append(s)

val_items = []
for (p, lbl), s_list in by_plat_label.items():
    for s in s_list[:50]:  # 50 real, 50 fake per platform
        val_items.append(s)

print(f"Total selected validation items: {len(val_items)} (Expected: 5 platforms * 100 = 500)")
plat_val_counts = defaultdict(int)
for v in val_items:
    plat_val_counts[v.get("platform", "original")] += 1
print("Validation platform breakdown:", dict(plat_val_counts))
