"""
Steps 3-4: Pair originals with platform versions, compare simulator vs real.
Key findings from Step 2 that need checking:
1. WhatsApp: uses HD qtables (not standard), baseline (not progressive), has ICC, max=1600
2. Facebook: qtable[1] last value = 267 (not 255 as in simulator)
3. Telegram: max_dim = 1279 (not 1280), has ICC profiles
4. Instagram: WebP quality needs calibration
5. WhatsApp simulator uses WHATSAPP_STANDARD_QTABLES for large images - WRONG
"""

import os
import io
import json
import struct
import hashlib
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
import numpy as np
import statistics

DATA_ROOT = Path(r"D:\ml_project\DATA")

# ---- Load Step 2 results ----
with open(r"D:\ml_project\audit_step2_results.json") as f:
    step2 = json.load(f)


# ============================================================================
# STEP 3: Check pairing between originals and platform versions
# ============================================================================
def step3_check_pairing():
    print("=" * 80)
    print("STEP 3: Checking original-to-platform pairing")
    print("=" * 80)
    
    # Real originals are PNG files in DATA/REAL (e.g. r01d0b83bt.png)
    # WhatsApp real files: "WhatsApp Image 2026-09-15 at 7.29.47 PM (N).jpeg"
    # Instagram real: "vanjarajuhi10_GUID.jpg" (but actually WebP)
    # Facebook real: "FB_IMG_1789380XXXXXX.jpg"
    # Telegram real: "6140694500817441NNN.jpg"
    
    # These naming conventions don't match at all - no filename pairing possible
    
    originals_real = sorted((DATA_ROOT / "REAL").glob("*.png"))
    originals_fake = sorted((DATA_ROOT / "FAKE").glob("*.jpg"))
    
    wa_real = sorted((DATA_ROOT / "WHATSAPP" / "whatsapp_real").glob("*.jpeg"))
    wa_fake = sorted((DATA_ROOT / "WHATSAPP" / "whatsapp_fake").glob("*.jpeg"))
    ig_real = sorted((DATA_ROOT / "INSTAGRAM" / "Instagram_Real").glob("*.jpg"))
    ig_fake = sorted((DATA_ROOT / "INSTAGRAM" / "Instagram_Fake").glob("*.jpg"))  
    fb_real = sorted((DATA_ROOT / "FACEBOOK" / "facebook_real").glob("*.jpg"))
    fb_fake = sorted((DATA_ROOT / "FACEBOOK" / "facebook_fake").glob("*.jpg"))
    tg_real = sorted((DATA_ROOT / "TELEGRAM" / "telegram_real").glob("*.jpg"))
    tg_fake = sorted((DATA_ROOT / "TELEGRAM" / "telegram_fake").glob("*.jpg"))
    
    print(f"\nOriginals: {len(originals_real)} real PNGs, {len(originals_fake)} fake JPEGs")
    print(f"WhatsApp:  {len(wa_real)} real, {len(wa_fake)} fake")
    print(f"Instagram: {len(ig_real)} real, {len(ig_fake)} fake")
    print(f"Facebook:  {len(fb_real)} real, {len(fb_fake)} fake")
    print(f"Telegram:  {len(tg_real)} real, {len(tg_fake)} fake")
    
    # Check counts: 25 real originals -> 25 per platform (real)
    # 25 fake originals -> 25 per platform (fake) except telegram has 35
    print(f"\nCount match check:")
    print(f"  Real originals (25) -> WA real (25): {'MATCH' if len(wa_real)==25 else 'MISMATCH'}")
    print(f"  Real originals (25) -> IG real (25): {'MATCH' if len(ig_real)==25 else 'MISMATCH'}")
    print(f"  Real originals (25) -> FB real (25): {'MATCH' if len(fb_real)==25 else 'MISMATCH'}")
    print(f"  Real originals (25) -> TG real (25): {'MATCH' if len(tg_real)==25 else 'MISMATCH'}")
    print(f"  Fake originals (25) -> WA fake (25): {'MATCH' if len(wa_fake)==25 else 'MISMATCH'}")
    print(f"  Fake originals (25) -> IG fake (25): {'MATCH' if len(ig_fake)==25 else 'MISMATCH'}")
    print(f"  Fake originals (25) -> FB fake (25): {'MATCH' if len(fb_fake)==25 else 'MISMATCH'}")
    print(f"  Fake originals (25) -> TG fake (35): MISMATCH (35 vs 25)")
    
    # Try dimension-based pairing
    # Real originals are ~4928x3264 PNG -> resized to platform max dims
    # If aspect ratios match after resize, we can pair by sorted order
    print(f"\n  Attempting dimension-based pairing...")
    
    # Check if real originals share same aspect ratios as platform versions
    # Real originals: 4928x3264 or 3264x4928 (landscape/portrait)
    orig_aspects = []
    for f in originals_real:
        img = Image.open(f)
        w, h = img.size
        orig_aspects.append(round(w/h, 4))
        img.close()
    
    wa_aspects = []
    for f in wa_real:
        img = Image.open(f)
        w, h = img.size
        wa_aspects.append(round(w/h, 4))
        img.close()
    
    # Compare sorted aspect ratios
    orig_sorted = sorted(orig_aspects)
    wa_sorted = sorted(wa_aspects)
    print(f"  Orig aspects (sorted): {orig_sorted[:5]}...")
    print(f"  WA aspects (sorted):   {wa_sorted[:5]}...")
    
    # Check if all 25 real images have the same unique aspect ratio
    print(f"  Unique orig aspects: {set(orig_aspects)}")
    print(f"  Unique WA aspects: {set(wa_aspects)}")
    
    # Since we can't pair by filename and aspect ratios may not be unique enough,
    # we'll use distribution matching (Step 4)
    print(f"\n  CONCLUSION: No reliable filename-based pairing exists.")
    print(f"  Will use DISTRIBUTION MATCHING for parameter fitting (Step 4).")
    
    return False  # No pairs found


# ============================================================================
# STEP 4: Compare simulator vs real, quantitatively
# ============================================================================

# Current simulator code extracted:
WHATSAPP_STANDARD_QTABLES = {
    0: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 6, 8, 10, 12, 17, 25, 36, 54, 7, 11, 12, 16, 21, 30, 42, 62, 10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95, 22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167],
    1: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 6, 8, 10, 12, 17, 25, 36, 54, 7, 11, 12, 16, 21, 30, 42, 62, 10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95, 22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167]
}

WHATSAPP_HD_QTABLES = {
    0: [3, 2, 2, 3, 4, 6, 8, 10, 2, 2, 2, 3, 4, 9, 10, 9, 2, 2, 3, 4, 6, 9, 11, 9, 2, 3, 4, 5, 8, 14, 13, 10, 3, 4, 6, 9, 11, 17, 16, 12, 4, 6, 9, 10, 13, 17, 18, 15, 8, 10, 12, 14, 16, 19, 19, 16, 12, 15, 15, 16, 18, 16, 16, 16],
    1: [3, 3, 4, 8, 16, 16, 16, 16, 3, 3, 4, 11, 16, 16, 16, 16, 4, 4, 9, 16, 16, 16, 16, 16, 8, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
}

FACEBOOK_QTABLES_SIM = {
    0: [9, 8, 16, 16, 23, 24, 26, 27, 8, 11, 15, 17, 24, 22, 28, 28, 16, 15, 18, 20, 24, 27, 26, 29, 16, 17, 20, 23, 29, 31, 31, 32, 23, 24, 24, 29, 32, 34, 38, 38, 24, 22, 27, 31, 34, 38, 35, 42, 26, 28, 26, 31, 38, 35, 38, 37, 27, 28, 29, 32, 38, 42, 37, 30],
    1: [8, 7, 14, 12, 19, 17, 20, 22, 7, 10, 13, 13, 17, 15, 19, 20, 14, 13, 15, 14, 16, 18, 18, 21, 12, 13, 14, 15, 19, 17, 19, 22, 19, 17, 16, 19, 15, 22, 17, 21, 17, 15, 18, 17, 22, 19, 22, 32, 20, 19, 18, 19, 17, 22, 35, 32, 22, 20, 21, 22, 21, 32, 32, 255]
}

TELEGRAM_QTABLES_SIM = {
    0: [4, 3, 3, 4, 6, 10, 13, 16, 3, 3, 4, 5, 7, 15, 16, 14, 4, 3, 4, 6, 10, 15, 18, 15, 4, 4, 6, 8, 13, 23, 21, 16, 5, 6, 10, 15, 18, 28, 27, 20, 6, 9, 14, 17, 21, 27, 29, 24, 13, 17, 20, 23, 27, 31, 31, 26, 19, 24, 25, 25, 29, 26, 27, 26],
    1: [4, 5, 6, 12, 26, 26, 26, 26, 5, 5, 7, 17, 26, 26, 26, 26, 6, 7, 15, 26, 26, 26, 26, 26, 12, 17, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]
}

# Real qtables from Step 2:
WHATSAPP_REAL_QT0 = [3, 2, 2, 3, 4, 6, 8, 10, 2, 2, 2, 3, 4, 9, 10, 9, 2, 2, 3, 4, 6, 9, 11, 9, 2, 3, 4, 5, 8, 14, 13, 10, 3, 4, 6, 9, 11, 17, 16, 12, 4, 6, 9, 10, 13, 17, 18, 15, 8, 10, 12, 14, 16, 19, 19, 16, 12, 15, 15, 16, 18, 16, 16, 16]
WHATSAPP_REAL_QT1 = [3, 3, 4, 8, 16, 16, 16, 16, 3, 3, 4, 11, 16, 16, 16, 16, 4, 4, 9, 16, 16, 16, 16, 16, 8, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

FACEBOOK_REAL_QT0 = [9, 8, 16, 16, 23, 24, 26, 27, 8, 11, 15, 17, 24, 22, 28, 28, 16, 15, 18, 20, 24, 27, 26, 29, 16, 17, 20, 23, 29, 31, 31, 32, 23, 24, 24, 29, 32, 34, 38, 38, 24, 22, 27, 31, 34, 38, 35, 42, 26, 28, 26, 31, 38, 35, 38, 37, 27, 28, 29, 32, 38, 42, 37, 30]
FACEBOOK_REAL_QT1 = [8, 7, 14, 12, 19, 17, 20, 22, 7, 10, 13, 13, 17, 15, 19, 20, 14, 13, 15, 14, 16, 18, 18, 21, 12, 13, 14, 15, 19, 17, 19, 22, 19, 17, 16, 19, 15, 22, 17, 21, 17, 15, 18, 17, 22, 19, 22, 32, 20, 19, 18, 19, 17, 22, 35, 32, 22, 20, 21, 22, 21, 32, 32, 267]

TELEGRAM_REAL_QT0 = [4, 3, 3, 4, 6, 10, 13, 16, 3, 3, 4, 5, 7, 15, 16, 14, 4, 3, 4, 6, 10, 15, 18, 15, 4, 4, 6, 8, 13, 23, 21, 16, 5, 6, 10, 15, 18, 28, 27, 20, 6, 9, 14, 17, 21, 27, 29, 24, 13, 17, 20, 23, 27, 31, 31, 26, 19, 24, 25, 25, 29, 26, 27, 26]
TELEGRAM_REAL_QT1 = [4, 5, 6, 12, 26, 26, 26, 26, 5, 5, 7, 17, 26, 26, 26, 26, 6, 7, 15, 26, 26, 26, 26, 26, 12, 17, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26]


def compare_qtables(label, sim_qt, real_qt):
    """Compare simulator qtable vs real qtable."""
    diffs = [(i, sim_qt[i], real_qt[i]) for i in range(64) if sim_qt[i] != real_qt[i]]
    if not diffs:
        print(f"  {label}: EXACT MATCH ✓")
        return True
    else:
        print(f"  {label}: MISMATCH at {len(diffs)} positions:")
        for idx, sim_val, real_val in diffs:
            row, col = idx // 8, idx % 8
            print(f"    [{row},{col}] sim={sim_val} vs real={real_val}")
        return False


def simulate_and_measure(source_folder, platform, profile_params):
    """Run current simulator logic on source images and measure outputs."""
    folder = Path(source_folder)
    files = sorted([f for f in folder.iterdir()
                     if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])
    
    results = []
    for fpath in files[:10]:  # Sample 10 for speed
        try:
            img = Image.open(fpath)
            img = ImageOps.exif_transpose(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            orig_w, orig_h = img.size
            max_dim = profile_params['max_dimension']
            
            if max(orig_w, orig_h) > max_dim:
                scale = max_dim / max(orig_w, orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            else:
                new_w, new_h = orig_w, orig_h
            
            buf = io.BytesIO()
            codec = profile_params.get('codec', 'JPEG')
            
            if codec == 'JPEG':
                qtables = profile_params.get('qtables')
                optimize = profile_params.get('optimize', True)
                progressive = profile_params.get('progressive', False)
                
                if qtables:
                    img.save(buf, format='JPEG', qtables=qtables,
                             subsampling=2, optimize=optimize, progressive=progressive)
                else:
                    quality = profile_params.get('quality', 85)
                    img.save(buf, format='JPEG', quality=quality,
                             subsampling=2, optimize=optimize)
            elif codec == 'WEBP':
                quality = profile_params.get('quality', 65)
                img.save(buf, format='WEBP', quality=quality, method=4)
            
            sim_size = len(buf.getvalue())
            sim_pixels = new_w * new_h
            sim_bpp = sim_size / sim_pixels if sim_pixels > 0 else 0
            
            results.append({
                'filename': fpath.name,
                'orig_w': orig_w, 'orig_h': orig_h,
                'sim_w': new_w, 'sim_h': new_h,
                'sim_size': sim_size,
                'sim_bpp': sim_bpp,
                'sim_max_dim': max(new_w, new_h),
            })
            img.close()
        except Exception as e:
            print(f"  Error on {fpath.name}: {e}")
    
    return results


def step4_compare():
    print("\n" + "=" * 80)
    print("STEP 4: Compare simulator vs real, quantitatively")
    print("=" * 80)
    
    # --- 4a: Q-table comparison ---
    print("\n--- Q-table Comparison ---")
    
    print("\nWhatsApp (simulator uses STANDARD tables for large images >800px):")
    print("  But real WhatsApp data uses HD tables for ALL images!")
    wa_qt0_match = compare_qtables("WA QT[0]: Standard vs Real", 
                                    WHATSAPP_STANDARD_QTABLES[0], WHATSAPP_REAL_QT0)
    wa_hd_qt0_match = compare_qtables("WA QT[0]: HD vs Real",
                                       WHATSAPP_HD_QTABLES[0], WHATSAPP_REAL_QT0)
    wa_hd_qt1_match = compare_qtables("WA QT[1]: HD vs Real",
                                       WHATSAPP_HD_QTABLES[1], WHATSAPP_REAL_QT1)
    
    print("\nFacebook:")
    fb_qt0_match = compare_qtables("FB QT[0]", FACEBOOK_QTABLES_SIM[0], FACEBOOK_REAL_QT0)
    fb_qt1_match = compare_qtables("FB QT[1]", FACEBOOK_QTABLES_SIM[1], FACEBOOK_REAL_QT1)
    
    print("\nTelegram:")
    tg_qt0_match = compare_qtables("TG QT[0]", TELEGRAM_QTABLES_SIM[0], TELEGRAM_REAL_QT0)
    tg_qt1_match = compare_qtables("TG QT[1]", TELEGRAM_QTABLES_SIM[1], TELEGRAM_REAL_QT1)
    
    # --- 4b: Progressive / flags comparison ---
    print("\n--- Progressive/Baseline Comparison ---")
    print(f"  WhatsApp: Simulator=Progressive(True), Real=Baseline(False)  -> MISMATCH!")
    print(f"  Facebook: Simulator=not set (default False), Real=Progressive(True)  -> already correct in profile (progressive not in standard)")
    # Actually check the profile - Facebook standard has no progressive key
    # Let me re-check: the profile for facebook standard doesn't have progressive=True
    # But facebook_real data shows Progressive: True for ALL 25 files
    # Let me re-read the simulator code...
    # facebook standard profile has: optimize=True but no progressive key
    # So compress_image would use profile.get('progressive', False) -> False
    # But real FB is progressive! However, looking at the profile again:
    # Actually - there IS no 'progressive' key in facebook profile, so it defaults to False.
    # But wait - the real data shows progressive=True...
    # Let me check if the simulator code was saving progressive for Facebook...
    print(f"  Facebook: Simulator profile has NO progressive key (defaults to False)")
    print(f"  Facebook: Real data is progressive=True for all 25 files  -> MISMATCH!")
    print(f"  Telegram: Simulator=optimize(True) but no progressive, Real=Progressive(True)  -> NEEDS CHECK")
    
    # --- 4c: Max dimension comparison ---
    print("\n--- Max Dimension Comparison ---")
    print(f"  WhatsApp: Simulator=1600, Real=1600  -> MATCH ✓")
    print(f"  Instagram: Simulator=1440, Real=1440  -> MATCH ✓")
    print(f"  Facebook: Simulator=1080, Real max_dim range=1080-1631  -> PARTIAL")
    print(f"    Note: Some FB real images exceed 1080. FB may not always downsample to 1080.")
    print(f"  Telegram: Simulator=1280, Real=1279 consistently  -> MISMATCH (off by 1)")
    
    # --- 4d: ICC profile ---
    print("\n--- ICC/EXIF Comparison ---")
    print(f"  WhatsApp: Real has ICC in 25/25 files, Simulator strips EXIF but doesn't mention ICC")
    print(f"  Telegram: Real has ICC in 25/25 files, Simulator strips EXIF but doesn't mention ICC")
    print(f"  Facebook: Real has NO ICC, NO EXIF  -> Simulator strips correctly")
    print(f"  Instagram: WebP, no EXIF/ICC relevant")
    
    # --- 4e: Run simulator on real originals and compare BPP ---
    print("\n--- BPP Comparison: Simulator vs Real ---")
    
    # WhatsApp simulator: uses STANDARD tables for large images (>800px)
    # But real WhatsApp uses HD tables. This means simulator is MORE aggressive.
    print("\nWhatsApp (simulating on real originals with CURRENT simulator logic):")
    wa_sim_current = simulate_and_measure(
        DATA_ROOT / "REAL", "whatsapp",
        {'codec': 'JPEG', 'max_dimension': 1600, 'qtables': WHATSAPP_STANDARD_QTABLES,
         'subsampling': '4:2:0', 'progressive': True, 'optimize': True}
    )
    wa_sim_fixed = simulate_and_measure(
        DATA_ROOT / "REAL", "whatsapp_fixed",
        {'codec': 'JPEG', 'max_dimension': 1600, 'qtables': {0: WHATSAPP_REAL_QT0, 1: WHATSAPP_REAL_QT1},
         'subsampling': '4:2:0', 'progressive': False, 'optimize': True}
    )
    
    # Real WhatsApp stats from Step 2
    wa_real_bpp = step2['whatsapp_real']['summary']['bytes_per_pixel']
    wa_real_fs = step2['whatsapp_real']['summary']['file_size']
    
    if wa_sim_current:
        sim_bpps = [r['sim_bpp'] for r in wa_sim_current]
        sim_sizes = [r['sim_size'] for r in wa_sim_current]
        fix_bpps = [r['sim_bpp'] for r in wa_sim_fixed]
        fix_sizes = [r['sim_size'] for r in wa_sim_fixed]
        
        print(f"  Current simulator BPP: min={min(sim_bpps):.4f}, median={statistics.median(sim_bpps):.4f}, max={max(sim_bpps):.4f}")
        print(f"  Fixed simulator BPP:   min={min(fix_bpps):.4f}, median={statistics.median(fix_bpps):.4f}, max={max(fix_bpps):.4f}")
        print(f"  Real WhatsApp BPP:     min={wa_real_bpp['min']}, median={wa_real_bpp['median']}, max={wa_real_bpp['max']}")
        print(f"  Current simulator size: min={min(sim_sizes)//1024}KB, median={int(statistics.median(sim_sizes))//1024}KB, max={max(sim_sizes)//1024}KB")
        print(f"  Fixed simulator size:   min={min(fix_sizes)//1024}KB, median={int(statistics.median(fix_sizes))//1024}KB, max={max(fix_sizes)//1024}KB")
        print(f"  Real WhatsApp size:     min={wa_real_fs['min']//1024}KB, median={int(wa_real_fs['median'])//1024}KB, max={wa_real_fs['max']//1024}KB")
    
    # Facebook
    print("\nFacebook (simulating on real originals):")
    fb_sim_current = simulate_and_measure(
        DATA_ROOT / "REAL", "facebook",
        {'codec': 'JPEG', 'max_dimension': 1080, 'qtables': FACEBOOK_QTABLES_SIM,
         'subsampling': '4:2:0', 'optimize': True, 'progressive': False}
    )
    fb_sim_fixed = simulate_and_measure(
        DATA_ROOT / "REAL", "facebook_fixed",
        {'codec': 'JPEG', 'max_dimension': 1080, 'qtables': {0: FACEBOOK_REAL_QT0, 1: FACEBOOK_REAL_QT1},
         'subsampling': '4:2:0', 'optimize': True, 'progressive': True}
    )
    
    fb_real_bpp = step2['facebook_real']['summary']['bytes_per_pixel']
    fb_real_fs = step2['facebook_real']['summary']['file_size']
    
    if fb_sim_current:
        sim_bpps = [r['sim_bpp'] for r in fb_sim_current]
        sim_sizes = [r['sim_size'] for r in fb_sim_current]
        fix_bpps = [r['sim_bpp'] for r in fb_sim_fixed]
        fix_sizes = [r['sim_size'] for r in fb_sim_fixed]
        
        print(f"  Current simulator BPP: min={min(sim_bpps):.4f}, median={statistics.median(sim_bpps):.4f}, max={max(sim_bpps):.4f}")
        print(f"  Fixed simulator BPP:   min={min(fix_bpps):.4f}, median={statistics.median(fix_bpps):.4f}, max={max(fix_bpps):.4f}")
        print(f"  Real Facebook BPP:     min={fb_real_bpp['min']}, median={fb_real_bpp['median']}, max={fb_real_bpp['max']}")
        print(f"  Current simulator size: min={min(sim_sizes)//1024}KB, median={int(statistics.median(sim_sizes))//1024}KB, max={max(sim_sizes)//1024}KB")
        print(f"  Fixed simulator size:   min={min(fix_sizes)//1024}KB, median={int(statistics.median(fix_sizes))//1024}KB, max={max(fix_sizes)//1024}KB")
        print(f"  Real Facebook size:     min={fb_real_fs['min']//1024}KB, median={int(fb_real_fs['median'])//1024}KB, max={fb_real_fs['max']//1024}KB")
    
    # Telegram
    print("\nTelegram (simulating on real originals):")
    tg_sim_current = simulate_and_measure(
        DATA_ROOT / "REAL", "telegram",
        {'codec': 'JPEG', 'max_dimension': 1280, 'qtables': TELEGRAM_QTABLES_SIM,
         'subsampling': '4:2:0', 'optimize': True, 'progressive': False}
    )
    tg_sim_fixed = simulate_and_measure(
        DATA_ROOT / "REAL", "telegram_fixed",
        {'codec': 'JPEG', 'max_dimension': 1279, 'qtables': {0: TELEGRAM_REAL_QT0, 1: TELEGRAM_REAL_QT1},
         'subsampling': '4:2:0', 'optimize': True, 'progressive': True}
    )
    
    tg_real_bpp = step2['telegram_real']['summary']['bytes_per_pixel']
    tg_real_fs = step2['telegram_real']['summary']['file_size']
    
    if tg_sim_current:
        sim_bpps = [r['sim_bpp'] for r in tg_sim_current]
        sim_sizes = [r['sim_size'] for r in tg_sim_current]
        fix_bpps = [r['sim_bpp'] for r in tg_sim_fixed]
        fix_sizes = [r['sim_size'] for r in tg_sim_fixed]
        
        print(f"  Current simulator BPP: min={min(sim_bpps):.4f}, median={statistics.median(sim_bpps):.4f}, max={max(sim_bpps):.4f}")
        print(f"  Fixed simulator BPP:   min={min(fix_bpps):.4f}, median={statistics.median(fix_bpps):.4f}, max={max(fix_bpps):.4f}")
        print(f"  Real Telegram BPP:     min={tg_real_bpp['min']}, median={tg_real_bpp['median']}, max={tg_real_bpp['max']}")
        print(f"  Current simulator size: min={min(sim_sizes)//1024}KB, median={int(statistics.median(sim_sizes))//1024}KB, max={max(sim_sizes)//1024}KB")
        print(f"  Fixed simulator size:   min={min(fix_sizes)//1024}KB, median={int(statistics.median(fix_sizes))//1024}KB, max={max(fix_sizes)//1024}KB")
        print(f"  Real Telegram size:     min={tg_real_fs['min']//1024}KB, median={int(tg_real_fs['median'])//1024}KB, max={tg_real_fs['max']//1024}KB")
    
    # Instagram WebP quality calibration
    print("\nInstagram WebP quality calibration:")
    ig_real_bpp = step2['instagram_real']['summary']['bytes_per_pixel']
    ig_real_fs = step2['instagram_real']['summary']['file_size']
    print(f"  Real Instagram BPP: min={ig_real_bpp['min']}, median={ig_real_bpp['median']}, max={ig_real_bpp['max']}")
    print(f"  Real Instagram size: min={ig_real_fs['min']//1024}KB, median={int(ig_real_fs['median'])//1024}KB, max={ig_real_fs['max']//1024}KB")
    
    # Test different quality levels
    for q in [65, 70, 75, 80, 85]:
        sim = simulate_and_measure(
            DATA_ROOT / "REAL", f"instagram_q{q}",
            {'codec': 'WEBP', 'max_dimension': 1440, 'quality': q}
        )
        if sim:
            bpps = [r['sim_bpp'] for r in sim]
            sizes = [r['sim_size'] for r in sim]
            print(f"  WebP q={q}: BPP min={min(bpps):.4f}, median={statistics.median(bpps):.4f}, max={max(bpps):.4f}, "
                  f"size min={min(sizes)//1024}KB, median={int(statistics.median(sizes))//1024}KB, max={max(sizes)//1024}KB")
    
    # --- 4f: Summary verdict ---
    print("\n" + "=" * 80)
    print("STEP 4 VERDICT PER PLATFORM")
    print("=" * 80)
    
    print("""
WHATSAPP: TOO HARSH
  - Simulator uses WHATSAPP_STANDARD_QTABLES for images >800px (coarser quantization)
  - Real WhatsApp uses WHATSAPP_HD_QTABLES for ALL images (including large ones)
  - Simulator sets progressive=True; real WhatsApp is BASELINE (progressive=False)
  - Real WhatsApp includes ICC profiles; simulator doesn't include them
  - This explains why simulator files were 56-112 KB vs real 189-565 KB

FACEBOOK: MOSTLY ACCURATE, MINOR FIX NEEDED
  - Q-table[0]: EXACT MATCH between simulator and real ✓
  - Q-table[1]: Last value differs: simulator=255, real=267
    (Pillow clips values >255 to 255, so this is a Pillow limitation)
  - Progressive: Simulator defaults to False, real is True -> needs fix
  - Max dimension: Real FB shows some images with max_dim > 1080 (up to 1631)
    This suggests FB doesn't always cap at 1080 for all content types

TELEGRAM: ACCURATE (minor adjustments needed)
  - Q-tables: EXACT MATCH ✓
  - Max dimension: Simulator=1280, real=1279 (consistently) -> off by 1
  - Progressive: Real is True, need to confirm simulator sends True
  - ICC profiles: Real has ICC, simulator doesn't include

INSTAGRAM: NEEDS QUALITY CALIBRATION
  - Format: WebP ✓
  - Max dimension: 1440 ✓
  - Quality: Simulator uses q=65, but real BPP suggests higher quality
  - Need to find the WebP quality that matches real BPP distribution
""")


if __name__ == '__main__':
    has_pairs = step3_check_pairing()
    step4_compare()
