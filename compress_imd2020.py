"""
compress_imd2020.py
===================
Batch-compress every image in the IMD2020 dataset using the exact same
platform simulation engine that lives in SIMULATOR/server.py.

The quantization tables, platform profiles, and compression functions below
are copied verbatim from SIMULATOR/server.py so that this standalone script
requires no Flask or web-framework dependency — only Pillow.

Platforms simulated (standard mode):
  WhatsApp  – max 1600px, empirical Q-tables, 4:2:0, JPEG, progressive
  Instagram – max 1440px, quality=65, WebP
  Telegram  – max 1280px, empirical Q-tables, 4:2:0, JPEG
  Facebook  – max 1080px, empirical Q-tables, 4:2:0, JPEG

Ground-truth mask files (*_mask.png) are copied WITHOUT compression because
applying lossy encoding to binary masks destroys pixel-precision.

Usage:
  python compress_imd2020.py --input IMD2020 [--dry-run]
"""

import os
import io
import sys
import shutil
import argparse
from PIL import Image, ImageOps

# ============================================================================
# EMPIRICAL QUANTIZATION TABLES — copied verbatim from SIMULATOR/server.py
# ============================================================================
WHATSAPP_STANDARD_QTABLES = {
    0: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 6, 8, 10, 12, 17, 25, 36, 54,
        7, 11, 12, 16, 21, 30, 42, 62, 10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95,
        22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167],
    1: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 6, 8, 10, 12, 17, 25, 36, 54,
        7, 11, 12, 16, 21, 30, 42, 62, 10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95,
        22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167],
}

WHATSAPP_HD_QTABLES = {
    0: [3, 2, 2, 3, 4, 6, 8, 10, 2, 2, 2, 3, 4, 9, 10, 9, 2, 2, 3, 4, 6, 9, 11, 9,
        2, 3, 4, 5, 8, 14, 13, 10, 3, 4, 6, 9, 11, 17, 16, 12, 4, 6, 9, 10, 13, 17, 18, 15,
        8, 10, 12, 14, 16, 19, 19, 16, 12, 15, 15, 16, 18, 16, 16, 16],
    1: [3, 3, 4, 8, 16, 16, 16, 16, 3, 3, 4, 11, 16, 16, 16, 16, 4, 4, 9, 16, 16, 16, 16, 16,
        8, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
}

TELEGRAM_QTABLES = {
    0: [4, 3, 3, 4, 6, 10, 13, 16, 3, 3, 4, 5, 7, 15, 16, 14, 4, 3, 4, 6, 10, 15, 18, 15,
        4, 4, 6, 8, 13, 23, 21, 16, 5, 6, 10, 15, 18, 28, 27, 20, 6, 9, 14, 17, 21, 27, 29, 24,
        13, 17, 20, 23, 27, 31, 31, 26, 19, 24, 25, 25, 29, 26, 27, 26],
    1: [4, 5, 6, 12, 26, 26, 26, 26, 5, 5, 7, 17, 26, 26, 26, 26, 6, 7, 15, 26, 26, 26, 26, 26,
        12, 17, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
        26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
}

FACEBOOK_QTABLES = {
    0: [9, 8, 16, 16, 23, 24, 26, 27, 8, 11, 15, 17, 24, 22, 28, 28, 16, 15, 18, 20, 24, 27, 26, 29,
        16, 17, 20, 23, 29, 31, 31, 32, 23, 24, 24, 29, 32, 34, 38, 38, 24, 22, 27, 31, 34, 38, 35, 42,
        26, 28, 26, 31, 38, 35, 38, 37, 27, 28, 29, 32, 38, 42, 37, 30],
    1: [8, 7, 14, 12, 19, 17, 20, 22, 7, 10, 13, 13, 17, 15, 19, 20, 14, 13, 15, 14, 16, 18, 18, 21,
        12, 13, 14, 15, 19, 17, 19, 22, 19, 17, 16, 19, 15, 22, 17, 21, 17, 15, 18, 17, 22, 19, 22, 32,
        20, 19, 18, 19, 17, 22, 35, 32, 22, 20, 21, 22, 21, 32, 32, 255],
}

# ============================================================================
# PLATFORM PROFILES — copied verbatim from SIMULATOR/server.py (standard only)
# ============================================================================
PLATFORM_PROFILES = {
    'whatsapp': {
        'codec': 'JPEG',
        'max_dimension': 1600,
        'adaptive_whatsapp': True,
        'subsampling': '4:2:0',
        'progressive': True,
        'optimize': True,
        'strip_exif': True,
        'label': 'WhatsApp (Standard)',
    },
    'instagram': {
        'codec': 'WEBP',
        'max_dimension': 1440,
        'quality': 65,
        'subsampling': None,
        'strip_exif': True,
        'label': 'Instagram (WebP)',
    },
    'telegram': {
        'codec': 'JPEG',
        'max_dimension': 1280,
        'qtables': TELEGRAM_QTABLES,
        'subsampling': '4:2:0',
        'optimize': True,
        'strip_exif': True,
        'label': 'Telegram (Standard)',
    },
    'facebook': {
        'codec': 'JPEG',
        'max_dimension': 1080,
        'qtables': FACEBOOK_QTABLES,
        'subsampling': '4:2:0',
        'optimize': True,
        'strip_exif': True,
        'label': 'Facebook',
    },
}

# ============================================================================
# COMPRESSION FUNCTIONS — copied verbatim from SIMULATOR/server.py
# ============================================================================

def prepare_image_for_processing(img: Image.Image) -> Image.Image:
    """
    1. Auto-orient based on camera EXIF tag (critical for smartphone photos).
    2. Convert RGBA/palette with transparency to RGB on a clean white canvas.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return img.convert('RGB')


def compress_image(img: Image.Image, profile: dict) -> tuple:
    """
    Simulate platform compression on a PIL Image using exact empirical quantization tables.
    Normalizes EXIF orientation to prevent sideways rotations on phone photos.

    Returns:
        (compressed_bytes: bytes, new_width: int, new_height: int, codec: str)
    """
    img = prepare_image_for_processing(img)
    orig_w, orig_h = img.size
    max_dim = profile['max_dimension']

    # Downscale if needed (preserving natural aspect ratio)
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w, new_h = orig_w, orig_h

    # Compress to bytes
    buf = io.BytesIO()
    codec = profile['codec']
    quality = profile.get('quality', 85)
    optimize = profile.get('optimize', True)
    progressive = profile.get('progressive', False)

    if codec == 'JPEG':
        # Adaptive WhatsApp table:
        # Standard WhatsApp uses the standard compression matrix on high-res camera photos (>800px)
        # and uses HD matrix on already-small images (<=800px) to prevent over-compression.
        if profile.get('adaptive_whatsapp'):
            if max(orig_w, orig_h) > 800:
                qtables = WHATSAPP_STANDARD_QTABLES
            else:
                qtables = WHATSAPP_HD_QTABLES
        else:
            qtables = profile.get('qtables')

        if qtables:
            img.save(buf, format='JPEG', qtables=qtables, subsampling=2,
                     optimize=optimize, progressive=progressive)
        else:
            img.save(buf, format='JPEG', quality=quality, subsampling=2, optimize=optimize)
    elif codec == 'WEBP':
        img.save(buf, format='WEBP', quality=quality, method=4)
    else:
        img.save(buf, format='JPEG', quality=quality)

    compressed_bytes = buf.getvalue()
    return compressed_bytes, new_w, new_h, codec


# ============================================================================
# BATCH PROCESSING LOGIC
# ============================================================================

PLATFORMS = ['whatsapp', 'instagram', 'telegram', 'facebook']
VALID_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def build_file_lists(input_dir: str):
    """
    Walk input_dir recursively and return:
      - images_to_compress: list of absolute paths to non-mask image files
      - masks_to_copy:      list of absolute paths to *_mask.png files
    """
    images_to_compress = []
    masks_to_copy = []

    for root, dirs, files in os.walk(input_dir):
        dirs.sort()   # deterministic order
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXTS:
                continue
            full = os.path.join(root, fname)
            # Ground-truth masks: copy verbatim, no lossy encoding
            if fname.lower().endswith('_mask.png'):
                masks_to_copy.append(full)
            else:
                images_to_compress.append(full)

    return images_to_compress, masks_to_copy


def run_batch(input_dir: str, base_output_dir: str, dry_run_limit: int = None):
    """
    Core batch routine.

    Parameters
    ----------
    input_dir        : Path to IMD2020 root folder.
    base_output_dir  : Parent directory in which IMD2020_<platform> folders are created.
    dry_run_limit    : If set, process only this many images (for smoke-testing).
    """
    images, masks = build_file_lists(input_dir)

    if dry_run_limit and dry_run_limit > 0:
        images = images[:dry_run_limit]
        # Also limit masks to those that live in the same dirs as selected images
        selected_dirs = set(os.path.dirname(p) for p in images)
        masks = [m for m in masks if os.path.dirname(m) in selected_dirs]
        print(f"[DRY RUN] Processing {len(images)} images and {len(masks)} masks.\n")
    else:
        print(f"Processing {len(images)} images and {len(masks)} masks.\n")

    # Create output root directories
    output_dirs = {}
    for plat in PLATFORMS:
        out = os.path.join(base_output_dir, f"IMD2020_{plat}")
        os.makedirs(out, exist_ok=True)
        output_dirs[plat] = out

    success_counts = {p: 0 for p in PLATFORMS}
    failed_images = []   # list of (src_path, platform, error_msg)

    total = len(images)
    for idx, img_path in enumerate(images, 1):
        rel_path = os.path.relpath(img_path, input_dir)
        rel_dir  = os.path.dirname(rel_path)
        filename = os.path.basename(img_path)

        # Progress feedback every 50 images
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] {rel_path}")

        for plat in PLATFORMS:
            out_dir = os.path.join(output_dirs[plat], rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            profile = PLATFORM_PROFILES[plat]
            try:
                raw_img = Image.open(img_path)
                compressed_bytes, new_w, new_h, codec = compress_image(raw_img, profile)

                new_ext   = '.jpg' if codec == 'JPEG' else '.webp'
                new_fname = os.path.splitext(filename)[0] + new_ext
                out_path  = os.path.join(out_dir, new_fname)

                with open(out_path, 'wb') as f:
                    f.write(compressed_bytes)

                success_counts[plat] += 1

            except Exception as e:
                failed_images.append((img_path, plat, str(e)))

    # Copy masks verbatim into all 4 platform folders
    print(f"\nCopying {len(masks)} mask file(s) (unmodified) into all platform folders...")
    for mask_path in masks:
        rel_path = os.path.relpath(mask_path, input_dir)
        rel_dir  = os.path.dirname(rel_path)
        fname    = os.path.basename(mask_path)
        for plat in PLATFORMS:
            out_dir = os.path.join(output_dirs[plat], rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(mask_path, os.path.join(out_dir, fname))

    # ── VERIFICATION SUMMARY ─────────────────────────────────────────────────
    expected_total = len(images) * 4
    actual_total   = sum(success_counts.values())

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total input images (excl. masks): {len(images)}")
    print(f"Total masks copied               : {len(masks)}")
    print()
    print(f"Expected output images           : {len(images)} × 4 = {expected_total}")
    print()
    for plat in PLATFORMS:
        status = "[OK]" if success_counts[plat] == len(images) else "[!!]"
        print(f"  {plat.capitalize():12s}: {success_counts[plat]}  {status}")
    print()
    print(f"Total generated                  : {actual_total}")
    print()

    count_ok   = actual_total == expected_total
    mapping_ok = all(success_counts[p] == len(images) for p in PLATFORMS)

    print(f"Count verification               : {'PASSED' if count_ok   else 'FAILED'}")
    print(f"One-to-four mapping verification : {'PASSED' if mapping_ok else 'FAILED'}")

    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for src, plat, err in failed_images:
            print(f"  [{plat}] {src}  →  {err}")
    else:
        print("\nFailed images: None")

    print("=" * 60)

    return count_ok and mapping_ok


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compress IMD2020 dataset for 4 social media platforms using the TruFor-Compressed simulator engine."
    )
    parser.add_argument(
        '--input', default='IMD2020',
        help="Path to the root IMD2020 dataset directory (default: IMD2020)"
    )
    parser.add_argument(
        '--output', default='.',
        help="Parent directory where IMD2020_<platform> folders are created (default: current directory)"
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Test pipeline on first 10 images only"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory '{args.input}' not found.")
        sys.exit(1)

    limit = 10 if args.dry_run else None
    ok = run_batch(
        input_dir=os.path.abspath(args.input),
        base_output_dir=os.path.abspath(args.output),
        dry_run_limit=limit,
    )
    sys.exit(0 if ok else 1)
