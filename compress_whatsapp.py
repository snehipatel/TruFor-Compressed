#!/usr/bin/env python3
"""
Compress images using the exact lossless/near-lossless JPEG settings
used for WhatsApp preprocessing in this dataset pipeline:
- Format: JPEG
- Quality: 98
- Chroma Subsampling: 4:4:4 (subsampling=0)
- Preserves full resolution and color integrity
"""

import os
import sys
import argparse
from PIL import Image

def compress_image(src_path, dst_path, quality=98, subsampling=0):
    with Image.open(src_path) as im:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(dst_path, format='JPEG', quality=quality, subsampling=subsampling)

def process_directory(input_dir, output_dir, quality=98, subsampling=0):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    
    if not files:
        print(f"No valid image files found in '{input_dir}'")
        return []

    print(f"Compressing {len(files)} images from '{input_dir}' to '{output_dir}'")
    print(f"Settings: JPEG Quality={quality}, Chroma Subsampling={subsampling} (4:4:4)")
    print("-" * 75)
    print(f"{'Filename':<25} {'Original Size':<15} {'Compressed Size':<15} {'Ratio':<10}")
    print("-" * 75)

    results = []
    total_orig = 0
    total_comp = 0

    for f in files:
        src_path = os.path.join(input_dir, f)
        stem = os.path.splitext(f)[0]
        dst_name = f"{stem}.jpg"
        dst_path = os.path.join(output_dir, dst_name)

        orig_size = os.path.getsize(src_path)
        compress_image(src_path, dst_path, quality=quality, subsampling=subsampling)
        comp_size = os.path.getsize(dst_path)

        ratio = (comp_size / orig_size) * 100
        total_orig += orig_size
        total_comp += comp_size

        print(f"{dst_name:<25} {orig_size:>12,} B {comp_size:>12,} B {ratio:>8.1f}%")
        results.append({
            'source': src_path,
            'dest': dst_path,
            'orig_size': orig_size,
            'comp_size': comp_size,
            'ratio': ratio
        })

    print("-" * 75)
    total_ratio = (total_comp / total_orig) * 100 if total_orig > 0 else 0
    print(f"{'TOTAL':<25} {total_orig:>12,} B {total_comp:>12,} B {total_ratio:>8.1f}%")
    print(f"Compression completed successfully. {len(results)} images written.")
    return results

def main():
    parser = argparse.ArgumentParser(description="Lossless/near-lossless WhatsApp JPEG pre-compression tool")
    parser.add_argument('--input_dir', '-i', default='DATA/real_test', help="Source directory containing raw images (default: DATA/real_test)")
    parser.add_argument('--output_dir', '-o', default='DATA/whatsapp_real_test_losslessly_compressed', help="Destination directory (default: DATA/whatsapp_real_test_losslessly_compressed)")
    parser.add_argument('--quality', '-q', type=int, default=98, help="JPEG quality (default: 98)")
    parser.add_argument('--subsampling', '-s', type=int, default=0, help="Chroma subsampling (0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0; default: 0)")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, quality=args.quality, subsampling=args.subsampling)

if __name__ == '__main__':
    main()
