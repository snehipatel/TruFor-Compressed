"""
Step 2: Measure real platform images exhaustively.
Extracts format, dimensions, quantization tables, chroma subsampling,
progressive flag, EXIF/ICC presence, file size, bytes-per-pixel,
and JFIF/Adobe markers for every image in the training DATA folders.
"""

import os
import io
import json
import struct
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import statistics

# ---- Platform folder mapping (training only, NOT test) ----
DATA_ROOT = Path(r"D:\ml_project\DATA")
PLATFORM_FOLDERS = {
    "whatsapp_real":    DATA_ROOT / "WHATSAPP" / "whatsapp_real",
    "whatsapp_fake":    DATA_ROOT / "WHATSAPP" / "whatsapp_fake",
    "instagram_real":   DATA_ROOT / "INSTAGRAM" / "Instagram_Real",
    "instagram_fake":   DATA_ROOT / "INSTAGRAM" / "Instagram_Fake",
    "facebook_real":    DATA_ROOT / "FACEBOOK" / "facebook_real",
    "facebook_fake":    DATA_ROOT / "FACEBOOK" / "facebook_fake",
    "telegram_real":    DATA_ROOT / "TELEGRAM" / "telegram_real",
    "telegram_fake":    DATA_ROOT / "TELEGRAM" / "telegram_fake",
}

ORIGINAL_FOLDERS = {
    "originals_real":   DATA_ROOT / "REAL",
    "originals_fake":   DATA_ROOT / "FAKE",
}


def parse_jpeg_markers(filepath):
    """Parse raw JPEG markers to find JFIF, Adobe, progressive, Huffman tables."""
    markers = {}
    with open(filepath, 'rb') as f:
        data = f.read()

    markers['has_jfif'] = b'JFIF' in data[:50]
    markers['has_adobe'] = b'Adobe' in data[:200]
    markers['has_exif'] = b'Exif' in data[:200]
    markers['has_icc'] = b'ICC_PROFILE' in data

    pos = 0
    markers['is_progressive'] = False
    markers['sof_marker'] = None
    while pos < len(data) - 1:
        if data[pos] == 0xFF:
            marker = data[pos + 1]
            if marker == 0xC0:
                markers['sof_marker'] = 'SOF0 (Baseline)'
                markers['is_progressive'] = False
                break
            elif marker == 0xC2:
                markers['sof_marker'] = 'SOF2 (Progressive)'
                markers['is_progressive'] = True
                break
            elif marker == 0xC1:
                markers['sof_marker'] = 'SOF1 (Extended Sequential)'
                break
        pos += 1

    return markers


def get_subsampling_from_file(filepath):
    """Parse SOF marker to get sampling factors."""
    with open(filepath, 'rb') as f:
        data = f.read()

    pos = 0
    while pos < len(data) - 10:
        if data[pos] == 0xFF and data[pos + 1] in (0xC0, 0xC2):
            length = struct.unpack('>H', data[pos+2:pos+4])[0]
            precision = data[pos + 4]
            height = struct.unpack('>H', data[pos+5:pos+7])[0]
            width = struct.unpack('>H', data[pos+7:pos+9])[0]
            num_components = data[pos + 9]

            components = []
            for i in range(num_components):
                offset = pos + 10 + i * 3
                comp_id = data[offset]
                sampling = data[offset + 1]
                h_samp = (sampling >> 4) & 0x0F
                v_samp = sampling & 0x0F
                qt_id = data[offset + 2]
                components.append({
                    'id': comp_id, 'h_samp': h_samp, 'v_samp': v_samp, 'qt_id': qt_id
                })

            if len(components) >= 3:
                y = components[0]
                cb = components[1]
                if y['h_samp'] == 2 and y['v_samp'] == 2 and cb['h_samp'] == 1 and cb['v_samp'] == 1:
                    return '4:2:0'
                elif y['h_samp'] == 2 and y['v_samp'] == 1 and cb['h_samp'] == 1 and cb['v_samp'] == 1:
                    return '4:2:2'
                elif y['h_samp'] == 1 and y['v_samp'] == 1 and cb['h_samp'] == 1 and cb['v_samp'] == 1:
                    return '4:4:4'
                else:
                    return f'{y["h_samp"]}x{y["v_samp"]}:{cb["h_samp"]}x{cb["v_samp"]}'
            return 'grayscale' if num_components == 1 else 'unknown'
        pos += 1
    return 'unknown'


def estimate_quality_from_qtable(qtable):
    """Estimate JPEG quality from luminance quantization table using IJG method."""
    std_lum = [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    ]

    if not qtable or len(qtable) < 64:
        return None

    q_sum = sum(qtable[:64])
    s_sum = sum(std_lum)

    if q_sum == 0:
        return 100

    if q_sum < s_sum:
        quality = (200 * s_sum - 100 * q_sum) // (2 * s_sum)
    else:
        quality = max(1, (200 * s_sum) // (2 * q_sum))

    return min(100, max(1, quality))


def analyze_image(filepath):
    """Extract comprehensive metadata from a single image file."""
    info = {}
    filepath = str(filepath)
    info['filename'] = os.path.basename(filepath)
    info['file_size'] = os.path.getsize(filepath)

    try:
        img = Image.open(filepath)
        info['format'] = img.format
        info['width'] = img.size[0]
        info['height'] = img.size[1]
        info['mode'] = img.mode
        info['max_dim'] = max(img.size)
        info['min_dim'] = min(img.size)
        info['pixels'] = img.size[0] * img.size[1]
        info['bytes_per_pixel'] = info['file_size'] / info['pixels'] if info['pixels'] > 0 else 0

        if img.format == 'JPEG':
            qtables = img.quantization if hasattr(img, 'quantization') and img.quantization else None
            if qtables:
                info['qtable_0'] = list(qtables.get(0, []))
                info['qtable_1'] = list(qtables.get(1, []))
                info['est_quality'] = estimate_quality_from_qtable(info['qtable_0'])
            else:
                info['qtable_0'] = []
                info['qtable_1'] = []
                info['est_quality'] = None

            info['subsampling'] = get_subsampling_from_file(filepath)
            markers = parse_jpeg_markers(filepath)
            info.update(markers)
            info['has_exif_pil'] = bool(img.info.get('exif'))
            info['has_icc_pil'] = bool(img.info.get('icc_profile'))

        elif img.format == 'WEBP':
            info['is_webp'] = True
            info['est_quality'] = None
            info['subsampling'] = 'N/A (WebP)'

        elif img.format == 'PNG':
            info['is_png'] = True
            info['est_quality'] = None
            info['subsampling'] = 'N/A (PNG)'

        img.close()
    except Exception as e:
        info['error'] = str(e)

    return info


def analyze_folder(folder_path, label):
    """Analyze all images in a folder and return summary statistics."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"  [SKIP] {label}: folder does not exist: {folder}")
        return None

    files = sorted([f for f in folder.iterdir()
                     if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')])

    if not files:
        print(f"  [SKIP] {label}: no image files found")
        return None

    print(f"\n{'='*80}")
    print(f"  Analyzing: {label} ({len(files)} files)")
    print(f"  Path: {folder}")
    print(f"{'='*80}")

    all_info = []
    qtable_0_counter = Counter()
    qtable_1_counter = Counter()

    for f in files:
        info = analyze_image(f)
        all_info.append(info)
        if info.get('qtable_0'):
            qtable_0_counter[tuple(info['qtable_0'])] += 1
        if info.get('qtable_1'):
            qtable_1_counter[tuple(info['qtable_1'])] += 1

    summary = {
        'label': label,
        'count': len(files),
        'formats': dict(Counter(i.get('format', 'unknown') for i in all_info)),
    }

    widths = [i['width'] for i in all_info if 'width' in i]
    heights = [i['height'] for i in all_info if 'height' in i]
    max_dims = [i['max_dim'] for i in all_info if 'max_dim' in i]
    file_sizes = [i['file_size'] for i in all_info if 'file_size' in i]
    bpp = [i['bytes_per_pixel'] for i in all_info if 'bytes_per_pixel' in i]

    if widths:
        summary['width'] = {'min': min(widths), 'max': max(widths), 'median': statistics.median(widths)}
        summary['height'] = {'min': min(heights), 'max': max(heights), 'median': statistics.median(heights)}
        summary['max_dim'] = {'min': min(max_dims), 'max': max(max_dims), 'median': statistics.median(max_dims)}
    if file_sizes:
        summary['file_size'] = {'min': min(file_sizes), 'max': max(file_sizes),
                                 'median': statistics.median(file_sizes), 'mean': statistics.mean(file_sizes)}
    if bpp:
        summary['bytes_per_pixel'] = {'min': round(min(bpp), 4), 'max': round(max(bpp), 4),
                                       'median': round(statistics.median(bpp), 4), 'mean': round(statistics.mean(bpp), 4)}

    jpeg_files = [i for i in all_info if i.get('format') == 'JPEG']
    if jpeg_files:
        qualities = [i['est_quality'] for i in jpeg_files if i.get('est_quality') is not None]
        subsamplings = Counter(i.get('subsampling', 'unknown') for i in jpeg_files)
        progressive = Counter(i.get('is_progressive', False) for i in jpeg_files)
        exif_count = sum(1 for i in jpeg_files if i.get('has_exif') or i.get('has_exif_pil'))
        icc_count = sum(1 for i in jpeg_files if i.get('has_icc') or i.get('has_icc_pil'))
        jfif_count = sum(1 for i in jpeg_files if i.get('has_jfif'))
        adobe_count = sum(1 for i in jpeg_files if i.get('has_adobe'))

        summary['est_quality'] = {'min': min(qualities), 'max': max(qualities),
                                   'median': statistics.median(qualities)} if qualities else None
        summary['subsampling'] = dict(subsamplings)
        summary['progressive'] = dict(progressive)
        summary['exif_present'] = exif_count
        summary['icc_present'] = icc_count
        summary['jfif_present'] = jfif_count
        summary['adobe_present'] = adobe_count
        summary['unique_qtable_0_count'] = len(qtable_0_counter)
        summary['unique_qtable_1_count'] = len(qtable_1_counter)

        if qtable_0_counter:
            most_common_qt0 = qtable_0_counter.most_common(1)[0]
            summary['dominant_qtable_0'] = list(most_common_qt0[0])
            summary['dominant_qtable_0_count'] = most_common_qt0[1]
        if qtable_1_counter:
            most_common_qt1 = qtable_1_counter.most_common(1)[0]
            summary['dominant_qtable_1'] = list(most_common_qt1[0])
            summary['dominant_qtable_1_count'] = most_common_qt1[1]

    # Print summary
    print(f"\n  File count: {summary['count']}")
    print(f"  Formats: {summary['formats']}")
    if 'file_size' in summary:
        fs = summary['file_size']
        print(f"  File size: min={fs['min']//1024}KB, median={int(fs['median'])//1024}KB, max={fs['max']//1024}KB, mean={int(fs['mean'])//1024}KB")
    if 'width' in summary:
        print(f"  Width: min={summary['width']['min']}, median={summary['width']['median']}, max={summary['width']['max']}")
        print(f"  Height: min={summary['height']['min']}, median={summary['height']['median']}, max={summary['height']['max']}")
        print(f"  Max dimension: min={summary['max_dim']['min']}, median={summary['max_dim']['median']}, max={summary['max_dim']['max']}")
    if 'bytes_per_pixel' in summary:
        bp = summary['bytes_per_pixel']
        print(f"  Bytes/pixel: min={bp['min']}, median={bp['median']}, max={bp['max']}, mean={bp['mean']}")
    if summary.get('est_quality'):
        eq = summary['est_quality']
        print(f"  Est. JPEG quality: min={eq['min']}, median={eq['median']}, max={eq['max']}")
    if 'subsampling' in summary:
        print(f"  Chroma subsampling: {summary['subsampling']}")
    if 'progressive' in summary:
        print(f"  Progressive: {summary['progressive']}")
    if 'exif_present' in summary:
        n = len(jpeg_files)
        print(f"  EXIF: {summary['exif_present']}/{n}, ICC: {summary['icc_present']}/{n}")
        print(f"  JFIF: {summary['jfif_present']}/{n}, Adobe: {summary['adobe_present']}/{n}")
    if 'unique_qtable_0_count' in summary:
        print(f"  Unique luminance Q-tables: {summary['unique_qtable_0_count']}")
        print(f"  Unique chrominance Q-tables: {summary['unique_qtable_1_count']}")
        if summary.get('dominant_qtable_0'):
            n = len(jpeg_files)
            print(f"  Dominant Q-table[0] ({summary['dominant_qtable_0_count']}/{n} files):")
            qt = summary['dominant_qtable_0']
            for row in range(8):
                print(f"    {qt[row*8:(row+1)*8]}")
        if summary.get('dominant_qtable_1'):
            print(f"  Dominant Q-table[1] ({summary['dominant_qtable_1_count']}/{n} files):")
            qt = summary['dominant_qtable_1']
            for row in range(8):
                print(f"    {qt[row*8:(row+1)*8]}")

    return {'summary': summary, 'all_info': all_info}


def main():
    results = {}

    for label, folder in PLATFORM_FOLDERS.items():
        result = analyze_folder(folder, label)
        if result:
            results[label] = result

    for label, folder in ORIGINAL_FOLDERS.items():
        result = analyze_folder(folder, label)
        if result:
            results[label] = result

    # Save raw data
    output_path = Path(r"D:\ml_project\audit_step2_results.json")
    json_results = {}
    for key, val in results.items():
        summary = val['summary'].copy()
        json_results[key] = {
            'summary': summary,
            'per_file': [{k: v for k, v in info.items()
                          if not isinstance(v, (bytes, memoryview))}
                         for info in val['all_info']]
        }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n\nResults saved to {output_path}")

    # Cross-platform comparison
    print(f"\n\n{'='*80}")
    print("CROSS-PLATFORM COMPARISON (Training data only)")
    print(f"{'='*80}")
    print(f"{'Platform':<20} {'Format':<8} {'MaxDim':<16} {'FileSize(KB)':<20} {'BPP':<16} {'Quality':<12} {'Subsamp':<10} {'Progr':<8}")
    print("-" * 110)
    for label in ['whatsapp_real', 'instagram_real', 'facebook_real', 'telegram_real']:
        if label in results:
            s = results[label]['summary']
            fmt = list(s['formats'].keys())[0] if s.get('formats') else '?'
            md = s.get('max_dim', {})
            fs = s.get('file_size', {})
            bp = s.get('bytes_per_pixel', {})
            eq = s.get('est_quality', {})
            sub = list(s.get('subsampling', {}).keys())
            prog = s.get('progressive', {})

            md_str = f"{md.get('min','?')}-{md.get('max','?')}"
            fs_str = f"{fs.get('min',0)//1024}-{fs.get('max',0)//1024}KB"
            bp_str = f"{bp.get('min','?')}-{bp.get('max','?')}"
            eq_str = f"{eq.get('min','?')}-{eq.get('max','?')}" if eq else 'N/A'
            sub_str = ','.join(sub) if sub else '?'
            prog_str = str(prog)

            print(f"{label:<20} {fmt:<8} {md_str:<16} {fs_str:<20} {bp_str:<16} {eq_str:<12} {sub_str:<10} {prog_str:<8}")


if __name__ == '__main__':
    main()
