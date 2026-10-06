#!/usr/bin/env python3
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# TruFor - Trustworthy Image Forgery Detection & Localization Runner
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import sys
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for reliable figure rendering
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

# Ensure path to model components
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'test_docker', 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import _C as config, update_config
from data_core import myDataset
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx


class SafeDataset(torch.utils.data.Dataset):
    """Memory-safe dataset that caps extreme resolutions to prevent CUDA OOM."""
    def __init__(self, list_img=None, max_size=2048):
        self.img_list = list_img
        self.max_size = max_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        rgb_path = self.img_list[index]
        pil_img = Image.open(rgb_path).convert("RGB")
        w, h = pil_img.size
        if self.max_size is not None and max(w, h) > self.max_size:
            scale = float(self.max_size) / float(max(w, h))
            new_w = max(8, (int(round(w * scale)) // 8) * 8)
            new_h = max(8, (int(round(h * scale)) // 8) * 8)
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w = max(8, (w // 8) * 8)
            new_h = max(8, (h // 8) * 8)
            if (new_w, new_h) != (w, h):
                pil_img = pil_img.crop((0, 0, new_w, new_h))
        img_np = np.array(pil_img)
        return torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path


def classify_image(det_sig, pred_map, conf_map, threshold=0.15):
    """
    Dual-criteria forgery classification.

    An image is classified as TAMPERED if EITHER:
      1. The global detection score (det_sig) >= threshold, OR
      2. The localization map reveals a coherent tampered region:
         - Confidence-weighted anomaly area (map > 0.5 AND conf > 0.5) covers >= 1% of pixels, AND
         - Peak anomaly value in the map exceeds 0.80.

    Returns:
        (status_str, reason_str)
    """
    # Criterion 1: Global detection score
    if det_sig >= threshold:
        return ('TAMPERED / FAKE', f'global_score={det_sig:.4f} >= {threshold}')

    # Criterion 2: Localized anomaly region
    if conf_map is not None:
        weighted_map = pred_map * conf_map
    else:
        weighted_map = pred_map

    anomaly_mask = weighted_map > 0.5
    anomaly_area_ratio = anomaly_mask.mean()
    peak_anomaly = pred_map.max()

    if anomaly_area_ratio >= 0.01 and peak_anomaly >= 0.80:
        return ('TAMPERED / FAKE',
                f'localized_anomaly: area={anomaly_area_ratio*100:.2f}%, peak={peak_anomaly:.4f}')

    return ('PRISTINE / AUTHENTIC', f'score={det_sig:.4f}, below threshold')


def save_visualization(image_path, result_npz_path, output_png_path, status_override=None):
    result = np.load(result_npz_path)
    noisepr = result.get('np++', None)
    cols = 4 if noisepr is not None else 3

    fig, axs = plt.subplots(1, cols, figsize=(4.5 * cols, 4.5))
    score_val = float(result.get('score', 0.0))
    if status_override is not None:
        status_str = status_override
    else:
        status_str = "TAMPERED / FAKE" if score_val >= 0.5 else "PRISTINE / AUTHENTIC"
    color = '#d32f2f' if 'TAMPERED' in status_str else '#2e7d32'
    fig.suptitle(f'Integrity Score: {score_val:.4f} ({status_str})', fontsize=14, fontweight='bold', color=color)

    for ax in axs:
        ax.axis('off')

    idx = 0
    # 1. Original Image
    orig_img = Image.open(image_path).convert('RGB')
    map_h, map_w = result['map'].shape
    if orig_img.size != (map_w, map_h):
        orig_img = orig_img.resize((map_w, map_h), Image.BILINEAR)
    axs[idx].imshow(orig_img)
    axs[idx].set_title('Original Image', fontsize=12)

    # 2. Noiseprint++
    if noisepr is not None:
        idx += 1
        axs[idx].imshow(noisepr[16:-16:5, 16:-16:5], cmap='gray')
        axs[idx].set_title('Noiseprint++', fontsize=12)

    # 3. Localization Map
    idx += 1
    axs[idx].imshow(result['map'], cmap='RdBu_r', clim=[0, 1])
    axs[idx].set_title('Anomaly Localization Map', fontsize=12)

    # 4. Confidence Map
    idx += 1
    if 'conf' in result:
        axs[idx].imshow(result['conf'], cmap='gray', clim=[0, 1])
        axs[idx].set_title('Confidence Map', fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    default_device = 0 if torch.cuda.is_available() else -1
    default_weights = os.path.join(SCRIPT_DIR, 'pretrained_models', 'trufor.pth.tar')

    parser = argparse.ArgumentParser(description='Run TruFor Image Forgery Detection & Localization')
    parser.add_argument('-i', '--input', type=str, default='test_docker/images',
                        help='Input image file, directory, or glob pattern (e.g. "*.jpg")')
    parser.add_argument('-o', '--output', type=str, default='results',
                        help='Output directory to save predictions')
    parser.add_argument('-g', '--gpu', type=int, default=default_device,
                        help='GPU ID (e.g. 0), or -1 for CPU')
    parser.add_argument('-w', '--weights', type=str, default=default_weights,
                        help='Path to trufor.pth.tar weights file')
    parser.add_argument('--vis', action='store_true', default=True,
                        help='Generate side-by-side visual comparison PNG images (default: True)')
    parser.add_argument('--no-vis', dest='vis', action='store_false',
                        help='Disable visualization PNG generation')
    parser.add_argument('--save_np', action='store_true', default=True,
                        help='Save Noiseprint++ map in the output .npz (default: True)')
    parser.add_argument('--max_size', type=int, default=2048,
                        help='Max image dimension for inference to prevent CUDA OOM (default: 2048, 0 for full size)')
    parser.add_argument('--threshold', type=float, default=0.12,
                        help='Detection threshold for global score (default: 0.12, calibrated for inpainting)')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Additional config overrides')

    args = parser.parse_args()

    # Verify weights
    if not os.path.isfile(args.weights):
        alt_weights = os.path.join(SCRIPT_DIR, 'test_docker', 'weights', 'trufor.pth.tar')
        if os.path.isfile(alt_weights):
            args.weights = alt_weights
        else:
            print(f"[Error] Weights file not found at: {args.weights}")
            sys.exit(1)

    # Resolve device
    if args.gpu >= 0 and torch.cuda.is_available():
        device_name = f'cuda:{args.gpu}'
        gpu_name = torch.cuda.get_device_name(args.gpu)
        print(f"[*] Running on GPU ({gpu_name})")
    else:
        device_name = 'cpu'
        print("[*] Running on CPU")

    device = torch.device(device_name)

    # Collect images
    input_arg = args.input
    if '*' in input_arg:
        list_img = glob(input_arg, recursive=True)
        list_img = [img for img in list_img if os.path.isfile(img)]
    elif os.path.isfile(input_arg):
        list_img = [input_arg]
    elif os.path.isdir(input_arg):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        list_img = [
            os.path.join(r, f)
            for r, _, files in os.walk(input_arg)
            for f in files
            if os.path.splitext(f)[1].lower() in valid_exts
        ]
    else:
        print(f"[Error] Input path '{input_arg}' does not exist.")
        sys.exit(1)

    if not list_img:
        print(f"[Warning] No supported images found in '{input_arg}'.")
        return

    print(f"[*] Found {len(list_img)} image(s) to process.")
    print(f"[*] Loading model from {args.weights} ...")

    # Load configuration
    update_config(config, args)

    # Load model
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = confcmx(cfg=config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    max_sz = args.max_size if args.max_size > 0 else None
    test_dataset = SafeDataset(list_img=list_img, max_size=max_sz)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    os.makedirs(args.output, exist_ok=True)
    results_summary = []

    print(f"[*] Running inference (saving to '{args.output}')...\n")

    with torch.no_grad():
        for index, (rgb, path) in enumerate(tqdm(testloader, desc="Processing")):
            img_path = os.path.abspath(path[0])
            in_root = os.path.abspath(input_arg.split('*')[0])
            base_dir = in_root if os.path.isdir(in_root) else os.path.dirname(in_root)
            rel_path = os.path.relpath(img_path, base_dir)
            stem = os.path.splitext(rel_path)[0]

            out_npz = os.path.join(args.output, stem + '.npz')
            out_png = os.path.join(args.output, stem + '_vis.png')
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)

            try:
                rgb = rgb.to(device)
                pred, conf, det, npp = model(rgb)

                if conf is not None:
                    conf = torch.squeeze(conf, 0)
                    conf = torch.sigmoid(conf)[0].cpu().numpy()

                if npp is not None:
                    npp = torch.squeeze(npp, 0)[0].cpu().numpy()

                det_sig = 0.0
                if det is not None:
                    det_sig = torch.sigmoid(det).item()

                pred = torch.squeeze(pred, 0)
                pred = F.softmax(pred, dim=0)[1].cpu().numpy()

                out_dict = {
                    'map': pred,
                    'imgsize': tuple(rgb.shape[2:]),
                    'score': det_sig,
                    'conf': conf
                }
                if args.save_np and npp is not None:
                    out_dict['np++'] = npp

                np.savez(out_npz, **out_dict)

                # Dual-criteria classification
                status, reason = classify_image(det_sig, pred, conf, threshold=args.threshold)

                if args.vis:
                    save_visualization(img_path, out_npz, out_png, status_override=status)

                results_summary.append({
                    'name': os.path.basename(img_path),
                    'score': det_sig,
                    'status': status,
                    'reason': reason,
                    'npz': out_npz,
                    'vis': out_png if args.vis else None
                })
            except Exception as e:
                print(f"\n[Error processing {img_path}]: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Image':<25} | {'Score':<10} | {'Classification':<22} | {'Result Artifacts'}")
    print("=" * 80)
    for res in results_summary:
        vis_note = f" (Vis: {os.path.basename(res['vis'])})" if res['vis'] else ""
        print(f"{res['name']:<25} | {res['score']:<10.4f} | {res['status']:<22} | {os.path.basename(res['npz'])}{vis_note}")
    print("=" * 80)
    # Save CSV summary
    csv_path = os.path.join(args.output, 'summary.csv')
    try:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'score', 'status', 'reason', 'npz', 'vis'])
            writer.writeheader()
            for r in results_summary:
                writer.writerow(r)
        print(f"[+] Summary saved to CSV: {csv_path}")
    except Exception as e:
        print(f"[!] Could not save CSV summary: {e}")

    print(f"[+] Inference complete. All outputs saved to: {os.path.abspath(args.output)}\n")


if __name__ == '__main__':
    main()
