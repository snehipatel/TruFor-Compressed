#!/usr/bin/env python3
"""
Unified Batch Inference Runner for TruFor.
Loads the model once onto GPU and runs inference across all datasets:
- Standard platform datasets (REAL, FAKE, Facebook, Instagram, Telegram, WhatsApp)
- Test platform datasets (real_test, fake_test across all platforms)
Saves npz artifacts, visualization PNGs, and summary CSVs.
"""

import os
import sys
import argparse
import csv
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, 'test_docker', 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import _C as config, update_config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

class SafeDataset(torch.utils.data.Dataset):
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

def classify_image(det_sig, pred_map, conf_map, threshold=0.12):
    if det_sig >= threshold:
        return ('TAMPERED / FAKE', f'global_score={det_sig:.4f} >= {threshold}')
    if conf_map is not None:
        weighted_map = pred_map * conf_map
    else:
        weighted_map = pred_map
    anomaly_mask = weighted_map > 0.5
    anomaly_area_ratio = anomaly_mask.mean()
    peak_anomaly = pred_map.max()
    if anomaly_area_ratio >= 0.01 and peak_anomaly >= 0.80:
        return ('TAMPERED / FAKE', f'localized_anomaly: area={anomaly_area_ratio*100:.2f}%, peak={peak_anomaly:.4f}')
    return ('PRISTINE / AUTHENTIC', f'score={det_sig:.4f}, below threshold')

def save_visualization(image_path, result_npz_path, output_png_path, status_override=None):
    result = np.load(result_npz_path)
    noisepr = result.get('np++', None)
    cols = 4 if noisepr is not None else 3
    fig, axs = plt.subplots(1, cols, figsize=(4.5 * cols, 4.5))
    score_val = float(result.get('score', 0.0))
    status_str = status_override if status_override is not None else ("TAMPERED / FAKE" if score_val >= 0.5 else "PRISTINE / AUTHENTIC")
    color = '#d32f2f' if 'TAMPERED' in status_str else '#2e7d32'
    fig.suptitle(f'Integrity Score: {score_val:.4f} ({status_str})', fontsize=14, fontweight='bold', color=color)
    for ax in axs:
        ax.axis('off')
    idx = 0
    orig_img = Image.open(image_path).convert('RGB')
    map_h, map_w = result['map'].shape
    if orig_img.size != (map_w, map_h):
        orig_img = orig_img.resize((map_w, map_h), Image.BILINEAR)
    axs[idx].imshow(orig_img)
    axs[idx].set_title('Original Image', fontsize=12)
    if noisepr is not None:
        idx += 1
        axs[idx].imshow(noisepr[16:-16:5, 16:-16:5], cmap='gray')
        axs[idx].set_title('Noiseprint++', fontsize=12)
    idx += 1
    axs[idx].imshow(result['map'], cmap='RdBu_r', clim=[0, 1])
    axs[idx].set_title('Anomaly Localization Map', fontsize=12)
    idx += 1
    if 'conf' in result:
        axs[idx].imshow(result['conf'], cmap='gray', clim=[0, 1])
        axs[idx].set_title('Confidence Map', fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def run_dataset_inference(model, device, input_dir, output_dirs, threshold=0.12, max_size=2048):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    list_img = [
        os.path.join(r, f)
        for r, _, files in os.walk(input_dir)
        for f in files
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    if not list_img:
        print(f"[-] No supported images found in '{input_dir}'.")
        return []

    print(f"\n[*] Processing '{input_dir}' ({len(list_img)} images) -> Threshold: {threshold}")
    test_dataset = SafeDataset(list_img=list_img, max_size=max_size)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    for out_d in output_dirs:
        os.makedirs(out_d, exist_ok=True)

    results_summary = []
    with torch.no_grad():
        for rgb, path in tqdm(testloader, desc=os.path.basename(input_dir)):
            img_path = os.path.abspath(path[0])
            rel_name = os.path.basename(img_path)
            stem = os.path.splitext(rel_name)[0]

            primary_npz = os.path.join(output_dirs[0], stem + '.npz')
            primary_png = os.path.join(output_dirs[0], stem + '_vis.png')

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
                    'conf': conf,
                    'np++': npp
                }
                np.savez(primary_npz, **out_dict)

                status, reason = classify_image(det_sig, pred, conf, threshold=threshold)
                save_visualization(img_path, primary_npz, primary_png, status_override=status)

                # Copy to secondary output dirs if specified
                for secondary_d in output_dirs[1:]:
                    sec_npz = os.path.join(secondary_d, stem + '.npz')
                    sec_png = os.path.join(secondary_d, stem + '_vis.png')
                    import shutil
                    shutil.copyfile(primary_npz, sec_npz)
                    shutil.copyfile(primary_png, sec_png)

                results_summary.append({
                    'name': rel_name,
                    'score': det_sig,
                    'status': status,
                    'reason': reason,
                    'npz': primary_npz,
                    'vis': primary_png
                })
            except Exception as e:
                print(f"[Error processing {img_path}]: {e}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save summary CSV to all output dirs
    for out_d in output_dirs:
        csv_path = os.path.join(out_d, 'summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'score', 'status', 'reason', 'npz', 'vis'])
            writer.writeheader()
            for r in results_summary:
                writer.writerow(r)

    avg_score = np.mean([r['score'] for r in results_summary]) if results_summary else 0
    tampered_cnt = sum(1 for r in results_summary if 'TAMPERED' in r['status'])
    pristine_cnt = sum(1 for r in results_summary if 'PRISTINE' in r['status'])
    print(f"    Avg Score: {avg_score:.4f} | Authentic: {pristine_cnt} | Tampered: {tampered_cnt}")
    return results_summary

def main():
    parser = argparse.ArgumentParser(description='Batch TruFor Inference Across All Datasets')
    parser.add_argument('--weights', type=str, default='pretrained_models/trufor.pth.tar')
    parser.add_argument('--gpu', type=int, default=0 if torch.cuda.is_available() else -1)
    parser.add_argument('--target', type=str, default='all', choices=['all', 'main', 'test'])
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Hardware setup
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"[*] Running on GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("[*] Running on CPU")

    weights_path = os.path.abspath(args.weights)
    if not os.path.isfile(weights_path):
        weights_path = os.path.abspath('test_docker/weights/trufor.pth.tar')

    # Load configuration
    update_config(config, args)

    print(f"[*] Loading model checkpoint from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = confcmx(cfg=config)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # Define tasks: (input_dir, [output_dirs], threshold, is_test)
    tasks = []

    # 1. Main datasets (25+ images each)
    main_tasks = [
        ('DATA/REAL', ['results/REAL'], 0.12),
        ('DATA/FAKE', ['results/FAKE'], 0.12),
        ('DATA/FACEBOOK/facebook_real', ['results/facebook_real', 'results/FACEBOOK/facebook_real'], 0.30),
        ('DATA/FACEBOOK/facebook_fake', ['results/facebook_fake', 'results/FACEBOOK/facebook_fake'], 0.30),
        ('DATA/INSTAGRAM/Instagram_Real', ['results/Instagram_Real', 'results/INSTAGRAM/Instagram_Real'], 0.30),
        ('DATA/INSTAGRAM/Instagram_Fake', ['results/Instagram_Fake', 'results/INSTAGRAM/Instagram_Fake'], 0.30),
        ('DATA/TELEGRAM/telegram_real', ['results/telegram_real', 'results/TELEGRAM/telegram_real'], 0.30),
        ('DATA/TELEGRAM/telegram_fake', ['results/telegram_fake', 'results/TELEGRAM/telegram_fake'], 0.30),
        ('DATA/WHATSAPP/whatsapp_real', ['results/whatsapp_real', 'results/WHATSAPP/whatsapp_real'], 0.30),
        ('DATA/WHATSAPP/whatsapp_fake', ['results/whatsapp_fake', 'results/WHATSAPP/whatsapp_fake'], 0.30),
    ]

    # 2. Test datasets (10 images each)
    test_tasks = [
        ('DATA/REAL/real_test', ['results/REAL/real_test', 'results/test/real_test'], 0.12),
        ('DATA/FAKE/fake_test', ['results/FAKE/fake_test', 'results/test/fake_test'], 0.12),
        ('DATA/FACEBOOK/facebook_test_real', ['results/FACEBOOK/facebook_test_real', 'results/test/facebook_test_real'], 0.30),
        ('DATA/FACEBOOK/facebook_test_fake', ['results/FACEBOOK/facebook_test_fake', 'results/test/facebook_test_fake'], 0.30),
        ('DATA/INSTAGRAM/Instagram_real_test', ['results/INSTAGRAM/Instagram_real_test', 'results/test/Instagram_real_test'], 0.30),
        ('DATA/INSTAGRAM/Instagram_fake_test', ['results/INSTAGRAM/Instagram_fake_test', 'results/test/Instagram_fake_test'], 0.30),
        ('DATA/TELEGRAM/telegram_real_test', ['results/TELEGRAM/telegram_real_test', 'results/test/telegram_real_test'], 0.30),
        ('DATA/TELEGRAM/telegram_fake_test', ['results/TELEGRAM/telegram_fake_test', 'results/test/telegram_fake_test'], 0.30),
        ('DATA/WHATSAPP/whatsapp_real_test', ['results/WHATSAPP/whatsapp_real_test', 'results/test/whatsapp_real_test'], 0.30),
        ('DATA/WHATSAPP/whatsapp_fake_test', ['results/WHATSAPP/whatsapp_fake_test', 'results/test/whatsapp_fake_test'], 0.30),
        # 3. Simulator test datasets (10 images each)
        ('SIMULATOR/output/simulated_facebook_real_test', ['results/simulated/facebook_real_test'], 0.30),
        ('SIMULATOR/output/simulated_facebook_fake_test', ['results/simulated/facebook_fake_test'], 0.30),
        ('SIMULATOR/output/simulated_instagram_real_test', ['results/simulated/instagram_real_test'], 0.30),
        ('SIMULATOR/output/simulated_instagram_fake_test', ['results/simulated/instagram_fake_test'], 0.30),
        ('SIMULATOR/output/simulated_telegram_real_test', ['results/simulated/telegram_real_test'], 0.30),
        ('SIMULATOR/output/simulated_telegram_fake_test', ['results/simulated/telegram_fake_test'], 0.30),
        ('SIMULATOR/output/simulated_whatsapp_real_test', ['results/simulated/whatsapp_real_test'], 0.30),
        ('SIMULATOR/output/simulated_whatsapp_fake_test', ['results/simulated/whatsapp_fake_test'], 0.30),
    ]

    if args.target in ['all', 'main']:
        tasks.extend(main_tasks)
    if args.target in ['all', 'test']:
        tasks.extend(test_tasks)

    overall_metrics = []
    for input_d, output_dirs, thresh in tasks:
        if not os.path.exists(input_d):
            print(f"[-] Directory {input_d} does not exist, skipping.")
            continue
        res = run_dataset_inference(model, device, input_d, output_dirs, threshold=thresh)
        if res:
            scores = [r['score'] for r in res]
            tampered = sum(1 for r in res if 'TAMPERED' in r['status'])
            pristine = sum(1 for r in res if 'PRISTINE' in r['status'])
            is_real = 'real' in os.path.basename(input_d).lower() or input_d == 'DATA/REAL' or 'real_test' in input_d.lower()
            overall_metrics.append({
                'dataset': input_d,
                'total': len(res),
                'ground_truth': 'Authentic' if is_real else 'Tampered',
                'mean_score': float(np.mean(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'std_score': float(np.std(scores)),
                'authentic_pred': pristine,
                'tampered_pred': tampered,
                'accuracy': (pristine / len(res) * 100) if is_real else (tampered / len(res) * 100)
            })

    # Save overall comprehensive summary CSV
    overall_csv = os.path.join('results', f'evaluation_metrics_{args.target}.csv')
    with open(overall_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'total', 'ground_truth', 'mean_score', 'min_score', 'max_score', 'std_score',
            'authentic_pred', 'tampered_pred', 'accuracy'
        ])
        writer.writeheader()
        for m in overall_metrics:
            writer.writerow(m)

    print("\n" + "=" * 90)
    print(f"{'Dataset':<35} | {'Type':<9} | {'Count':<5} | {'Mean Score':<10} | {'Accuracy':<8}")
    print("=" * 90)
    for m in overall_metrics:
        print(f"{m['dataset']:<35} | {m['ground_truth']:<9} | {m['total']:<5} | {m['mean_score']:<10.4f} | {m['accuracy']:>6.1f}%")
    print("=" * 90)
    print(f"\n[+] Master evaluation metrics saved to: {overall_csv}\n")

if __name__ == '__main__':
    main()
