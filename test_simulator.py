#!/usr/bin/env python3
"""
Test Suite & Benchmark for Platform Compression & Forensic Simulator.
1. Tests Simulator Engine: applies WhatsApp, Instagram, Facebook, and Telegram compression
   profiles to real_test and fake_test images.
2. Tests Simulator Flask API: checks /api/simulate, /api/batch, /api/demo endpoints.
3. Benchmarks Simulator Output against Real Platform-compressed test images.
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Add SIMULATOR directory to path
SIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SIMULATOR')
if SIM_DIR not in sys.path:
    sys.path.insert(0, SIM_DIR)

import server
from server import PLATFORM_PROFILES, prepare_image_for_processing, compress_image, compute_difference_map, rate_visual_difference

def test_simulator_api():
    print("\n" + "=" * 80)
    print("  TEST 1: SIMULATOR FLASK API ENDPOINT VERIFICATION")
    print("=" * 80)
    client = server.app.test_client()

    # 1. Test /api/demo
    print("[*] Testing GET /api/demo ...")
    resp = client.get('/api/demo?platform=whatsapp&quality_mode=standard')
    assert resp.status_code == 200, f"Demo failed with status {resp.status_code}"
    data = resp.get_json()
    assert data['success'] is True
    print(f"    [PASS] /api/demo -> Success! Output: {data['compressed']['filename']}, size: {data['compressed']['size']:,} bytes")

    # 2. Test /api/simulate with real_test sample
    print("[*] Testing POST /api/simulate ...")
    sample_path = os.path.join('DATA', 'REAL', 'real_test', 'r0225b88bt.png')
    with open(sample_path, 'rb') as f:
        resp = client.post('/api/simulate', data={
            'image': (f, 'r0225b88bt.png'),
            'platform': 'instagram',
            'quality_mode': 'standard'
        }, content_type='multipart/form-data')
    assert resp.status_code == 200, f"Simulate failed: {resp.status_code}"
    sim_data = resp.get_json()
    assert sim_data['success'] is True
    print(f"    [PASS] /api/simulate -> Instagram WebP output: {sim_data['compressed']['filename']}")
    print(f"           Original: {sim_data['original']['size']:,} B -> Compressed: {sim_data['compressed']['size']:,} B ({sim_data['stats']['reduction_pct']} % reduction)")

    # 3. Test /api/batch
    print("[*] Testing POST /api/batch ...")
    batch_out = os.path.join('SIMULATOR', 'output', 'api_batch_test')
    resp = client.post('/api/batch', json={
        'input_folder': os.path.abspath('DATA/FAKE/fake_test'),
        'output_folder': os.path.abspath(batch_out),
        'platform': 'whatsapp',
        'quality_mode': 'standard'
    })
    assert resp.status_code == 200, f"Batch failed: {resp.status_code}"
    bdata = resp.get_json()
    assert bdata['success'] is True
    print(f"    [PASS] /api/batch -> Processed {bdata['total_images']} images successfully!")
    print("=" * 80)
    print("  ALL API TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")

def run_simulator_on_test_sets():
    print("=" * 80)
    print("  TEST 2: SIMULATING PLATFORM COMPRESSION ON REAL_TEST & FAKE_TEST")
    print("=" * 80)

    platforms = ['whatsapp', 'instagram', 'facebook', 'telegram']
    input_sets = [
        ('real_test', os.path.join('DATA', 'REAL', 'real_test')),
        ('fake_test', os.path.join('DATA', 'FAKE', 'fake_test')),
    ]

    all_sim_results = {}

    for set_name, in_dir in input_sets:
        all_sim_results[set_name] = {}
        images = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        print(f"\n[*] Processing {set_name} ({len(images)} images) from {in_dir}:")

        for plat in platforms:
            profile = PLATFORM_PROFILES[plat]['standard']
            out_dir = os.path.join('SIMULATOR', 'output', f'simulated_{plat}_{set_name}')
            os.makedirs(out_dir, exist_ok=True)

            plat_results = []
            total_in = 0
            total_out = 0

            for fname in images:
                fpath = os.path.join(in_dir, fname)
                orig_size = os.path.getsize(fpath)
                total_in += orig_size

                raw_im = Image.open(fpath)
                prep_im = prepare_image_for_processing(raw_im)
                comp_bytes, new_w, new_h, codec = compress_image(prep_im, profile)
                comp_size = len(comp_bytes)
                total_out += comp_size

                ext = '.jpg' if codec == 'JPEG' else '.webp'
                stem = os.path.splitext(fname)[0]
                out_name = f"{stem}{ext}"
                out_path = os.path.join(out_dir, out_name)

                with open(out_path, 'wb') as f:
                    f.write(comp_bytes)

                plat_results.append({
                    'original_name': fname,
                    'compressed_name': out_name,
                    'orig_size': orig_size,
                    'comp_size': comp_size,
                    'orig_res': f"{raw_im.size[0]}x{raw_im.size[1]}",
                    'comp_res': f"{new_w}x{new_h}",
                    'reduction_pct': round((orig_size - comp_size) / orig_size * 100, 1),
                    'out_path': out_path
                })

            # Save manifest
            manifest_path = os.path.join(out_dir, 'manifest.csv')
            with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'original_name', 'compressed_name', 'orig_size', 'comp_size',
                    'orig_res', 'comp_res', 'reduction_pct', 'out_path'
                ])
                writer.writeheader()
                writer.writerows(plat_results)

            overall_red = (total_in - total_out) / total_in * 100 if total_in > 0 else 0
            avg_comp_size = total_out / len(images) if images else 0
            print(f"    -> {plat.upper():<10}: {len(plat_results)} images | Avg size: {avg_comp_size:,.0f} B | Overall reduction: {overall_red:.1f}% | Out: {out_dir}")
            all_sim_results[set_name][plat] = {
                'output_dir': out_dir,
                'total_in': total_in,
                'total_out': total_out,
                'overall_reduction': overall_red,
                'results': plat_results
            }

    return all_sim_results

def benchmark_against_real_platforms():
    print("\n" + "=" * 80)
    print("  TEST 3: BENCHMARKING SIMULATOR FIDELITY VS REAL PLATFORM TRANSMISSIONS")
    print("=" * 80)

    comparison_mappings = [
        ('Facebook Real', 'SIMULATOR/output/simulated_facebook_real_test', 'DATA/FACEBOOK/facebook_test_real'),
        ('Facebook Fake', 'SIMULATOR/output/simulated_facebook_fake_test', 'DATA/FACEBOOK/facebook_test_fake'),
        ('Instagram Real', 'SIMULATOR/output/simulated_instagram_real_test', 'DATA/INSTAGRAM/Instagram_real_test'),
        ('Instagram Fake', 'SIMULATOR/output/simulated_instagram_fake_test', 'DATA/INSTAGRAM/Instagram_fake_test'),
        ('Telegram Real', 'SIMULATOR/output/simulated_telegram_real_test', 'DATA/TELEGRAM/telegram_real_test'),
        ('Telegram Fake', 'SIMULATOR/output/simulated_telegram_fake_test', 'DATA/TELEGRAM/telegram_fake_test'),
        ('WhatsApp Real', 'SIMULATOR/output/simulated_whatsapp_real_test', 'DATA/WHATSAPP/whatsapp_real_test'),
        ('WhatsApp Fake', 'SIMULATOR/output/simulated_whatsapp_fake_test', 'DATA/WHATSAPP/whatsapp_fake_test'),
    ]

    benchmark_rows = []
    for label, sim_dir, real_dir in comparison_mappings:
        sim_files = [f for f in os.listdir(sim_dir) if not f.endswith('.csv')]
        real_files = [f for f in os.listdir(real_dir) if not f.startswith('.')]

        sim_sizes = [os.path.getsize(os.path.join(sim_dir, f)) for f in sim_files]
        real_sizes = [os.path.getsize(os.path.join(real_dir, f)) for f in real_files]

        sim_res = []
        for f in sim_files[:3]:
            im = Image.open(os.path.join(sim_dir, f))
            sim_res.append(f"{im.size[0]}x{im.size[1]}")

        real_res = []
        for f in real_files[:3]:
            im = Image.open(os.path.join(real_dir, f))
            real_res.append(f"{im.size[0]}x{im.size[1]}")

        avg_sim_kb = np.mean(sim_sizes) / 1024 if sim_sizes else 0
        avg_real_kb = np.mean(real_sizes) / 1024 if real_sizes else 0
        size_ratio = avg_sim_kb / avg_real_kb if avg_real_kb > 0 else 0

        row = {
            'platform_dataset': label,
            'sim_count': len(sim_files),
            'real_count': len(real_files),
            'sim_avg_kb': round(avg_sim_kb, 1),
            'real_avg_kb': round(avg_real_kb, 1),
            'size_fidelity_ratio': round(size_ratio, 2),
            'sim_resolutions': ", ".join(set(sim_res)),
            'real_resolutions': ", ".join(set(real_res)),
        }
        benchmark_rows.append(row)

    print(f"{'Platform Dataset':<18} | {'Sim Avg (KB)':<12} | {'Real Avg (KB)':<13} | {'Size Fidelity':<13} | {'Real Resolutions'}")
    print("-" * 80)
    for r in benchmark_rows:
        print(f"{r['platform_dataset']:<18} | {r['sim_avg_kb']:>10.1f} KB | {r['real_avg_kb']:>11.1f} KB | {r['size_fidelity_ratio']:>11.2f}x | {r['real_resolutions']}")
    print("-" * 80)

    # Save benchmark CSV
    bench_csv = os.path.join('SIMULATOR', 'output', 'simulator_fidelity_benchmark.csv')
    with open(bench_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'platform_dataset', 'sim_count', 'real_count', 'sim_avg_kb', 'real_avg_kb',
            'size_fidelity_ratio', 'sim_resolutions', 'real_resolutions'
        ])
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"\n[+] Simulator fidelity benchmark saved to: {bench_csv}")

def main():
    test_simulator_api()
    run_simulator_on_test_sets()
    benchmark_against_real_platforms()
    print("\n[+] SIMULATOR SUITE COMPLETED SUCCESSFULLY!\n")

if __name__ == '__main__':
    main()
