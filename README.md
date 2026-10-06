# TruFor-Compressed: Image Forgery Detection & Forensic Analysis under Platform Compression

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange.svg)](https://git-lfs.github.com/)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-lightgrey.svg)](LICENSE.txt)
[![Official TruFor Paper](https://img.shields.io/badge/CVPR%202023-Paper-red.svg)](https://doi.org/10.48550/arXiv.2212.10957)

Official extension of **TruFor** (*"TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization"*, Guillaro et al., CVPR 2023) focused on **social media compression robustness**, empirical channel transcoding simulation, and cross-platform forensic benchmarking.

---

## Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [What Was Excluded From Git & Why](#what-was-excluded-from-git--why)
3. [Git LFS Configuration](#git-lfs-configuration)
4. [How It Works: Architecture & Pipeline](#how-it-works-architecture--pipeline)
5. [Interactive Platform Compression Simulator](#interactive-platform-compression-simulator)
6. [Repository Structure](#repository-structure)
7. [Installation & Setup](#installation--setup)
8. [Usage Workflows](#usage-workflows)
9. [Empirical Evaluation & Benchmark Findings](#empirical-evaluation--benchmark-findings)
10. [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview & Motivation

State-of-the-art forensic models like **TruFor** combine high-level RGB semantic feature extraction with low-level sensor noise analysis (**Noiseprint++**) via a Cross-Modal Transformer (CMX) decoder. While TruFor excels on pristine camera images, modern social media and messaging platforms (such as **WhatsApp, Instagram, Telegram, and Facebook**) aggressively transcode uploaded media. 

These platforms apply:
- **Lossy JPEG & WebP Requantization**: Destroys high-frequency residual noise patterns (PRNU).
- **Chroma Subsampling (4:2:0)**: Discards subtle color interpolation traces.
- **Dynamic Downscaling**: Dispatches images into fixed spatial envelopes (e.g., max 1080px, 1280px, 1600px).
- **EXIF Metadata Stripping**: Erases camera model and provenance tags.

**TruFor-Compressed** addresses this domain shift by providing:
1. **Reverse-Engineered Platform Quantization Profiles**: Exact empirical quantization matrices ($Q_0, Q_1$) and subsampling parameters matching actual production pipelines of WhatsApp, Instagram, Telegram, and Facebook.
2. **Interactive Platform Compression & Forensic Simulator**: A full-stack web application (`SIMULATOR/`) allowing real-time forensic comparison, compression fidelity benchmarking, and visual difference heatmaps.
3. **Comprehensive Forensic Benchmark Suite**: Evaluates detection accuracy, localization maps, and reliability scores across pristine vs. compressed transmission channels.
4. **Memory-Safe GPU Inference**: Dynamic image dimension clamping (`SafeDataset`) ensuring stable inference on consumer GPUs (e.g., 8 GB VRAM).

---

## What Was Excluded From Git & Why

To keep the repository fast, clean, and strictly within GitHub's file and storage limits, specific large, generated, or transient assets were omitted from Git tracking via `.gitignore`. 

Below is the complete accounting of all excluded items, the reason for exclusion, and how to acquire or reproduce them:

| Item / Directory | Disk Footprint | Reason for Exclusion | How to Obtain / Reproduce |
| :--- | :--- | :--- | :--- |
| **`.venv/`** | ~4.5 GB | Python virtual environment contains machine-specific precompiled binaries and libraries. Committing virtual environments violates Git best practices. | Run `python -m venv .venv` and install dependencies via `pip install -r SIMULATOR/requirements.txt` and `trufor_conda.yaml`. |
| **`results/*.npz` & `results/**/*.png`** | ~11.1 GB | Contains 899 per-pixel floating-point localization arrays (`.npz`) and high-resolution anomaly heatmaps (`.png`) generated during batch testing. Exceeds standard Git limits. | **All 47 benchmark CSV summaries** and [`results/evaluation_metrics_all.csv`](results/evaluation_metrics_all.csv) **ARE tracked in Git**. You can regenerate the full arrays anytime by running `python batch_inference_all.py`. |
| **`DATA/REAL/`** | ~717 MB | 35 raw, uncompressed 4K camera photos from the compRAISE dataset (~20–30 MB per PNG). Keeping raw photos out ensures fast clones. | Download compRAISE pristine photos from the [CAT-Net repository](https://github.com/mjkwon2021/CAT-Net) or place custom pristine camera images in `DATA/REAL/`. |
| **`DATA/whatsapp_real_*`** | ~306 MB | Preprocessed derivative JPEG sets generated from `DATA/REAL/`. | Easily regenerated in seconds by executing: `python compress_whatsapp.py --input DATA/REAL --output DATA/whatsapp_real_losslessly_compressed`. |
| **`output/`, `output_test/`, `output_user/`, `demo_results/`** | ~450 MB | Local test run directories and temporary scratch visualizations. | Auto-generated during manual testing runs of `run.py` and `test_simulator.py`. |
| **`SIMULATOR/uploads/*` & `SIMULATOR/output/*`** | ~250 MB | Transient uploaded files and generated preview artifacts created while interacting with the Simulator web app. | Folders are auto-created at runtime. Empty directory anchors (`.gitkeep`) are committed to ensure proper directory structure. |
| **`TruFor_weights.zip`** | ~260 MB | Redundant zip archive of TruFor model weights. Exceeds GitHub's 100 MB hard limit. | Unnecessary: the unzipped weights are already tracked and uploaded directly using **Git LFS**! |

> [!NOTE]
> **What IS included in `DATA/`**:  
> All lightweight platform test sets (~44 MB total) for **Facebook**, **Instagram**, **Telegram**, **WhatsApp**, and **Inpainting Fakes** are committed and available out-of-the-box for instant testing!

---

## Git LFS Configuration

Deep learning checkpoint files exceed GitHub's 100 MB standard push ceiling. This repository uses **Git LFS (Large File Storage)** to version the pre-trained weights without bloating the Git commit tree:

- `pretrained_models/trufor.pth.tar` (281.5 MB)
- `test_docker/weights/trufor.pth.tar` (281.5 MB)
- `TruFor_train_test/pretrained_models/trufor.pth.tar` (281.5 MB)
- `TruFor_train_test/pretrained_models/segformers/mit_b2.pth` (98.9 MB)
- `TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th` (2.3 MB)

*(Note: Identical `trufor.pth.tar` files share the same SHA-256 hash in Git LFS, consuming only ~383 MB of total remote LFS storage).*

### Cloning with Git LFS
To clone this repository and automatically fetch the full binary weights:
```bash
git lfs install
git clone https://github.com/snehipatel/TruFor-Compressed.git
cd TruFor-Compressed
git lfs pull
```

---

## How It Works: Architecture & Pipeline

```
                                    +-----------------------+
                             +----->| SegFormer-B2 (RGB)    |------+
                             |      +-----------------------+      |
                             |                                     v
+------------------+         |                               +------------+      +---------------------------+
| Input Image      |---------+                               |    CMX     |----->| Anomaly Localization Map  |
| (RGB / WxH)      |---------+                               |  Decoder   |----->| Confidence Map            |
+------------------+         |                               +------------+      +---------------------------+
                             |      +-----------------------+      ^                           |
                             +----->| Noiseprint++ (PRNU)   |------+                           v
                                    +-----------------------+                    +---------------------------+
                                                                                 | Global Integrity Score    |
                                                                                 +---------------------------+
```

### 1. Dual-Branch Feature Extraction
- **RGB Branch (High-Level Semantic Analysis)**: Built upon a `SegFormer-B2` backbone pretrained on ImageNet. Processes RGB pixel content to detect unnatural boundaries, chromatic inconsistencies, and contextual tampering.
- **Noiseprint++ Branch (Low-Level Sensor Noise Analysis)**: A deep Siamese denoiser trained in a self-supervised fashion on authentic camera images. Extracts the camera's Photo-Response Non-Uniformity (PRNU) fingerprint. When an object is spliced, pasted, or AI-generated, the PRNU pattern is disrupted.

### 2. Cross-Modal Transformer (CMX) Decoder
Combines RGB and Noiseprint++ features using cross-modal attention, generating three outputs:
1. **Anomaly Localization Map ($M \in [0, 1]^{H \times W}$)**: Per-pixel probability that a given location has been manipulated.
2. **Confidence Map ($C \in [0, 1]^{H \times W}$)**: Per-pixel reliability score indicating whether the forensic prediction is trustworthy or prone to false alarms (e.g., in low-texture dark zones).
3. **Global Integrity Score ($s \in [0, 1]$)**: A scalar whole-image forgery score derived from confidence-weighted pooling. A score close to `0.0` denotes an authentic image, while values approaching `1.0` indicate high confidence of forgery.

### 3. Impact of Social Media Transcoding
When an image is uploaded to WhatsApp or Facebook, DCT frequency quantization dampens high-frequency residuals. Consequently, the Noiseprint++ response is attenuated. TruFor-Compressed addresses this by providing platform-calibrated detection thresholds (e.g., dual-criteria detection with $0.12$ calibrated threshold + spatial connected component verification) rather than relying on a naive static $0.50$ threshold.

---

## Interactive Platform Compression Simulator

Located in the [`SIMULATOR/`](SIMULATOR/) folder, this tool is an interactive testbed for forensic analysis:

- **Empirical Platform Profiles**:
  - **WhatsApp Standard**: Max 1600px dimension, empirical WhatsApp $Q_0/Q_1$ tables, 4:2:0 subsampling, ~185–220 KB target.
  - **WhatsApp HD**: Near-lossless high-resolution profile ($Q=98$, 4:4:4 chroma preservation).
  - **Instagram Standard**: 1440×1440 square constraint, WebP transcoding ($Q=82$), aggressive downsampling.
  - **Telegram Standard**: 1280px max constraint, custom Telegram quantization tables ($Q \approx 78$).
  - **Facebook Standard**: 1080px max constraint, empirical Facebook double-quantization matrix.
- **Visual Analytics**:
  - Interactive split-screen comparison (Original vs. Compressed).
  - Amplified RGB Difference Heatmaps ($10\times$ gain) to expose quantization grid artifacts.
  - PSNR (dB), SSIM, and Compression Ratio analytics.
- **Batch Processing**:
  - Bulk compression of entire image folders with downloadable ZIP archives and CSV manifests.

---

## Repository Structure

```
TruFor-Compressed/
├── .gitattributes                  # Git LFS tracking rules (*.pth.tar, *.pth, *.th)
├── .gitignore                      # Exclusion rules (.venv, 11GB npz, raw photos, outputs)
├── README.md                       # Comprehensive project documentation
├── TruFor_Evaluation_Report.txt    # 1,500+ line scientific evaluation report
├── batch_inference_all.py          # Unified multi-dataset batch inference runner
├── compress_whatsapp.py            # WhatsApp near-lossless JPEG preprocessor (Q=98, 4:4:4)
├── run.py                          # Single-image or folder inference script
├── test_simulator.py               # Benchmark test suite for the simulator engine & API
│
├── SIMULATOR/                      # Platform Compression Simulator Web App
│   ├── index.html                  # Responsive frontend dashboard
│   ├── server.py                   # Flask server with empirical quantization tables
│   ├── script.js                   # Client-side UI & difference visualizer logic
│   ├── styles.css                  # Dark-mode forensic styling
│   ├── requirements.txt            # Lightweight simulator dependencies
│   ├── mountains.jpg               # Standard demo benchmark sample
│   └── simulator_fidelity_benchmark.csv # Ground-truth fidelity metrics
│
├── DATA/                           # Lightweight test datasets (~44 MB)
│   ├── FACEBOOK/                   # Facebook real and tampered sample sets
│   ├── INSTAGRAM/                  # Instagram real and tampered WebP sample sets
│   ├── TELEGRAM/                   # Telegram real and tampered JPEG sample sets
│   ├── WHATSAPP/                   # WhatsApp real and tampered sample sets
│   └── FAKE/                       # Inpainted forgery sample sets
│
├── results/                        # Evaluation Benchmark Results & Manifests
│   ├── evaluation_metrics_all.csv  # Master metric summary across all channels
│   └── [FACEBOOK, INSTAGRAM, ...]  # 47 platform-specific summary.csv files
│
├── pretrained_models/              # Git LFS tracked weights
│   └── trufor.pth.tar              # TruFor official checkpoint (281 MB via LFS)
│
├── TruFor_train_test/              # Training & Test Library (Official Architecture)
│   ├── dataset/                    # Dataset loaders and training lists
│   ├── lib/                        # Backbone models and configs
│   ├── pretrained_models/          # SegFormer and Noiseprint++ weights (LFS)
│   ├── test.py                     # Official evaluation script
│   ├── train.py                    # Multi-stage training pipeline
│   └── trufor_conda.yaml           # Conda environment definition
│
└── test_docker/                    # Docker Inference Environment
    ├── Dockerfile                  # Self-contained container definition
    ├── docker_build.sh             # Build script
    ├── docker_run.sh               # Execution script
    ├── src/                        # Inference source files
    └── weights/                    # trufor.pth.tar checkpoint (LFS)
```

---

## Installation & Setup

### 1. Prerequisites
- **OS**: Windows, Linux, or macOS.
- **Python**: 3.10 or higher.
- **CUDA**: 11.8+ or 12.x (Recommended: GPU with $\ge 8$ GB VRAM).
- **Git LFS**: Installed on your system (`git-lfs.github.com`).

### 2. Environment Setup
Create a virtual environment and install core dependencies:

```bash
# 1. Clone repository with LFS
git clone https://github.com/snehipatel/TruFor-Compressed.git
cd TruFor-Compressed
git lfs pull

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install PyTorch with CUDA support (example for CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install additional forensic & web dependencies
pip install -r SIMULATOR/requirements.txt
pip install tqdm yacs>=0.1.8 timm>=0.5.4 numpy matplotlib scikit-learn
```

---

## Usage Workflows

### Workflow 1: Launch the Platform Compression Simulator
To run the interactive simulator dashboard locally:
```bash
cd SIMULATOR
python server.py
```
Open your browser and navigate to:  
👉 **`http://localhost:5000`**

### Workflow 2: Run Forensic Inference on a Single Image
Run TruFor on any image to produce an anomaly heatmap, confidence map, and detection score:
```bash
python run.py --input path/to/image.jpg --output output_dir/
```
Output files created in `output_dir/`:
- `<image_stem>.npz`: Compressed archive containing `map`, `conf`, and `score`.
- `<image_stem>_vis.png`: High-resolution tri-panel visualization (Input RGB, Localization Map, Reliability Map).

### Workflow 3: Run Batch Inference Across All Channels
To benchmark all platform datasets and generate summary metrics:
```bash
python batch_inference_all.py --weights pretrained_models/trufor.pth.tar --data_dir DATA --out_dir results
```

### Workflow 4: WhatsApp Lossless Preprocessing
To compress high-resolution images using WhatsApp's exact high-fidelity settings ($Q=98$, 4:4:4 subsampling):
```bash
python compress_whatsapp.py --input DATA/REAL --output DATA/whatsapp_real_losslessly_compressed
```

### Workflow 5: Run Simulator Validation Tests
To verify all simulator endpoints and compute fidelity against platform test images:
```bash
python test_simulator.py
```

---

## Empirical Evaluation & Benchmark Findings

Below is an excerpt from [`results/evaluation_metrics_all.csv`](results/evaluation_metrics_all.csv) summarizing model performance across uncompressed versus social-media-compressed transmission channels:

| Dataset / Channel | Ground Truth | Mean Integrity Score | Accuracy (%) | Key Forensic Observation |
| :--- | :---: | :---: | :---: | :--- |
| **DATA/REAL (compRAISE)** | Authentic | **0.051** | **94.3%** | Pristine camera images yield very low scores; sensor noise is consistent. |
| **DATA/FAKE (Inpainting)** | Tampered | **0.851** | **97.1%** | Strong disruption in Noiseprint++ creates clear localization anomalies. |
| **WhatsApp Real** | Authentic | **0.141** | **100.0%** | Excellent fidelity retention; minimal false positives under WhatsApp JPEG transcoding. |
| **WhatsApp Fake** | Tampered | **0.816** | **92.0%** | Forgery localization remains sharp despite lossy compression. |
| **Telegram Real** | Authentic | **0.244** | **80.0%** | Lossy requantization increases noise variance, slightly elevating baseline scores. |
| **Telegram Fake** | Tampered | **0.730** | **94.3%** | Tampered regions remain reliably detectable above the 0.12 calibrated threshold. |
| **Facebook Real** | Authentic | **0.263** | **72.0%** | Aggressive 1080px downsampling and high compression trigger false positives at 0.50 threshold. |
| **Facebook Fake** | Tampered | **0.700** | **88.0%** | Detection requires calibrated thresholding to compensate for smoothed edges. |
| **Instagram Real** | Authentic | **0.245** | **76.0%** | WebP encoding preserves visual sharpness but introduces block boundary artifacts. |
| **Instagram Fake** | Tampered | **0.366** | **60.0%** | Most challenging channel: WebP compression significantly attenuates subtle noise disparities. |

*Full analysis, per-image breakdown, and mathematical derivations are available in [TruFor_Evaluation_Report.txt](TruFor_Evaluation_Report.txt).*

---

## Citation & Acknowledgments

If you use TruFor or this repository in your research, please cite the original TruFor publication:

```bibtex
@InProceedings{Guillaro_2023_CVPR,
    author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
    title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20606-20615}
}
```

This research builds upon the foundational work by the **Image Processing Research Group at University Federico II of Naples (GRIP-UNINA)**, supported by the Defense Advanced Research Projects Agency (DARPA) and European Union Horizon Europe vera.ai project.
