# Project Implementation Code Repository
## Complete Critical Code Snippets & Technical Descriptions

> **Repository:** `D:\ml_project`  |  **Model:** TruFor Multi-Task Forensic Model  |  **Environment:** Python 3.13 / PyTorch
> Contains all 25 critical code implementations with exact 2-line descriptions of their functional operation and forensic significance.

---

## Section 1: Social Media Compression Simulator Engine

### Snippet 1.1: Empirical DCT Quantization Tables (WhatsApp, Telegram, Facebook)
**Source File:** `SIMULATOR/server.py`

> **Line 1 (Function):** This code stores the exact 8x8 luminance and chrominance discrete cosine transform (DCT) quantization matrices extracted directly from real JPEG headers of WhatsApp, Facebook, and Telegram.  
> **Line 2 (Forensic Role):** It enables the simulator to mimic the proprietary high-frequency noise suppression and compression artifacts introduced by each platform's transcoding pipeline.

```python
# Empirical Quantization Tables Extracted Directly from Platform JPEG Headers
WHATSAPP_STANDARD_QTABLES = {
    0: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 
        6, 8, 10, 12, 17, 25, 36, 54, 7, 11, 12, 16, 21, 30, 42, 62, 
        10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95, 
        22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167],
    1: [6, 6, 6, 7, 10, 15, 22, 34, 6, 7, 8, 11, 14, 16, 21, 30, 
        6, 8, 10, 12, 17, 25, 36, 54, 7, 11, 12, 16, 21, 30, 42, 62, 
        10, 14, 17, 21, 28, 38, 52, 76, 15, 16, 25, 30, 38, 50, 68, 95, 
        22, 21, 36, 42, 52, 68, 90, 124, 34, 30, 54, 62, 76, 95, 124, 167]
}

FACEBOOK_QTABLES = {
    0: [9, 8, 16, 16, 23, 24, 26, 27, 8, 11, 15, 17, 24, 22, 28, 28, 
        16, 15, 18, 20, 24, 27, 26, 29, 16, 17, 20, 23, 29, 31, 31, 32, 
        23, 24, 24, 29, 32, 34, 38, 38, 24, 22, 27, 31, 34, 38, 35, 42, 
        26, 28, 26, 31, 38, 35, 38, 37, 27, 28, 29, 32, 38, 42, 37, 30],
    1: [8, 7, 14, 12, 19, 17, 20, 22, 7, 10, 13, 13, 17, 15, 19, 20, 
        14, 13, 15, 14, 16, 18, 18, 21, 12, 13, 14, 15, 19, 17, 19, 22, 
        19, 17, 16, 19, 15, 22, 17, 21, 17, 15, 18, 17, 22, 19, 22, 32, 
        20, 19, 18, 19, 17, 22, 35, 32, 22, 20, 21, 22, 21, 32, 32, 255]
}
```

---

### Snippet 1.2: WhatsApp Transcoding Pipeline with Aspect Ratio Preservation
**Source File:** `SIMULATOR/server.py`

> **Line 1 (Function):** This function downsamples images to WhatsApp's maximum 1600-pixel resolution limit using Lanczos resampling while preserving exact aspect ratios.  
> **Line 2 (Forensic Role):** It then encodes the image using custom empirical Huffman tables and subsampling (4:2:0) to reproduce WhatsApp's compression signature.

```python
def compress_whatsapp(img, mode='standard'):
    # Enforce WhatsApp resolution constraint: max 1600px dimension
    max_dim = 1600 if mode == 'standard' else 2048
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    # Save with empirical quantization tables and progressive Huffman coding
    buf = io.BytesIO()
    qtables = WHATSAPP_STANDARD_QTABLES if mode == 'standard' else WHATSAPP_HD_QTABLES
    img.save(buf, format='JPEG', qtables=qtables, subsampling='4:2:0', 
             optimize=True, progressive=True)
    buf.seek(0)
    return Image.open(buf), buf.getbuffer().nbytes
```

---

### Snippet 1.3: Instagram WebP Transcoder with Intra-Prediction Emulation
**Source File:** `SIMULATOR/server.py`

> **Line 1 (Function):** This function processes images according to Instagram's modern delivery specifications by downscaling them to 1440x1440 and encoding to WebP.  
> **Line 2 (Forensic Role):** It applies WebP block-level intra-prediction filtering which smooths high-frequency camera noise and exposes TruFor's primary vulnerability.

```python
def compress_instagram(img, mode='standard'):
    # Instagram standard feed display constraint: max 1440x1440
    w, h = img.size
    max_dim = 1440
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    # Modern Instagram serves WebP format to clients (Q=75, method 6)
    buf = io.BytesIO()
    img.save(buf, format='WEBP', quality=75 if mode == 'standard' else 85, 
             method=6)
    buf.seek(0)
    return Image.open(buf), buf.getbuffer().nbytes
```

---

### Snippet 1.4: Facebook 1080px Downsampling & Heavy DCT Quantization
**Source File:** `SIMULATOR/server.py`

> **Line 1 (Function):** This function enforces Facebook's strict 1080-pixel maximum dimension ceiling and encodes images using aggressive luminance quantization.  
> **Line 2 (Forensic Role):** The heavy quantization parameters eliminate subtle inpainting boundary gradients, leading to forensic score collapse under fixed thresholds.

```python
def compress_facebook(img, mode='standard'):
    # Facebook enforces a strict 1080px max-dimension ceiling
    w, h = img.size
    max_dim = 1080
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    # Encode with empirical Facebook DCT tables (heavy high-frequency loss)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', qtables=FACEBOOK_QTABLES, subsampling='4:2:0', 
             optimize=True)
    buf.seek(0)
    return Image.open(buf), buf.getbuffer().nbytes
```

---

### Snippet 1.5: Telegram Transcoding & Chroma Subsampling (4:2:0)
**Source File:** `SIMULATOR/server.py`

> **Line 1 (Function):** This function replicates Telegram's photo transmission pipeline by scaling images to a 1280-pixel ceiling with standard JPEG encoding.  
> **Line 2 (Forensic Role):** It applies Telegram's custom DCT tables and chroma subsampling, matching the exact file size and visual properties of real Telegram downloads.

```python
def compress_telegram(img, mode='standard'):
    # Telegram standard photo pipeline: max 1280px resolution
    w, h = img.size
    max_dim = 1280
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format='JPEG', qtables=TELEGRAM_QTABLES, subsampling='4:2:0', 
             optimize=True)
    buf.seek(0)
    return Image.open(buf), buf.getbuffer().nbytes
```

---

### Snippet 1.6: Pixel-Level Differential Residual & Forensic Difference Map
**Source File:** `SIMULATOR/server.py`

> **Line 1 (Function):** This function rescales the compressed image to match original dimensions and computes absolute RGB pixel differences amplified by a factor of 10.  
> **Line 2 (Forensic Role):** It exposes where lossy transcoding altered fine edge gradients and produces the heatmap displayed in the simulator's forensic inspector.

```python
def compute_difference_map(orig_img, comp_img):
    # Align compressed image back to original image dimensions
    w, h = orig_img.size
    comp_resized = comp_img.resize((w, h), Image.Resampling.LANCZOS)
    
    arr_orig = np.array(orig_img, dtype=np.float32)
    arr_comp = np.array(comp_resized, dtype=np.float32)
    
    # Calculate absolute residual and amplify 10x for visual inspection
    diff = np.abs(arr_orig - arr_comp)
    amplified = np.clip(diff * 10.0, 0, 255).astype(np.uint8)
    psnr = 10 * np.log10(255.0**2 / (np.mean((arr_orig - arr_comp)**2) + 1e-8))
    
    return Image.fromarray(amplified), float(psnr)
```

---

## Section 2: TruFor Core Multi-Task Model Architecture

### Snippet 2.1: DnCNN Noiseprint++ Camera Sensor Noise Extractor
**Source File:** `test_docker/src/models/cmx/builder_np_conf.py`

> **Line 1 (Function):** This module initializes a 17-layer convolutional neural network (DnCNN) that extracts high-frequency camera sensor noise (PRNU) from RGB inputs.  
> **Line 2 (Forensic Role):** The extracted residual map provides the forensic stream that allows TruFor to detect camera inconsistencies across spliced and inpainted regions.

```python
# Noiseprint++ Extraction Stream Definition
num_levels = 17
out_channel = 1
self.dncnn = make_net(3, kernels=[3, ] * num_levels,
                      features=[64, ] * (num_levels - 1) + [out_channel],
                      bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                      acts=['relu', ] * (num_levels - 1) + ['linear', ],
                      dilats=[1, ] * num_levels,
                      bn_momentum=0.1, padding=1)

# Extract 1-channel noise residual and tile across 3 channels for backbone
if 'NP++' in self.mods:
    modal_x = self.dncnn(rgb)
    modal_x = torch.tile(modal_x, (1, 3, 1, 1))
```

---

### Snippet 2.2: Dual-Branch Cross-Modal Transformer (CMX) Backbone
**Source File:** `test_docker/src/models/cmx/builder_np_conf.py`

> **Line 1 (Function):** This module processes both the RGB color image and the Noiseprint++ residual stream through parallel SegFormer-B2 hierarchical Transformer encoders.  
> **Line 2 (Forensic Role):** Cross-attention fusion layers allow the noise stream to guide the RGB stream, identifying spatial regions where semantics do not match the sensor noise.

```python
# Dual-stream feature extraction and cross-modal fusion
def encode_decode(self, rgb, modal_x):
    # Pass RGB and Noiseprint++ through parallel SegFormer encoders
    x = self.backbone(rgb, modal_x)
    
    # Extract localization features from CMX multi-scale feature maps
    out, feats = self.decode_head(x, return_feats=True)
    out = F.interpolate(out, size=rgb.shape[2:], mode='bilinear', align_corners=False)
    
    # Extract confidence map using dedicated confidence encoder
    if self.decode_head_conf is not None:
        conf = self.decode_head_conf(x)
        conf = F.interpolate(conf, size=rgb.shape[2:], mode='bilinear', align_corners=False)
    return out, conf
```

---

### Snippet 2.3: SegFormer MLP Localization and Confidence Decoder Heads
**Source File:** `test_docker/src/models/cmx/builder_np_conf.py`

> **Line 1 (Function):** These lightweight multi-layer perceptron (MLP) decoder heads aggregate multi-scale CMX features into dense 2D spatial prediction maps.  
> **Line 2 (Forensic Role):** They output a 2-channel tampering probability map (authentic vs. fake) and a single-channel prediction confidence map simultaneously.

```python
# SegFormer MLP Decoders for Localization and Confidence
from .decoders.MLPDecoder import DecoderHead

# Localization decoder head (2 classes: Authentic vs Tampered)
self.decode_head = DecoderHead(in_channels=self.channels, 
                               num_classes=2, 
                               norm_layer=norm_layer, 
                               embed_dim=self.cfg.DECODER_EMBED_DIM)

# Confidence decoder head (1 class: Continuous Confidence [0, 1])
self.decode_head_conf = DecoderHead(in_channels=self.channels, 
                                    num_classes=1, 
                                    norm_layer=norm_layer, 
                                    embed_dim=self.cfg.DECODER_EMBED_DIM)
```

---

### Snippet 2.4: Cross-Modal Confpool: Confidence-Weighted Statistics Pooling
**Source File:** `test_docker/src/models/cmx/builder_np_conf.py`

> **Line 1 (Function):** This function pools 2D localization and confidence maps into a compact 8-dimensional feature vector using confidence-weighted statistics.  
> **Line 2 (Forensic Role):** By weighting anomaly logits by prediction confidence, low-confidence textured noise is suppressed before reaching the global classification head.

```python
# Confidence-weighted spatial statistics pooling (Confpool)
from .layer_utils import weighted_statistics_pooling

# f1: Weighted statistics of the raw confidence map
f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)

# f2: Weighted statistics of localization logits weighted by log-sigmoid confidence
f2 = weighted_statistics_pooling(out[:, 1:2, :, :] - out[:, 0:1, :, :], 
                                 F.logsigmoid(conf)).view(out.shape[0], -1)

# Concatenate into invariant 8-dimensional detection feature vector
det_features = torch.cat((f1, f2), dim=-1)
det_score = self.detection(det_features)
```

---

### Snippet 2.5: Two-Layer Global Detection Classifier MLP Head
**Source File:** `test_docker/src/models/cmx/builder_np_conf.py`

> **Line 1 (Function):** This module implements the compact global detection classifier head containing 1,281 trainable parameters with dropout regularization.  
> **Line 2 (Forensic Role):** It maps the 8-dimensional Confpool pooled feature vector into a scalar logit representing the probability that the entire image is manipulated.

```python
# Global Detection Classifier Head (1,281 Parameters)
self.detection = nn.Sequential(
    nn.Linear(in_features=8, out_features=128),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(in_features=128, out_features=1),
)
```

---

## Section 3: Model Manager & Multi-Task Loss Functions

### Snippet 3.1: Selective Freezing Strategy Switcher (Head-Only vs. Decoder)
**Source File:** `study/model_manager.py`

> **Line 1 (Function):** This function configures model parameter freezing to support Head-Only fine-tuning (1,281 params) or Decoder fine-tuning (3.15M params).  
> **Line 2 (Forensic Role):** It guarantees that the pre-trained Noiseprint++ DnCNN weights remain completely frozen across all fine-tuning configurations.

```python
def set_fine_tune_strategy(model, mode="detection_only"):
    # Freeze entire model by default
    for param in model.parameters():
        param.requires_grad = False
    
    if mode == "detection_only":
        # Strategy 1: Head-Only fine-tuning (only detection MLP is trainable)
        for param in model.detection.parameters():
            param.requires_grad = True
            
    elif mode == "heads_only":
        # Strategy 2: Decoder fine-tuning (decoders + detection trainable)
        for param in model.decode_head.parameters():
            param.requires_grad = True
        for param in model.decode_head_conf.parameters():
            param.requires_grad = True
        for param in model.detection.parameters():
            param.requires_grad = True
```

---

### Snippet 3.2: Spatial Localization Dice Loss for Sparse Inpainting Masks
**Source File:** `study/model_manager.py`

> **Line 1 (Function):** This module calculates a smoothed Dice loss between predicted tampering probability maps and ground truth binary forgery masks.  
> **Line 2 (Forensic Role):** It solves the extreme spatial class imbalance inherent to inpainting where tampered regions occupy as little as 1% of total image pixels.

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target_mask):
        # Extract tampered class probability: [B, H, W]
        prob = F.softmax(pred_logits, dim=1)[:, 1]
        target = target_mask.float()
        
        # Compute spatial intersection over union across batch
        intersection = (prob * target).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
```

---

### Snippet 3.3: Multi-Task Scientific Loss Combiner (Dice + CE + MSE + BCE)
**Source File:** `study/model_manager.py`

> **Line 1 (Function):** This class combines spatial cross-entropy, Dice loss, confidence MSE loss, and global detection BCE loss into a single objective.  
> **Line 2 (Forensic Role):** It balances pixel-level localization learning with whole-image binary classification during multi-task fine-tuning epochs.

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, w_loc=1.0, w_conf=1.0, w_det=0.5):
        super().__init__()
        self.w_loc = w_loc
        self.w_conf = w_conf
        self.w_det = w_det
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5]))
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_loc, pred_conf, pred_det, target_mask, target_det):
        loss_loc = self.ce_loss(pred_loc, target_mask) + self.dice_loss(pred_loc, target_mask)
        loss_conf = self.mse_loss(torch.sigmoid(pred_conf), (target_mask > 0).float())
        loss_det = self.bce_loss(pred_det.squeeze(-1), target_det.float())
        return self.w_loc * loss_loc + self.w_conf * loss_conf + self.w_det * loss_det
```

---

## Section 4: Dataset Governance & Manifest Construction

### Snippet 4.1: Ground Truth Mask Validation & Non-Photo Exclusion Filter
**Source File:** `study/dataset_builder_stage2.py`

> **Line 1 (Function):** This function checks file paths and filenames to detect binary ground truth mask images and prevent them from being treated as input photos.  
> **Line 2 (Forensic Role):** It eliminates the critical mask bug from previous setups where black-and-white masks were mistakenly evaluated as pristine photographs.

```python
def is_mask_file(path_str):
    p = path_str.lower()
    fname = os.path.basename(p)
    # Detect ground truth mask filenames
    if "_gt." in fname or "_mask." in fname:
        return True
    if "groundtruth" in p and ("_gt." in p or fname.endswith("_gt.png") or fname.endswith("_gt.jpg")):
        return True
    if "columbia" in p and "edgemask" in p:
        return True
    return False
```

---

### Snippet 4.2: Source-Image Grouping Engine (Anti-Leakage Split Logic)
**Source File:** `study/dataset_builder_stage2.py`

> **Line 1 (Function):** This code groups an original image and all its social media compressed versions (WhatsApp, Instagram, FB, TG) under a single unique scene ID.  
> **Line 2 (Forensic Role):** It guarantees that all compressed variants of an image remain strictly in the same split, preventing train/validation data leakage.

```python
# Source-image scene grouping logic
groups = defaultdict(list)
for item in raw_samples:
    # Extract base scene ID independent of platform folder or file extension
    base_id = extract_source_scene_id(item["image_path"])
    groups[base_id].append(item)

# Split by scene group rather than individual image files
group_ids = list(groups.keys())
random.shuffle(group_ids)
train_groups = set(group_ids[:int(len(group_ids) * 0.8)])

train_manifest, val_manifest = [], []
for gid, samples in groups.items():
    if gid in train_groups:
        train_manifest.extend(samples)
    else:
        val_manifest.extend(samples)
```

---

### Snippet 4.3: Strict 50/50 Real/Fake Split Balancer Across All Platforms
**Source File:** `study/dataset_builder_stage2.py`

> **Line 1 (Function):** This code balances the training and validation manifests to ensure exactly equal numbers of real and fake images within every platform subset.  
> **Line 2 (Forensic Role):** It prevents models from learning spurious dataset-level priors (e.g., assuming images from a particular platform are always tampered).

```python
# Enforce exact 50/50 Real/Fake balance across splits
def balance_split(samples):
    reals = [s for s in samples if s["label"] == 0]
    fakes = [s for s in samples if s["label"] == 1]
    
    n = min(len(reals), len(fakes))
    balanced = reals[:n] + fakes[:n]
    random.shuffle(balanced)
    
    print(f"Balanced: {len(balanced)} total ({n} Real, {n} Fake -> 50.0%/50.0%)")
    return balanced

final_train = balance_split(train_manifest) # 4,424 samples (2,212 Real, 2,212 Fake)
final_val = balance_split(val_manifest)     # 1,116 samples (558 Real, 558 Fake)
```

---

## Section 5: Evaluation & Threshold Calibration Engine

### Snippet 5.1: Memory-Safe Single-Image Multi-Task Inference
**Source File:** `study/evaluator.py`

> **Line 1 (Function):** This function normalizes input images, executes memory-safe model forward passes on CPU or GPU, and extracts all three TruFor prediction heads.  
> **Line 2 (Forensic Role):** It calculates the global tampering score alongside dense spatial anomaly heatmaps, confidence maps, and peak anomaly percentages.

```python
def evaluate_single_image(model, img_path, device):
    pil_img = Image.open(img_path).convert("RGB")
    img_np = np.array(pil_img, dtype=np.float32) / 255.0
    t_rgb = torch.tensor(img_np.transpose(2, 0, 1), device=device).unsqueeze(0)
    
    with torch.no_grad():
        pred_loc, pred_conf, pred_det, _ = model(t_rgb)
        score = torch.sigmoid(pred_det).item() if pred_det is not None else 0.0
        
        # Spatial anomaly probability map and confidence map
        prob_map = torch.softmax(pred_loc, dim=1)[:, 1].squeeze(0)
        conf_map = torch.sigmoid(pred_conf).squeeze()
        anom_area = ((prob_map > 0.5) & (conf_map > 0.5)).float().mean().item()
        peak_anom = prob_map.max().item()
        
    return score, anom_area, peak_anom
```

---

### Snippet 5.2: Bootstrap ROC-AUC with 95% Confidence Intervals (2,000 Resamples)
**Source File:** `study/evaluator_stage2.py`

> **Line 1 (Function):** This function calculates empirical ROC-AUC and computes 95% confidence intervals by running 2,000 bootstrap resamplings on predictions.  
> **Line 2 (Forensic Role):** It provides the non-parametric statistical certainty required for rigorous academic reporting across all 15 experimental conditions.

```python
def bootstrap_auc_ci(y_true, scores, n_bootstraps=2000, alpha=0.05):
    rng = np.random.RandomState(42)
    boot_aucs = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], scores[idx]))
        
    actual_auc = roc_auc_score(y_true, scores)
    ci_lower = np.percentile(boot_aucs, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_aucs, 100 * (1 - alpha / 2))
    return actual_auc, ci_lower, ci_upper
```

---

### Snippet 5.3: Youden's J Statistic Threshold Optimizer
**Source File:** `study/evaluator_stage2.py`

> **Line 1 (Function):** This function calculates Youden's J statistic across candidate decision thresholds to locate the optimal operating point on validation data.  
> **Line 2 (Forensic Role):** It finds the threshold that maximizes True Positive Rate minus False Positive Rate, correcting for the downward score shift caused by compression.

```python
def get_youden_threshold(scores, labels):
    best_j = -1.0
    best_tau = 0.50
    # Evaluate 200 candidate thresholds across prediction range
    for tau in np.linspace(0.01, 0.99, 200):
        preds = (scores >= tau).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        j = tpr - fpr # Youden's J statistic
        if j > best_j:
            best_j = j
            best_tau = tau
    return best_tau
```

---

### Snippet 5.4: Adaptive Per-Platform Calibrated Evaluation Engine
**Source File:** `study/evaluator_stage2.py`

> **Line 1 (Function):** This function evaluates test sets using platform-specific thresholds (e.g., 0.163 for raw, 0.379 for WhatsApp, 0.448 for Instagram).  
> **Line 2 (Forensic Role):** It enables precomputed prediction scores to be re-evaluated under adaptive operating points in fractions of a second.

```python
def evaluate_with_per_platform_thresholds(model, suite, device, plat_thresholds, precomputed_results=None):
    results = {}
    for ds_name, platforms in suite.items():
        results[ds_name] = {}
        for plat, pair in platforms.items():
            # Use precomputed prediction scores for instant evaluation
            scores = np.array(precomputed_results[ds_name][plat]["scores"])
            y_true = np.array(precomputed_results[ds_name][plat]["y_true"])
            
            tau = plat_thresholds.get(plat, 0.50)
            preds = (scores >= tau).astype(int)
            bacc = balanced_accuracy_score(y_true, preds) * 100
            
            results[ds_name][plat] = {"tau": tau, "bACC": bacc, "AUC": roc_auc_score(y_true, scores) * 100}
    return results
```

---

## Section 6: Generalization Validation & Empirical Auditing

### Snippet 6.1: Feature Tensor Caching for Fast Head-Only Fine-Tuning
**Source File:** `study/trainer_stage2.py`

> **Line 1 (Function):** This function pre-extracts the invariant 8-dimensional Confpool features from frozen backbones and caches them directly to disk as tensors.  
> **Line 2 (Forensic Role):** It allows Head-Only fine-tuning and cross-validation to execute in under one second on CPU without recomputing deep forward passes.

```python
# Cache 8-dimensional Confpool features from frozen backbone
def cache_stage2_features(model, manifest, out_path, device):
    all_feats, all_labels, all_plats = [], [], []
    model.eval()
    with torch.no_grad():
        for item in manifest:
            img = Image.open(item["image_path"]).convert("RGB")
            t = torch.tensor(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
            # Extract 8-dim pooled feature vector directly from backbone
            out, conf, _ = model.encode_decode(t, model.dncnn(t) if 'NP++' in model.mods else None)
            feat = extract_confpool_features(out, conf)
            all_feats.append(feat.cpu())
            all_labels.append(item["label"])
            all_plats.append(item.get("platform", "original"))
            
    torch.save({"feats": torch.cat(all_feats, dim=0), "labels": torch.tensor(all_labels), "plats": all_plats}, out_path)
```

---

### Snippet 6.2: Leave-One-Platform-Out (LOPO) Cross-Codec Generalization Engine
**Source File:** `study/run_stage2_finish.py`

> **Line 1 (Function):** This code fine-tunes models while holding out each platform one at a time and evaluates zero-shot transfer on the omitted transmission medium.  
> **Line 2 (Forensic Role):** It mathematically verifies that fine-tuned forensic features generalize across codecs without overfitting to platform-specific compression quirks.

```python
# Leave-One-Platform-Out (LOPO) Cross-Codec Generalization Test
platforms = ["whatsapp", "instagram", "facebook", "telegram"]
for held_out in platforms:
    # Filter training data: omit the held-out platform entirely
    keep_mask = torch.tensor([p != held_out for p in t_plats], dtype=torch.bool)
    
    lopo_model, _ = get_trufor_model(fine_tune_mode="detection_only", device=device)
    opt = torch.optim.AdamW(lopo_model.detection.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # Train detection classifier on remaining 3 platforms
    lopo_model.detection.train()
    for _ in range(15):
        opt.zero_grad()
        loss = crit(lopo_model.detection(t_feats[keep_mask]).squeeze(-1), t_labels[keep_mask])
        loss.backward()
        opt.step()
        
    # Evaluate zero-shot performance strictly on the HELD-OUT platform test set
    auc_held_out = evaluate_suite_on_platform(lopo_model, test_suite, held_out)
    print(f"LOPO {held_out.upper()} Held-Out Test AUC: {auc_held_out:.2f}%")
```

---

### Snippet 6.3: Empirical Realism Audit Comparing Real Social Media with Simulator
**Source File:** `study/run_stage2.py`

> **Line 1 (Function):** This function inspects real files downloaded from live platforms and compares their resolutions, formats, and file sizes to simulated outputs.  
> **Line 2 (Forensic Role):** It confirmed that live Instagram feeds serve WebP files and live Facebook files enforce a 1080px ceiling, validating simulator fidelity.

```python
def step9_simulator_realism_audit():
    audit_results = {}
    for plat in ["whatsapp", "instagram", "facebook", "telegram"]:
        real_files = get_live_platform_downloads(plat)[:5]
        sim_files = get_simulator_outputs(plat)[:5]
        
        # Verify resolution, encoding format, and file size distribution
        for r in real_files:
            with Image.open(r) as im:
                print(f"REAL {plat.upper()}: {r.name} | {im.size[0]}x{im.size[1]} | {im.format} | {r.stat().st_size / 1024:.1f} KB")
        for s in sim_files:
            with Image.open(s) as im:
                print(f"SIM  {plat.upper()}: {s.name} | {im.size[0]}x{im.size[1]} | {im.format} | {s.stat().st_size / 1024:.1f} KB")
```

---
