"""
Model Loader and Multi-Task Loss Manager for TruFor Fine-Tuning.
Handles:
- Loading pre-trained TruFor model and official checkpoint
- Module freezing (Noiseprint++ DnCNN preserved)
- Parameter groups with differential learning rates
- Scientific Multi-task Loss: Localization (Dice+CE) + Confidence (MSE) + Global Detection (BCE)
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "test_docker" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target_mask):
        # pred_logits: [B, 2, H, W], target_mask: [B, H, W]
        prob = F.softmax(pred_logits, dim=1)[:, 1]
        target = target_mask.float()
        
        intersection = (prob * target).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, w_loc=1.0, w_conf=1.0, w_det=0.5):
        super().__init__()
        self.w_loc = w_loc
        self.w_conf = w_conf
        self.w_det = w_det
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5]).cuda() if torch.cuda.is_available() else torch.tensor([0.5, 2.5]))
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_loc, pred_conf, pred_det, target_mask, target_det):
        # 1. Localization Loss (CrossEntropy + Dice)
        ce = self.ce_loss(pred_loc, target_mask)
        dice = self.dice_loss(pred_loc, target_mask)
        loss_loc = ce + dice
        
        # 2. Confidence Loss (MSE between predicted confidence and True Class Probability)
        pred_prob = F.softmax(pred_loc, dim=1)
        # TCP = p(class 1) if mask==1 else p(class 0)
        tcp = pred_prob[:, 1:2] * (target_mask.unsqueeze(1) == 1) + pred_prob[:, 0:1] * (target_mask.unsqueeze(1) == 0)
        loss_conf = self.mse_loss(torch.sigmoid(pred_conf), tcp.detach())
        
        # 3. Detection Loss (Binary Cross-Entropy)
        loss_det = self.bce_loss(pred_det, target_det)
        
        total_loss = self.w_loc * loss_loc + self.w_conf * loss_conf + self.w_det * loss_det
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_loc": loss_loc.item(),
            "loss_conf": loss_conf.item(),
            "loss_det": loss_det.item()
        }


def get_trufor_model(weights_path: str = None, fine_tune_mode: str = "heads_only", device=None):
    """
    Builds TruFor model and loads pre-trained weights.
    fine_tune_mode:
        - 'heads_only': freezes backbone & DnCNN, only trains decode_head, conf_head, and det_head.
        - 'differential': backbone trained with 0.1x LR, heads trained with 1.0x LR.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    cfg_file = SRC_DIR / "trufor.yaml"
    config.defrost()
    config.merge_from_file(str(cfg_file))
    config.freeze()
    
    model = confcmx(cfg=config)
    
    # Load weights
    if weights_path is None:
        weights_path = BASE_DIR / "pretrained_models" / "trufor.pth.tar"
        if not weights_path.exists():
            weights_path = BASE_DIR / "TruFor_train_test" / "pretrained_models" / "trufor.pth.tar"
            
    print(f"[*] Loading checkpoint weights: {weights_path}")
    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Always freeze Noiseprint++ DnCNN (preserves fundamental noise residual extractor)
    for p in model.dncnn.parameters():
        p.requires_grad = False
    model.dncnn.eval()
    
    if fine_tune_mode == "detection_only":
        print("[*] Fine-tuning strategy: DETECTION CLASSIFIER ONLY (Backbone & Decoders completely frozen)")
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "detection") and model.detection is not None:
            for p in model.detection.parameters():
                p.requires_grad = True
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"    Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        return model, trainable_params

    elif fine_tune_mode == "heads_only":
        print("[*] Fine-tuning strategy: ALL HEADS (Backbone frozen, decode_head, conf_head, det_head trainable)")
        for p in model.backbone.parameters():
            p.requires_grad = False
        if model.backbone_conf is not None:
            for p in model.backbone_conf.parameters():
                p.requires_grad = False
                
        # Unfreeze heads
        for p in model.decode_head.parameters():
            p.requires_grad = True
        if model.decode_head_conf is not None:
            for p in model.decode_head_conf.parameters():
                p.requires_grad = True
        if hasattr(model, "detection") and model.detection is not None:
            for p in model.detection.parameters():
                p.requires_grad = True
                
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"    Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        return model, trainable_params
        
    elif fine_tune_mode == "differential":
        print("[*] Fine-tuning strategy: DIFFERENTIAL LR (Backbone 0.1x LR, Heads 1.0x LR)")
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.decode_head.parameters())
        if model.decode_head_conf is not None:
            head_params.extend(list(model.decode_head_conf.parameters()))
        if hasattr(model, "detection") and model.detection is not None:
            head_params.extend(list(model.detection.parameters()))
            
        param_groups = [
            {"params": backbone_params, "lr_scale": 0.1},
            {"params": head_params, "lr_scale": 1.0}
        ]
        return model, param_groups
    else:
        raise ValueError(f"Unknown fine_tune_mode: {fine_tune_mode}")
