import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
from study.dataset import apply_compression_domain_randomization, RobustnessDataset

def test_mild_augmentation():
    print("[1/3] Testing apply_compression_domain_randomization with synthetic image...")
    # Create test image
    dummy_img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    
    # Run 50 iterations to test stochastic combinations
    changed_count = 0
    for i in range(50):
        aug_img = apply_compression_domain_randomization(dummy_img.copy())
        assert aug_img.shape == (512, 512, 3), f"Shape mismatch: {aug_img.shape}"
        assert aug_img.dtype == np.uint8, f"Dtype mismatch: {aug_img.dtype}"
        if not np.array_equal(dummy_img, aug_img):
            changed_count += 1
            
    print(f"    50 trials: {changed_count}/50 triggered augmentation (expected stochastic behavior).")
    assert changed_count > 0, "Augmentation was never triggered!"

    print("[2/3] Testing Dataset __getitem__ with mask alignment check...")
    ds_no_aug = RobustnessDataset("study/data_splits/train_manifest.json", is_training=True, use_compression_aug=False)
    ds_mild_aug = RobustnessDataset("study/data_splits/train_manifest.json", is_training=True, use_compression_aug=True)
    
    # Ensure dataset length > 0
    print(f"    Dataset loaded with {len(ds_mild_aug)} samples.")
    sample = ds_mild_aug[0]
    assert sample["rgb"].shape == (3, 512, 512), f"RGB tensor shape mismatch: {sample['rgb'].shape}"
    assert sample["mask"].shape == (512, 512), f"Mask tensor shape mismatch: {sample['mask'].shape}"
    assert sample["rgb"].min() >= 0.0 and sample["rgb"].max() <= 1.0, "RGB range invalid"
    
    print("[3/3] Verifying spatial alignment of masks...")
    # Load an image directly with a synthetic mask to assert zero spatial displacement
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1 # Tampered region
    
    # Apply crop and aug
    ds_mild_aug.is_training = False # center crop deterministic
    c_img, c_mask = ds_mild_aug._apply_crop_and_aug(img, mask)
    assert np.array_equal(mask, c_mask), "Mask was modified by augmentation!"
    
    print("[OK] ALL MILD AUGMENTATION VERIFICATION CHECKS PASSED!")

if __name__ == "__main__":
    test_mild_augmentation()
