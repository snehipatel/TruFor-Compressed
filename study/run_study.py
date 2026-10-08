"""
Master Research Study CLI Controller.
Enables running the entire platform-robustness study directly in this IDE:
- Experiment 1: Conservative / Detection-Only Fine-Tuning (frozen decoder, lr=1e-5)
- Experiment 2: Low-LR All-Heads Fine-Tuning (decode_head + conf + det, lr=1e-5)
- 3-Way Comparative Benchmark: Baseline vs Experiment 1 vs Experiment 2 on unseen test datasets

Usage:
    # 1. Run Experiment 1 (Detection classifier only, lr=1e-5):
    python study/run_study.py --action exp1 --epochs 5 --lr 1e-5

    # 2. Run Experiment 2 (All heads, low-LR=1e-5):
    python study/run_study.py --action exp2 --epochs 5 --lr 1e-5

    # 3. Compare Baseline vs Exp1 vs Exp2 on unseen test datasets:
    python study/run_study.py --action compare --max_eval_per_class 50

    # 4. Run both experiments and final 3-way comparison end-to-end:
    python study/run_study.py --action both --epochs 5 --lr 1e-5
"""

import os
import sys
import argparse
from pathlib import Path

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from study.dataset_builder import build_train_val_manifests
from study.trainer import run_training
from study.evaluator import run_benchmark_comparison


def main():
    parser = argparse.ArgumentParser(description="TruFor Platform Compression Robustness Study")
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["prepare", "train", "evaluate", "all", "exp1", "exp2", "compare", "both"],
        help="Action to execute"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default 1e-5 for stable fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (1 recommended for 8GB VRAM)")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--samples_per_epoch", type=int, default=600, help="Balanced samples per epoch")
    parser.add_argument("--max_eval_per_class", type=int, default=50, help="Max test images per class for benchmark")
    parser.add_argument("--use_compression_aug", action="store_true", help="Enable compression-aware domain randomization")
    parser.add_argument("--save_name", type=str, default=None, help="Custom checkpoint save filename")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry-run to test end-to-end pipeline")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("\n" + "="*70)
        print(" RUNNING QUICK DRY-RUN SMOKE TEST")
        print("="*70)
        print("[1/3] Building manifests...")
        build_train_val_manifests()
        
        print("\n[2/3] Running 1 epoch with 10 samples...")
        ckpt = run_training(
            epochs=1,
            batch_size=1,
            accum_steps=2,
            samples_per_epoch=10,
            val_samples=6,
            save_name="dry_run_model.pth"
        )
        
        print("\n[3/3] Running benchmark evaluation on small test slice (5 per class)...")
        run_benchmark_comparison(models_dict={"Dry-Run Model": ckpt}, max_eval_per_class=5)
        print("\n[OK] DRY RUN COMPLETED SUCCESSFULLY!")
        return

    if args.action in ["prepare", "all", "both"]:
        print("\n[Step 1] Preparing Train/Val Manifests & Enforcing Zero Leakage...")
        build_train_val_manifests()
        
    if args.action in ["exp1", "both"]:
        print("\n" + "="*75)
        print(" [EXPERIMENT 1] CONSERVATIVE / DETECTION-ONLY FINE-TUNING")
        print(" Strategy: Backbone & Decoders FROZEN | Only 1,281-param classifier trained")
        print("="*75)
        run_training(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            samples_per_epoch=args.samples_per_epoch,
            fine_tune_mode="detection_only",
            save_name="exp1_detection_only.pth"
        )

    if args.action in ["exp2", "both"]:
        print("\n" + "="*75)
        print(" [EXPERIMENT 2] LOW-LR ALL-HEADS FINE-TUNING")
        print(" Strategy: Backbone FROZEN | decode_head, conf_head, det_head trained at low LR")
        print("="*75)
        if args.save_name:
            save_name = args.save_name
        else:
            save_name = "exp2_mild_aug.pth" if args.use_compression_aug else "exp2_low_lr.pth"
        run_training(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            samples_per_epoch=args.samples_per_epoch,
            fine_tune_mode="heads_only",
            save_name=save_name,
            use_compression_aug=args.use_compression_aug
        )

    if args.action in ["train", "all"]:
        print(f"\nTraining Platform-Robust TruFor ({args.epochs} epochs)...")
        run_training(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            samples_per_epoch=args.samples_per_epoch,
            save_name="best_robust_model.pth"
        )
        
    if args.action in ["evaluate", "all", "compare", "both"]:
        print("\n" + "="*75)
        print(" EVALUATING 3-WAY BENCHMARK ON UNSEEN TEST DATASETS")
        print(" Baseline (Pretrained) vs Approach 1 (Det-Only) vs Approach 2 (Low-LR)")
        print("="*75)
        run_benchmark_comparison(max_eval_per_class=args.max_eval_per_class)


if __name__ == "__main__":
    main()
