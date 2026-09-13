#!/usr/bin/env python3
"""
RT-DETR-L Baseline Training Script for Invoice Field Detection.

Supports configurable dataset path, model checkpoint, training hyperparameters,
and output directory structure for local validation and Kaggle GPU execution.
"""

import argparse
import sys
from pathlib import Path
import yaml
from ultralytics import RTDETR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RT-DETR-L baseline on Invoice Detection Dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/invoice_detection/dataset.yaml",
        help="Path to dataset.yaml configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rtdetr-l.pt",
        help="Model architecture or pretrained weights (default: rtdetr-l.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution size (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size per GPU (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device: '0', '0,1', 'cpu', etc. (default: 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="experiments/baseline/runs",
        help="Directory to save training experiment outputs (default: experiments/baseline/runs).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rtdetr_l_baseline",
        help="Experiment run name (default: rtdetr_l_baseline).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained backbone weights (default: True).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimizer choice (default: auto - framework AdamW default for RT-DETR).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Initial learning rate for RT-DETR (default: 0.0001).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15).",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        default=True,
        help="Run validation after each epoch (default: True).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save checkpoint checkpoints and best weights (default: True).",
    )

    return parser.parse_args()


def validate_dataset_yaml(yaml_path: Path) -> dict:
    """Validate that dataset.yaml exists, is parseable, and has the required fields."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset configuration file not found at: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    required_keys = ["train", "val", "names"]
    for k in required_keys:
        if k not in data_cfg:
            raise ValueError(f"dataset.yaml is missing required key '{k}'. Config: {data_cfg}")

    print(f"[INFO] Verified dataset config from {yaml_path}:")
    print(f"       Path root : {data_cfg.get('path', '<relative>')}")
    print(f"       Train set : {data_cfg['train']}")
    print(f"       Val set   : {data_cfg['val']}")
    print(f"       Classes ({len(data_cfg['names'])}): {data_cfg['names']}")

    return data_cfg


def main():
    args = parse_args()

    print("=" * 60)
    print(" RecoverOps: RT-DETR-L Baseline Training")
    print("=" * 60)
    print(f"Model          : {args.model}")
    print(f"Dataset YAML   : {args.data}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image Size     : {args.imgsz}")
    print(f"Batch Size     : {args.batch}")
    print(f"Device         : {args.device}")
    print(f"Dataloader Wkrs: {args.workers}")
    print(f"Seed           : {args.seed}")
    print(f"Project Dir    : {args.project}")
    print(f"Run Name       : {args.name}")
    print(f"Base LR (lr0)  : {args.lr0}")
    print("=" * 60)

    # 1. Validate dataset configuration
    data_path = Path(args.data).resolve()
    validate_dataset_yaml(data_path)

    # 2. Ensure project output directory exists
    project_path = Path(args.project).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    # 3. Initialize RT-DETR model
    print(f"[INFO] Initializing model architecture: {args.model}")
    model = RTDETR(args.model)

    # 4. Launch training
    print(f"[INFO] Starting RT-DETR baseline training...")
    train_results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(project_path),
        name=args.name,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        val=args.val,
        save=args.save,
        exist_ok=True,
        verbose=True,
    )

    print("=" * 60)
    print(" Training Completed Successfully")
    print("=" * 60)
    best_weights = project_path / args.name / "weights" / "best.pt"
    last_weights = project_path / args.name / "weights" / "last.pt"
    if best_weights.exists():
        print(f"Best model weights saved to : {best_weights}")
    if last_weights.exists():
        print(f"Last model weights saved to : {last_weights}")

    return train_results


if __name__ == "__main__":
    main()
