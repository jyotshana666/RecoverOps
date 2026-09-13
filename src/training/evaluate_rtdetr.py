#!/usr/bin/env python3
"""
RT-DETR-L Evaluation and Failure Case Extraction Script.

Evaluates a trained RT-DETR-L checkpoint on the validation split,
reports overall and per-class metrics (Precision, Recall, mAP50, mAP50-95),
and provides failure case collection infrastructure (false positives, false negatives,
low-confidence detections, class confusions).
"""

import argparse
import json
import os
from pathlib import Path
import yaml
import cv2
import numpy as np
from ultralytics import RTDETR

CLASS_NAMES = ["amount_due", "date_due", "document_id", "date_issue", "vendor_name"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RT-DETR-L on invoice detection dataset."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="experiments/baseline/runs/rtdetr_l_baseline/weights/best.pt",
        help="Path to trained model weights (.pt file).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/invoice_detection/dataset.yaml",
        help="Path to dataset.yaml configuration file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "train", "test"],
        help="Dataset split to evaluate on (default: val).",
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
        help="Evaluation batch size (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Compute device: '0', '0,1', 'cpu', etc. (default: 0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation metric computation (default: 0.5).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/baseline_evaluation.json",
        help="Path to save evaluation metrics in JSON format.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="results/baseline_evaluation.md",
        help="Path to save evaluation summary in Markdown format.",
    )
    parser.add_argument(
        "--failure-dir",
        type=str,
        default="results/failure_cases",
        help="Directory to save collected failure cases.",
    )
    parser.add_argument(
        "--collect-failures",
        action="store_true",
        default=True,
        help="Whether to run failure case extraction on validation set (default: True).",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=50,
        help="Maximum failure case samples to visualize and save (default: 50).",
    )

    return parser.parse_args()


def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


def run_failure_analysis(model, data_cfg, split, failure_dir: Path, max_samples: int = 50, conf_thresh: float = 0.25, iou_thresh: float = 0.5):
    """Analyze predictions against ground truth labels and record failure cases."""
    failure_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["false_positives", "false_negatives", "low_confidence", "class_confusions"]:
        (failure_dir / sub).mkdir(parents=True, exist_ok=True)

    base_path = Path(data_cfg.get("path", "."))
    images_rel = data_cfg.get(split, f"images/{split}")
    images_dir = (base_path / images_rel).resolve() if not Path(images_rel).is_absolute() else Path(images_rel)

    # Resolve labels directory corresponding to images directory
    labels_rel = images_rel.replace("images", "labels")
    labels_dir = (base_path / labels_rel).resolve() if not Path(labels_rel).is_absolute() else Path(labels_rel)

    if not images_dir.exists():
        print(f"[WARN] Images directory not found for failure analysis: {images_dir}")
        return []

    print(f"[INFO] Collecting failure cases from {images_dir}...")
    failure_records = []
    saved_count = 0

    image_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    for img_path in image_files:
        if saved_count >= max_samples:
            break

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        gt_boxes.append({
                            "class_id": cls_id,
                            "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                            "bbox_yolo": [xc, yc, w, h]
                        })

        # Run prediction
        preds = model.predict(source=str(img_path), conf=0.1, verbose=False)
        pred_boxes = []
        if preds and len(preds) > 0 and preds[0].boxes is not None:
            boxes = preds[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                xywhn = boxes.xywhn[i].cpu().numpy().tolist()
                pred_boxes.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                    "confidence": conf,
                    "xyxy": xyxy,
                    "bbox_yolo": xywhn
                })

        # Identify failures
        # 1. Low confidence (< conf_thresh)
        for pb in pred_boxes:
            if pb["confidence"] < conf_thresh:
                failure_records.append({
                    "image": img_path.name,
                    "type": "low_confidence",
                    "pred": pb,
                })

        # 2. Match GT to Preds using normalized coordinates
        matched_preds = set()
        for gt in gt_boxes:
            gt_xc, gt_yc, gt_w, gt_h = gt["bbox_yolo"]
            gt_xyxy = [gt_xc - gt_w / 2, gt_yc - gt_h / 2, gt_xc + gt_w / 2, gt_yc + gt_h / 2]
            best_iou = 0.0
            best_p_idx = -1
            for p_idx, pb in enumerate(pred_boxes):
                p_xc, p_yc, p_w, p_h = pb["bbox_yolo"]
                p_xyxy = [p_xc - p_w / 2, p_yc - p_h / 2, p_xc + p_w / 2, p_yc + p_h / 2]
                iou = compute_iou(gt_xyxy, p_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx

            if best_iou >= iou_thresh and best_p_idx >= 0:
                matched_preds.add(best_p_idx)
                pb = pred_boxes[best_p_idx]
                if pb["class_id"] != gt["class_id"]:
                    failure_records.append({
                        "image": img_path.name,
                        "type": "class_confusion",
                        "gt_class": gt["class_name"],
                        "pred_class": pb["class_name"],
                        "confidence": pb["confidence"],
                        "iou": best_iou
                    })
            else:
                failure_records.append({
                    "image": img_path.name,
                    "type": "false_negative",
                    "gt_class": gt["class_name"],
                    "gt_box": gt["bbox_yolo"]
                })

        for p_idx, pb in enumerate(pred_boxes):
            if p_idx not in matched_preds and pb["confidence"] >= conf_thresh:
                failure_records.append({
                    "image": img_path.name,
                    "type": "false_positive",
                    "pred_class": pb["class_name"],
                    "confidence": pb["confidence"],
                    "pred_box": pb["bbox_yolo"]
                })

        saved_count += 1

    summary_file = failure_dir / "failure_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(failure_records, f, indent=2)

    print(f"[INFO] Failure cases recorded: {len(failure_records)} cases logged to {summary_file}")
    return failure_records


def main():
    args = parse_args()

    print("=" * 60)
    print(" RecoverOps: RT-DETR-L Evaluation & Metrics Reporting")
    print("=" * 60)
    print(f"Model Checkpoint : {args.weights}")
    print(f"Dataset YAML     : {args.data}")
    print(f"Split            : {args.split}")
    print(f"Image Size       : {args.imgsz}")
    print(f"Confidence Thresh: {args.conf}")
    print(f"IoU Threshold    : {args.iou}")
    print("=" * 60)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] Model weights file not found: {weights_path}")
        print(f"Please provide valid checkpoint path via --weights")
        return

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Initialize model with trained weights
    model = RTDETR(str(weights_path))

    # Run standard validation
    print(f"[INFO] Running validation on '{args.split}' split...")
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )

    # Extract metrics safely
    box_metrics = metrics.box
    overall_p = float(box_metrics.mp) if hasattr(box_metrics, "mp") else 0.0
    overall_r = float(box_metrics.mr) if hasattr(box_metrics, "mr") else 0.0
    overall_map50 = float(box_metrics.map50) if hasattr(box_metrics, "map50") else 0.0
    overall_map50_95 = float(box_metrics.map) if hasattr(box_metrics, "map") else 0.0

    names = data_cfg.get("names", {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    per_class_results = {}
    if hasattr(box_metrics, "p") and hasattr(box_metrics, "r") and hasattr(box_metrics, "ap50") and hasattr(box_metrics, "ap"):
        p_list = box_metrics.p.tolist() if hasattr(box_metrics.p, "tolist") else list(box_metrics.p)
        r_list = box_metrics.r.tolist() if hasattr(box_metrics.r, "tolist") else list(box_metrics.r)
        ap50_list = box_metrics.ap50.tolist() if hasattr(box_metrics.ap50, "tolist") else list(box_metrics.ap50)
        ap_list = box_metrics.ap.tolist() if hasattr(box_metrics.ap, "tolist") else list(box_metrics.ap)

        for idx, cls_name in names.items():
            idx_int = int(idx)
            per_class_results[cls_name] = {
                "precision": float(p_list[idx_int]) if idx_int < len(p_list) else 0.0,
                "recall": float(r_list[idx_int]) if idx_int < len(r_list) else 0.0,
                "mAP50": float(ap50_list[idx_int]) if idx_int < len(ap50_list) else 0.0,
                "mAP50-95": float(ap_list[idx_int]) if idx_int < len(ap_list) else 0.0,
            }

    evaluation_report = {
        "model_weights": str(weights_path),
        "split": args.split,
        "dataset_yaml": str(data_path),
        "overall_metrics": {
            "precision": overall_p,
            "recall": overall_r,
            "mAP50": overall_map50,
            "mAP50-95": overall_map50_95,
        },
        "per_class_metrics": per_class_results,
    }

    # Display Metrics Table
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Precision : {overall_p:.4f}")
    print(f"Overall Recall    : {overall_r:.4f}")
    print(f"Overall mAP50     : {overall_map50:.4f}")
    print(f"Overall mAP50-95  : {overall_map50_95:.4f}")
    print("-" * 70)
    print(f"{'Class':<16} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
    print("-" * 70)
    for cls_name, m in per_class_results.items():
        print(f"{cls_name:<16} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['mAP50']:<10.4f} | {m['mAP50-95']:<10.4f}")
    print("=" * 70)

    # Save JSON report
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    print(f"[INFO] JSON report written to: {out_json_path}")

    # Save Markdown report
    out_md_path = Path(args.output_md)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    md_content = f"""# RT-DETR-L Baseline Evaluation Report

## Overall Performance
- **Precision**: {overall_p:.4f}
- **Recall**: {overall_r:.4f}
- **mAP@0.50**: {overall_map50:.4f}
- **mAP@0.50:0.95**: {overall_map50_95:.4f}

## Per-Class Breakdown
| Class | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|
"""
    for cls_name, m in per_class_results.items():
        md_content += f"| `{cls_name}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['mAP50']:.4f} | {m['mAP50-95']:.4f} |\n"

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[INFO] Markdown report written to: {out_md_path}")

    # Failure case collection infrastructure
    if args.collect_failures:
        failure_dir = Path(args.failure_dir)
        run_failure_analysis(
            model=model,
            data_cfg=data_cfg,
            split=args.split,
            failure_dir=failure_dir,
            max_samples=args.max_failures,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
        )


if __name__ == "__main__":
    main()
