# RT-DETR-L Baseline Experiment & Reproducibility Protocol

This document defines the exact specification, environment, hyperparameters, and commands to reproduce the RT-DETR-L object detection baseline for invoice field extraction.

---

## 1. Dataset Specification

- **Source Dataset**: DocILE Benchmark (Document Information Localization and Extraction)
- **Dataset State & Validation**:
  - Total source documents: 5,680
  - Train documents: 5,180
  - Validation documents: 500
  - Train/Val overlap: 0 documents (strictly disjoint splits)
  - Processed images/pages: 6,759 (train) + 635 (val) = 7,394 total pages
  - Total converted annotations: 26,718
  - Malformed / zero-area / duplicate annotations: 0
- **Target Classes & Mapping**:
  | Class ID | Class Name | Total Occurrences |
  |---|---|---|
  | `0` | `amount_due` | 6,125 |
  | `1` | `date_due` | 884 |
  | `2` | `document_id` | 6,141 |
  | `3` | `date_issue` | 6,214 |
  | `4` | `vendor_name` | 7,354 |

---

## 2. Model & Checkpoint Architecture

- **Architecture**: Real-Time DEtection TRansformer Large (`RT-DETR-L`)
- **Framework**: Ultralytics
- **Base Checkpoint**: `rtdetr-l.pt` (Official pretrained weights from Ultralytics)
- **Input Resolution**: `640 x 640`
- **Output**: 5 bounding box classes with confidence scores and normalized coordinates `(xc, yc, w, h)`.

---

## 3. Environment & Pinned Dependencies

- **Python**: `>=3.10`
- **Ultralytics**: `8.3.0`
- **PyTorch**: `>=2.0.0`
- **Torchvision**: `>=0.15.0`
- **OpenCV**: `opencv-python-headless>=4.6.0`
- **Pillow**: `>=10.0.0`
- **PyYAML**: `>=6.0`

Installation:
```bash
pip install ultralytics==8.3.0
```

---

## 4. Training Hyperparameters

All loss gains, warmup parameters, and optimizer settings follow the Ultralytics RT-DETR-L framework defaults:

- **Epochs**: 50
- **Image Size (`imgsz`)**: 640
- **Batch Size (`batch`)**: 8 (per GPU)
- **Optimizer**: `auto` (AdamW)
- **Initial Learning Rate (`lr0`)**: `0.0001`
- **Final Learning Rate Factor (`lrf`)**: `0.01`
- **Momentum**: `0.9`
- **Weight Decay**: `0.0001`
- **Warmup Epochs**: `3.0`
- **Warmup Momentum**: `0.8`
- **Box Loss Gain**: `7.5`
- **Classification Loss Gain**: `0.5`
- **Focal Loss Gamma**: `1.5`
- **Early Stopping Patience**: 15 epochs
- **Random Seed**: `42`

---

## 5. Kaggle GPU Hardware Environment

- **Accelerator**: NVIDIA Tesla T4 (16GB VRAM) or NVIDIA Tesla P100 (16GB VRAM)
- **OS**: Ubuntu Linux (Kaggle Standard Image)
- **Dataloader Workers**: 4

---

## 6. Exact Reproduction Commands

### Training Command:
```bash
python src/training/train_rtdetr.py \
  --data data/processed/invoice_detection/dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --seed 42 \
  --project experiments/baseline/runs \
  --name rtdetr_l_baseline
```

### Evaluation & Metrics Reporting:
```bash
python src/training/evaluate_rtdetr.py \
  --weights experiments/baseline/runs/rtdetr_l_baseline/weights/best.pt \
  --data data/processed/invoice_detection/dataset.yaml \
  --split val \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --output-json results/baseline_evaluation.json \
  --output-md results/baseline_evaluation.md \
  --failure-dir results/failure_cases \
  --collect-failures
```
