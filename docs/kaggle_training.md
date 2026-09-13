# Kaggle GPU Training & Evaluation Guide for RT-DETR-L Baseline

This document provides end-to-end, reproducible instructions to train and evaluate the **RT-DETR-L** baseline on a Kaggle GPU instance (NVIDIA Tesla T4 or P100).

---

## 1. Architecture & Data Flow Overview

- **Kaggle Input (Read-Only)**:
  - Mounted processed dataset folder containing `images/`, `labels/`, and `dataset.yaml` located under `/kaggle/input/<dataset-folder-name>/`.
  - Checkpoints / Pretrained weights (`rtdetr-l.pt` downloaded automatically or attached).
- **Kaggle Working Directory (Read/Write)**:
  - Repository code: `/kaggle/working/RecoverOps/`
  - Training runs, model checkpoints (`best.pt`, `last.pt`), logs, and failure case outputs: `/kaggle/working/RecoverOps/experiments/baseline/runs/` and `/kaggle/working/RecoverOps/results/`.
- **Kaggle Output (Exportable)**:
  - Final best model weights (`best.pt`), evaluation reports (`baseline_evaluation.json`, `baseline_evaluation.md`), curves (`results.png`, `confusion_matrix.png`), and failure cases.

> [!WARNING]
> **Security & Credentials**:
> - Never upload `.env`, API keys, or DocILE access tokens to Kaggle datasets or notebooks.
> - The processed dataset contains only extracted page images and YOLO format `.txt` label annotations.

---

## 2. Step-by-Step Kaggle Setup

### Step 1: Create the Kaggle Dataset
1. Zip or upload the local processed dataset directory:
   `data/processed/invoice_detection/` containing:
   ```text
   invoice_detection/
   ├── dataset.yaml
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```
2. On Kaggle, navigate to **Datasets -> New Dataset**.
3. Name your dataset (e.g., `invoice-detection-docile`) and upload the folder.

### Step 2: Create Kaggle Notebook & Attach Dataset
1. Create a new Python Notebook on Kaggle.
2. In the right-hand panel under **Input -> Add Input**, search for your uploaded dataset and click **Add**.
3. Under **Notebook options -> Accelerator**, select **GPU T4 x2** or **GPU P100**.
4. Set **Internet** to **On** (required for downloading `rtdetr-l.pt` and installing pip packages).

### Step 3: Determine the Mounted Dataset Path
In a notebook cell, dynamically locate the dataset path:
```python
import os
from pathlib import Path

# Scan Kaggle input directory
input_dirs = [d for d in Path("/kaggle/input").iterdir() if d.is_dir()]
print("Available inputs:", input_dirs)

# Locate dataset.yaml
data_yaml_candidates = list(Path("/kaggle/input").glob("**/dataset.yaml"))
if not data_yaml_candidates:
    raise FileNotFoundError("dataset.yaml not found under /kaggle/input!")

dataset_yaml_path = data_yaml_candidates[0]
dataset_root = dataset_yaml_path.parent
print(f"Dataset root: {dataset_root}")
print(f"Dataset YAML: {dataset_yaml_path}")
```

### Step 4: Clone / Set Up the RecoverOps Code
In a notebook cell:
```bash
%cd /kaggle/working
!git clone https://github.com/<your-org-or-user>/RecoverOps.git
%cd /kaggle/working/RecoverOps
```
*(Or upload the `src/` and `experiments/` directories directly into `/kaggle/working/RecoverOps`)*.

### Step 5: Install Pinned Dependencies
Ensure the exact pinned version of Ultralytics is installed:
```bash
!pip install --upgrade pip
!pip install ultralytics==8.3.0
```

Verify installed versions:
```python
import torch, ultralytics
print(f"PyTorch Version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
print(f"Ultralytics Version: {ultralytics.__version__}")
```

### Step 6: Generate Runtime dataset.yaml (Adjusted for Kaggle Path)
Ensure `dataset.yaml` points to the Kaggle input directory:
```python
import yaml

with open(dataset_yaml_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Point path directly to the Kaggle read-only input folder
cfg['path'] = str(dataset_root)
cfg['train'] = 'images/train'
cfg['val'] = 'images/val'

runtime_yaml = Path("/kaggle/working/runtime_dataset.yaml")
with open(runtime_yaml, 'w') as f:
    yaml.dump(cfg, f)

print(f"Runtime dataset YAML written to: {runtime_yaml}")
```

---

## 3. Training Execution

Run the baseline training script:

```bash
python src/training/train_rtdetr.py \
  --data /kaggle/working/runtime_dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --seed 42 \
  --project /kaggle/working/RecoverOps/experiments/baseline/runs \
  --name rtdetr_l_baseline
```

---

## 4. Evaluation & Failure Case Analysis

After training completes, evaluate the best checkpoint on the validation set:

```bash
python src/training/evaluate_rtdetr.py \
  --weights /kaggle/working/RecoverOps/experiments/baseline/runs/rtdetr_l_baseline/weights/best.pt \
  --data /kaggle/working/runtime_dataset.yaml \
  --split val \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --output-json /kaggle/working/RecoverOps/results/baseline_evaluation.json \
  --output-md /kaggle/working/RecoverOps/results/baseline_evaluation.md \
  --failure-dir /kaggle/working/RecoverOps/results/failure_cases \
  --collect-failures
```

---

## 5. Saving and Exporting Kaggle Artifacts

To download artifacts from Kaggle:
```python
import shutil

# Zip the run artifacts and failure cases
shutil.make_archive('/kaggle/working/baseline_artifacts', 'zip', '/kaggle/working/RecoverOps/experiments/baseline/runs/rtdetr_l_baseline')
shutil.make_archive('/kaggle/working/evaluation_results', 'zip', '/kaggle/working/RecoverOps/results')

print("Artifacts zipped and ready for download from /kaggle/working/")
```
