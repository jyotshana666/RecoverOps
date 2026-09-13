# Kaggle GPU Training & Smoke Test Guide for RT-DETR-L Baseline

This guide provides the complete, self-contained notebook execution workflow for training and verifying the **RT-DETR-L** baseline on Kaggle GPU instances (NVIDIA Tesla T4 or P100).

---

## 1. Notebook Configuration & Environment Setup

### Cell 1: Verify Hardware & GPU Environment
```python
import sys
import torch

print("=== Hardware & Environment Checks ===")
print(f"Python Version : {sys.version.split()[0]}")
print(f"CUDA Available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    print(f"GPU Device     : {device_name} ({device_count} GPU(s) detected)")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    raise RuntimeError("ERROR: GPU accelerator is not enabled. Please enable GPU in Notebook Settings (T4 x2 or P100).")
```

### Cell 2: Install Pinned Dependencies & Verify Ultralytics Version
```bash
%%bash
pip install --upgrade pip -q
pip install ultralytics==8.3.0 pyyaml==6.0.3 -q
```

```python
import ultralytics
print(f"Ultralytics Version : {ultralytics.__version__}")
assert ultralytics.__version__ == "8.3.0", f"Expected ultralytics==8.3.0 but got {ultralytics.__version__}"
```

---

## 2. Clone Repository & Setup Working Directory

### Cell 3: Clone RecoverOps Repository
```bash
%%bash
cd /kaggle/working
if [ ! -d "/kaggle/working/RecoverOps" ]; then
    git clone https://github.com/jyotshana666/RecoverOps.git
fi
cd /kaggle/working/RecoverOps
git status
```

---

## 3. Dataset Discovery & Runtime YAML Generation

### Cell 4: Locate Attached Dataset & Verify Integrity
```python
from pathlib import Path
import yaml

# Dynamically search /kaggle/input for dataset.yaml
yaml_candidates = list(Path("/kaggle/input").glob("**/dataset.yaml"))
if not yaml_candidates:
    raise FileNotFoundError("Could not find dataset.yaml in /kaggle/input. Please ensure your dataset is attached.")

dataset_yaml_path = yaml_candidates[0]
dataset_root = dataset_yaml_path.parent
print(f"Located Dataset Root : {dataset_root}")

# Verify images and labels
train_imgs = list((dataset_root / "images" / "train").glob("*.png"))
val_imgs = list((dataset_root / "images" / "val").glob("*.png"))
train_lbls = list((dataset_root / "labels" / "train").glob("*.txt"))
val_lbls = list((dataset_root / "labels" / "val").glob("*.txt"))

print("=== Dataset Counts ===")
print(f"Train Images : {len(train_imgs)} (Expected: 6759)")
print(f"Val Images   : {len(val_imgs)} (Expected: 635)")
print(f"Train Labels : {len(train_lbls)} (Expected: 6759)")
print(f"Val Labels   : {len(val_lbls)} (Expected: 635)")

assert len(train_imgs) == 6759, f"Train image count mismatch: {len(train_imgs)}"
assert len(val_imgs) == 635, f"Val image count mismatch: {len(val_imgs)}"
assert len(train_lbls) == 6759, f"Train label count mismatch: {len(train_lbls)}"
assert len(val_lbls) == 635, f"Val label count mismatch: {len(val_lbls)}"

# Verify 1:1 image and label pair alignment
train_img_stems = set(p.stem for p in train_imgs)
train_lbl_stems = set(p.stem for p in train_lbls)
val_img_stems = set(p.stem for p in val_imgs)
val_lbl_stems = set(p.stem for p in val_lbls)

assert train_img_stems == train_lbl_stems, "Train images and labels do not have 100% paired stems!"
assert val_img_stems == val_lbl_stems, "Validation images and labels do not have 100% paired stems!"

# Create runtime YAML pointing to attached Kaggle dataset
with open(dataset_yaml_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["path"] = str(dataset_root)
cfg["train"] = "images/train"
cfg["val"] = "images/val"

runtime_yaml = Path("/kaggle/working/runtime_dataset.yaml")
with open(runtime_yaml, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f)

print(f"[OK] Runtime dataset YAML created: {runtime_yaml}")
print(f"Classes ({len(cfg['names'])}): {cfg['names']}")
```

---

## 4. Phase 1: 1-Epoch GPU Smoke Test

Before launching the full 50-epoch training, run a single epoch to validate CUDA allocation, model initialization, dataloading, forward/backward passes, loss convergence, and checkpoint saving.

### Cell 5: Run 1-Epoch Smoke Test
```bash
%%bash
cd /kaggle/working/RecoverOps
python src/training/train_rtdetr.py \
  --data /kaggle/working/runtime_dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 4 \
  --seed 42 \
  --project /kaggle/working/RecoverOps/experiments/baseline/runs \
  --name smoke_test
```

### Cell 6: Verify Smoke Test Artifacts & Checkpoint
```python
from pathlib import Path

smoke_run_dir = Path("/kaggle/working/RecoverOps/experiments/baseline/runs/smoke_test")
best_pt = smoke_run_dir / "weights" / "best.pt"
last_pt = smoke_run_dir / "weights" / "last.pt"

print(f"Checking smoke test directory: {smoke_run_dir}")
assert smoke_run_dir.exists(), "Smoke test directory was not created!"
assert last_pt.exists() or best_pt.exists(), "Checkpoint weights (best.pt / last.pt) were not created!"

print("[SUCCESS] 1-Epoch Smoke Test passed completely.")
```

---

## 5. Phase 2: Full 50-Epoch Baseline Training (Run ONLY after Smoke Test Passes)

### Cell 7: Run Full Baseline Training
```bash
%%bash
cd /kaggle/working/RecoverOps
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

## 6. Phase 3: Evaluation & Failure Case Extraction

### Cell 8: Evaluate Best Checkpoint & Collect Failure Cases
```bash
%%bash
cd /kaggle/working/RecoverOps
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

## 7. Phase 4: Package and Download Artifacts

### Cell 9: Archive Outputs
```python
import shutil
from pathlib import Path

shutil.make_archive("/kaggle/working/rtdetr_baseline_run", "zip", "/kaggle/working/RecoverOps/experiments/baseline/runs")
shutil.make_archive("/kaggle/working/baseline_results", "zip", "/kaggle/working/RecoverOps/results")

print("Generated zip archives in /kaggle/working/:")
for p in Path("/kaggle/working").glob("*.zip"):
    print(f"  {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
```
