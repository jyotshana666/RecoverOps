# Kaggle Dataset Preparation & Upload Procedure

This document provides exact, manual instructions for packaging and creating the Kaggle Dataset from the processed invoice detection dataset.

---

## 1. Exact Local Source Directory
- **Source Path**: `c:\Work\Projects\RecoverOps\data\processed\invoice_detection\`

---

## 2. Exact Kaggle Dataset Directory Structure
The dataset archive to upload MUST maintain the following structure:
```text
invoice_detection/
├── dataset.yaml
├── images/
│   ├── train/        # 6,759 PNG images
│   └── val/          # 635 PNG images
└── labels/
    ├── train/        # 6,759 YOLO format .txt files
    └── val/          # 635 YOLO format .txt files
```

---

## 3. Recommended Dataset Title & Slug
- **Title**: `DocILE Invoice Detection RT-DETR`
- **Recommended Slug**: `docile-invoice-detection-rt-detr` (or `invoice-detection-docile`)

---

## 4. What Files MUST Be Included
- **`dataset.yaml`**: The YOLO/RT-DETR class definition configuration file.
- **`images/train/*.png`**: All 6,759 training page images rendered at 150 DPI.
- **`images/val/*.png`**: All 635 validation page images rendered at 150 DPI.
- **`labels/train/*.txt`**: All 6,759 training annotation files (YOLO normalized `<class_id> <x_center> <y_center> <w> <h>`).
- **`labels/val/*.txt`**: All 635 validation annotation files.

---

## 5. What MUST NOT Be Included
- ❌ Raw DocILE PDFs (`data/raw/docile/pdfs/`)
- ❌ Raw DocILE JSON annotations (`data/raw/docile/annotations/`)
- ❌ DocILE Access Tokens or API credentials
- ❌ Environment files (`.env`, `.env.local`)
- ❌ Pretrained weights or checkpoints (`.pt`, `.pth`, `.bin`)
- ❌ Intermediate cache files or python bytecode (`__pycache__`, `.pytest_cache`)

---

## 6. Manual Upload Instructions via Kaggle Web UI

1. Open your browser and go to [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets).
2. Click the **+ New Dataset** button in the top right.
3. In the upload dialog:
   - Option A: Drag and drop the `invoice_detection` folder.
   - Option B: Zip `data/processed/invoice_detection` into `invoice_detection.zip` and upload the archive.
4. Set the **Dataset Title** to `DocILE Invoice Detection RT-DETR`.
5. Set Visibility to **Private** (or Public according to your preference).
6. Click **Create**.

---

## 7. Verification After Attaching to a Kaggle Notebook

Run this Python verification block in the first cell of your Kaggle notebook:

```python
from pathlib import Path
import yaml

# Dynamically locate dataset
yaml_candidates = list(Path("/kaggle/input").glob("**/dataset.yaml"))
assert len(yaml_candidates) > 0, "ERROR: dataset.yaml not found under /kaggle/input!"

dataset_yaml = yaml_candidates[0]
dataset_root = dataset_yaml.parent

train_imgs = list((dataset_root / "images" / "train").glob("*.png"))
val_imgs = list((dataset_root / "images" / "val").glob("*.png"))
train_lbls = list((dataset_root / "labels" / "train").glob("*.txt"))
val_lbls = list((dataset_root / "labels" / "val").glob("*.txt"))

print("=== Kaggle Dataset Verification ===")
print(f"Dataset root : {dataset_root}")
print(f"Train images : {len(train_imgs)} (Expected: 6759)")
print(f"Val images   : {len(val_imgs)} (Expected: 635)")
print(f"Train labels : {len(train_lbls)} (Expected: 6759)")
print(f"Val labels   : {len(val_lbls)} (Expected: 635)")

assert len(train_imgs) == 6759, f"Train images mismatch: {len(train_imgs)}"
assert len(val_imgs) == 635, f"Val images mismatch: {len(val_imgs)}"
assert len(train_lbls) == 6759, f"Train labels mismatch: {len(train_lbls)}"
assert len(val_lbls) == 635, f"Val labels mismatch: {len(val_lbls)}"

# Verify 1:1 image and label pair alignment
train_img_stems = set(p.stem for p in train_imgs)
train_lbl_stems = set(p.stem for p in train_lbls)
val_img_stems = set(p.stem for p in val_imgs)
val_lbl_stems = set(p.stem for p in val_lbls)

assert train_img_stems == train_lbl_stems, "ERROR: Train image and label stems do not match!"
assert val_img_stems == val_lbl_stems, "ERROR: Val image and label stems do not match!"

print("[SUCCESS] All 7,394 images and labels verified 1:1 with zero mismatches.")
```
