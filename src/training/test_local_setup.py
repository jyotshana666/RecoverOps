#!/usr/bin/env python3
"""
Lightweight Local Validation Script for RT-DETR Baseline Setup.
"""

import py_compile
from pathlib import Path
import yaml
from ultralytics import RTDETR


def test_local_setup():
    print("=== 1. Checking Python Syntax ===")
    for p in Path("src").rglob("*.py"):
        py_compile.compile(str(p), doraise=True)
        print(f"  [OK] Syntax compiled: {p}")

    print("\n=== 2. Checking Dataset YAML & Paths ===")
    yaml_path = Path("data/processed/invoice_detection/dataset.yaml")
    assert yaml_path.exists(), f"Missing dataset.yaml at {yaml_path}"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    print(f"  Dataset YAML keys: {list(data_cfg.keys())}")
    print(f"  Class mapping: {data_cfg.get('names')}")

    train_img_dir = Path("data/processed/invoice_detection") / data_cfg["train"]
    val_img_dir = Path("data/processed/invoice_detection") / data_cfg["val"]
    assert train_img_dir.exists(), f"Train directory missing: {train_img_dir}"
    assert val_img_dir.exists(), f"Val directory missing: {val_img_dir}"

    train_count = len(list(train_img_dir.glob("*.png")))
    val_count = len(list(val_img_dir.glob("*.png")))
    print(f"  [OK] Train images verified: {train_count}")
    print(f"  [OK] Val images verified: {val_count}")

    print("\n=== 3. Checking Baseline Config YAML ===")
    with open("experiments/baseline/config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    assert base_cfg["model"]["name"] == "RT-DETR-L"
    assert base_cfg["model"]["checkpoint"] == "rtdetr-l.pt"
    assert base_cfg["training"]["epochs"] == 50
    assert base_cfg["training"]["seed"] == 42
    print(f"  [OK] Baseline Config verified (Model: {base_cfg['model']['name']}, Epochs: {base_cfg['training']['epochs']}, Seed: {base_cfg['training']['seed']})")

    print("\n=== 4. Checking Environment YAML ===")
    with open("experiments/baseline/environment.yaml", "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)
    assert env_cfg["pinned_dependencies"]["ultralytics"] == "8.3.0"
    print(f"  [OK] Environment Config verified (Ultralytics: {env_cfg['pinned_dependencies']['ultralytics']})")

    print("\n=== 5. Testing RT-DETR-L Architecture Initialization ===")
    model = RTDETR("rtdetr-l.yaml")
    print(f"  [OK] RT-DETR-L architecture successfully initialized: {type(model)}")

    print("\n" + "=" * 50)
    print(" ALL LIGHTWEIGHT LOCAL CHECKS PASSED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    test_local_setup()
