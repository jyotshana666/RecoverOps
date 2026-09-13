import os
from pathlib import Path
import json

def validate_dataset():
    out_dir = Path("data/processed/invoice_detection")
    
    report = {
        "valid": True,
        "errors": [],
        "counts": {
            "images": 0,
            "labels": 0,
            "boxes": 0
        }
    }
    
    # Check splits
    for split in ["train", "val"]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        
        if not img_dir.exists() or not lbl_dir.exists():
            report["errors"].append(f"Directory missing for split {split}")
            report["valid"] = False
            continue
            
        images = list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        
        img_stems = {img.stem for img in images}
        lbl_stems = {lbl.stem for lbl in labels}
        
        if img_stems != lbl_stems:
            report["errors"].append(f"Mismatch between images and labels in {split}")
            report["valid"] = False
            
        report["counts"]["images"] += len(images)
        report["counts"]["labels"] += len(labels)
        
        for lbl in labels:
            with open(lbl, "r") as f:
                lines = f.read().strip().split("\n")
                
            seen_boxes = set()
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    report["errors"].append(f"Malformed line in {lbl.name}: '{line}'")
                    report["valid"] = False
                    continue
                    
                cls_id, xc, yc, w, h = parts
                
                try:
                    cls_id = int(cls_id)
                    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                except ValueError:
                    report["errors"].append(f"Non-numeric values in {lbl.name}: '{line}'")
                    report["valid"] = False
                    continue
                    
                if cls_id not in range(5):
                    report["errors"].append(f"Invalid class ID {cls_id} in {lbl.name}")
                    report["valid"] = False
                    
                if w <= 0 or h <= 0:
                    report["errors"].append(f"Zero or negative area box in {lbl.name}: '{line}'")
                    report["valid"] = False
                    
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                    report["errors"].append(f"Center out of bounds in {lbl.name}: '{line}'")
                    report["valid"] = False
                    
                report["counts"]["boxes"] += 1
                
                if line in seen_boxes:
                    report["errors"].append(f"Duplicate box in {lbl.name}: '{line}'")
                    report["valid"] = False
                seen_boxes.add(line)
                
    Path("results").mkdir(exist_ok=True)
    with open("results/label_validation.json", "w") as f:
        json.dump(report, f, indent=2)
        
    md = f"""# Label Validation Report
- **Valid**: {report['valid']}
- **Total Images**: {report['counts']['images']}
- **Total Labels**: {report['counts']['labels']}
- **Total Boxes**: {report['counts']['boxes']}

## Errors
"""
    if report["errors"]:
        # Only show first 50 errors
        for e in report["errors"][:50]:
            md += f"- {e}\n"
        if len(report["errors"]) > 50:
            md += f"- ... and {len(report['errors']) - 50} more errors\n"
    else:
        md += "No errors found. All labels are valid.\n"
        
    with open("results/label_validation.md", "w") as f:
        f.write(md)
        
    if not report["valid"]:
        print("Validation FAILED. Check results/label_validation.md")
        exit(1)
    else:
        print("Validation PASSED.")

if __name__ == "__main__":
    validate_dataset()
