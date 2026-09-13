import json
import os
from pathlib import Path
from collections import defaultdict, Counter

def main():
    base_dir = Path("data/raw/docile")
    ann_dir = base_dir / "annotations"
    
    # Load splits
    train_ids = set()
    if (base_dir / "train.json").exists():
        with open(base_dir / "train.json") as f:
            train_ids = set(json.load(f))
            
    val_ids = set()
    if (base_dir / "val.json").exists():
        with open(base_dir / "val.json") as f:
            val_ids = set(json.load(f))
            
    trainval_ids = set()
    if (base_dir / "trainval.json").exists():
        with open(base_dir / "trainval.json") as f:
            trainval_ids = set(json.load(f))
            
    target_classes = {"amount_due", "date_due", "document_id", "date_issue", "vendor_name"}
    
    doc_count = 0
    total_pages = 0
    
    split_counts = {"train": 0, "val": 0, "unknown": 0}
    
    fieldtype_counts = Counter()
    target_class_counts = Counter()
    target_doc_counts = Counter() # documents containing each target class
    
    docs_missing_all_targets = 0
    
    # Inspect all annotations
    if not ann_dir.exists():
        print("Annotations directory not found.")
        return
        
    for ann_file in ann_dir.glob("*.json"):
        doc_id = ann_file.stem
        doc_count += 1
        
        if doc_id in train_ids:
            split_counts["train"] += 1
        elif doc_id in val_ids:
            split_counts["val"] += 1
        else:
            split_counts["unknown"] += 1
            
        with open(ann_file, "r") as f:
            data = json.load(f)
            
        total_pages += data.get("metadata", {}).get("page_count", 0)
        
        doc_fields = set()
        for field in data.get("field_extractions", []):
            ft = field.get("fieldtype")
            fieldtype_counts[ft] += 1
            if ft in target_classes:
                target_class_counts[ft] += 1
                doc_fields.add(ft)
                
        for t in doc_fields:
            target_doc_counts[t] += 1
            
        if not doc_fields:
            docs_missing_all_targets += 1

    missing_targets = [t for t in target_classes if target_class_counts[t] == 0]

    audit_result = {
        "dataset_structure": {
            "total_documents": doc_count,
            "total_pages": total_pages,
            "splits": split_counts
        },
        "field_statistics": {
            "available_field_types": list(fieldtype_counts.keys()),
            "all_field_counts": dict(fieldtype_counts),
        },
        "target_class_statistics": {
            "target_classes": list(target_classes),
            "occurrences": dict(target_class_counts),
            "documents_containing": dict(target_doc_counts),
            "missing_target_classes": missing_targets,
            "documents_missing_all_targets": docs_missing_all_targets
        }
    }
    
    with open("results/data_audit.json", "w") as f:
        json.dump(audit_result, f, indent=2)
        
    # Generate Markdown
    md = f"""# DocILE Dataset Audit

## Dataset Facts
- **Total Documents**: {doc_count}
- **Total Pages**: {total_pages}
- **Train split**: {split_counts['train']} documents
- **Val split**: {split_counts['val']} documents
- **Other/Unknown**: {split_counts['unknown']} documents

## Selected Target Subset
Our task targets the following fields:
- `amount_due`
- `date_due`
- `document_id`
- `date_issue`
- `vendor_name`

### Target Class Counts (Occurrences)
"""
    for t in sorted(target_classes):
        md += f"- **{t}**: {target_class_counts[t]}\n"
        
    md += "\n### Documents Containing Target Class\n"
    for t in sorted(target_classes):
        md += f"- **{t}**: {target_doc_counts[t]} documents\n"
        
    md += f"\n- **Documents completely missing all target classes**: {docs_missing_all_targets}\n"
    if missing_targets:
        md += f"- **Target classes completely missing from dataset**: {', '.join(missing_targets)}\n"
        
    md += "\n## All Observed Field Types\n"
    md += "| Field Type | Count |\n|---|---|\n"
    for ft, count in fieldtype_counts.most_common():
        md += f"| {ft} | {count} |\n"
        
    md += "\n## Assumptions & Limitations\n"
    md += "- Assuming `field_extractions` in JSON contains all relevant document-level annotations.\n"
    md += "- Assuming bounding boxes are normalized [left, top, right, bottom] (needs verification in visualization step).\n"
    md += "- Assuming pages are 0-indexed.\n"

    with open("results/data_audit.md", "w") as f:
        f.write(md)

    print("Audit complete. Saved to results/data_audit.json and results/data_audit.md")

if __name__ == "__main__":
    main()
