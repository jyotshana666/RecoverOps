import json
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import pymupdf
from config import CLASS_MAPPING

def main():
    base_dir = Path("data/raw/docile")
    ann_dir = base_dir / "annotations"
    pdf_dir = base_dir / "pdfs"
    
    out_dir = Path("data/processed/invoice_detection")
    img_train_dir = out_dir / "images" / "train"
    img_val_dir = out_dir / "images" / "val"
    lbl_train_dir = out_dir / "labels" / "train"
    lbl_val_dir = out_dir / "labels" / "val"
    
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load splits
    train_ids = set()
    if (base_dir / "train.json").exists():
        with open(base_dir / "train.json") as f:
            train_ids = set(json.load(f))
            
    val_ids = set()
    if (base_dir / "val.json").exists():
        with open(base_dir / "val.json") as f:
            val_ids = set(json.load(f))
            
    # Verify overlap
    overlap = train_ids.intersection(val_ids)
    if overlap:
        print(f"FATAL: Train and validation sets overlap. {len(overlap)} documents found in both.")
        exit(1)
        
    # Traceability
    traceability = []
        
    report = {
        "dataset": {
            "source_dataset": "DocILE",
            "source_train_docs": len(train_ids),
            "source_val_docs": len(val_ids),
            "processed_train_images": 0,
            "processed_val_images": 0,
            "overlap_count": 0
        },
        "annotations": {
            "total_source_target_annotations": 0,
            "converted_annotations": 0,
            "skipped_annotations": 0,
            "skipped_documents": 0,
            "annotations_per_class": Counter()
        },
        "classes": {cls: {"source_occurrence": 0, "converted": 0, "skipped": 0} for cls in CLASS_MAPPING},
        "errors": {
            "malformed_bbox_count": 0,
            "out_of_range_bbox_count": 0,
            "zero_area_bbox_count": 0,
            "missing_image_page_references": 0,
            "duplicate_annotations": 0
        }
    }
    
    # Generate yaml
    yaml_content = f"""path: ../data/processed/invoice_detection
train: images/train
val: images/val

names:
"""
    for k, v in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        yaml_content += f"  {v}: {k}\n"
        
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    ann_files = list(ann_dir.glob("*.json"))
    for ann_file in ann_files:
        doc_id = ann_file.stem
        
        if doc_id in train_ids:
            split = "train"
            img_dest = img_train_dir
            lbl_dest = lbl_train_dir
        elif doc_id in val_ids:
            split = "val"
            img_dest = img_val_dir
            lbl_dest = lbl_val_dir
        else:
            # Not in train/val splits we care about, skip
            continue
            
        with open(ann_file, "r") as f:
            data = json.load(f)
            
        target_fields = [f for f in data.get("field_extractions", []) if f.get("fieldtype") in CLASS_MAPPING]
        if not target_fields:
            report["annotations"]["skipped_documents"] += 1
            # Still process image if we want background images, but let's just create empty txt
            
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if not pdf_path.exists():
            report["errors"]["missing_image_page_references"] += len(target_fields)
            for f in target_fields:
                ft = f.get("fieldtype")
                report["classes"][ft]["source_occurrence"] += 1
                report["classes"][ft]["skipped"] += 1
                report["annotations"]["skipped_annotations"] += 1
                report["annotations"]["total_source_target_annotations"] += 1
            continue
            
        try:
            pdf_doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            continue
            
        page_count = len(pdf_doc)
        
        # Group targets by page
        page_targets = defaultdict(list)
        for f in target_fields:
            ft = f.get("fieldtype")
            page_idx = f.get("page")
            
            report["annotations"]["total_source_target_annotations"] += 1
            report["classes"][ft]["source_occurrence"] += 1
            
            if page_idx is None or page_idx < 0 or page_idx >= page_count:
                report["errors"]["missing_image_page_references"] += 1
                report["classes"][ft]["skipped"] += 1
                report["annotations"]["skipped_annotations"] += 1
                continue
                
            bbox = f.get("bbox")
            if not bbox or len(bbox) != 4:
                report["errors"]["malformed_bbox_count"] += 1
                report["classes"][ft]["skipped"] += 1
                report["annotations"]["skipped_annotations"] += 1
                continue
                
            left, top, right, bottom = bbox
            if left >= right or top >= bottom:
                report["errors"]["zero_area_bbox_count"] += 1
                report["classes"][ft]["skipped"] += 1
                report["annotations"]["skipped_annotations"] += 1
                continue
                
            if not (0.0 <= left <= 1.0 and 0.0 <= top <= 1.0 and 0.0 <= right <= 1.0 and 0.0 <= bottom <= 1.0):
                # Clamp or skip? Specification says out_of_range. Let's clamp mildly or report.
                report["errors"]["out_of_range_bbox_count"] += 1
                # Skip for safety if outside wildly, but small floating precision might be > 1.0
                left = max(0.0, min(1.0, left))
                right = max(0.0, min(1.0, right))
                top = max(0.0, min(1.0, top))
                bottom = max(0.0, min(1.0, bottom))
                if left >= right or top >= bottom:
                    report["classes"][ft]["skipped"] += 1
                    report["annotations"]["skipped_annotations"] += 1
                    continue
                    
            page_targets[page_idx].append({
                "class_id": CLASS_MAPPING[ft],
                "class_name": ft,
                "bbox": [left, top, right, bottom],
                "text": f.get("text")
            })
            
        # Write images and labels per page
        for page_idx in range(page_count):
            img_name = f"{doc_id}_{page_idx}.png"
            lbl_name = f"{doc_id}_{page_idx}.txt"
            
            img_path = img_dest / img_name
            lbl_path = lbl_dest / lbl_name
            
            # Save Image (only if we have targets? Wait, if we keep all pages we improve negative examples)
            # Let's save all pages to match dataset structure
            try:
                page = pdf_doc[page_idx]
                pix = page.get_pixmap(dpi=150)
                pix.save(str(img_path))
                if split == "train":
                    report["dataset"]["processed_train_images"] += 1
                else:
                    report["dataset"]["processed_val_images"] += 1
            except Exception as e:
                print(f"Error rendering page {page_idx} of {doc_id}: {e}")
                continue
                
            # Deduplicate targets per page
            unique_targets = set()
            converted = []
            
            for t in page_targets[page_idx]:
                cls_id = t["class_id"]
                l, top, r, b = t["bbox"]
                
                # YOLO format
                x_center = (l + r) / 2.0
                y_center = (top + b) / 2.0
                width = r - l
                height = b - top
                
                # Format string
                yolo_str = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                
                if yolo_str in unique_targets:
                    report["errors"]["duplicate_annotations"] += 1
                    report["classes"][t["class_name"]]["skipped"] += 1
                    report["annotations"]["skipped_annotations"] += 1
                else:
                    unique_targets.add(yolo_str)
                    converted.append(yolo_str)
                    
                    report["classes"][t["class_name"]]["converted"] += 1
                    report["annotations"]["converted_annotations"] += 1
                    report["annotations"]["annotations_per_class"][t["class_name"]] += 1
                    
                    # Traceability (saving a few for audit)
                    if len(traceability) < 1000:  # limit to not blow up memory
                        traceability.append({
                            "source_document_id": doc_id,
                            "source_page": page_idx,
                            "source_field_type": t["class_name"],
                            "source_bbox": t["bbox"],
                            "target_image": img_name,
                            "target_label": lbl_name,
                            "target_class_id": cls_id,
                            "target_yolo_str": yolo_str
                        })
                        
            # Write label file even if empty (background image)
            with open(lbl_path, "w") as f:
                f.write("\n".join(converted) + "\n")
                
        pdf_doc.close()
        
    Path("results").mkdir(exist_ok=True)
    with open("results/conversion_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    with open("results/annotation_traceability.json", "w") as f:
        json.dump(traceability, f, indent=2)
        
    # Generate Markdown
    md = f"""# Conversion Report
## Dataset
- **Source**: DocILE
- **Source Train Docs**: {report['dataset']['source_train_docs']}
- **Source Val Docs**: {report['dataset']['source_val_docs']}
- **Overlap Count**: {report['dataset']['overlap_count']}
- **Processed Train Images/Pages**: {report['dataset']['processed_train_images']}
- **Processed Val Images/Pages**: {report['dataset']['processed_val_images']}

## Annotations
- **Total Source Target Annotations**: {report['annotations']['total_source_target_annotations']}
- **Converted Annotations**: {report['annotations']['converted_annotations']}
- **Skipped Annotations**: {report['annotations']['skipped_annotations']}
- **Skipped Documents (No targets)**: {report['annotations']['skipped_documents']}

## Class Breakdown
| Class | Source Occurrences | Converted | Skipped |
|---|---|---|---|
"""
    for cls, stats in report["classes"].items():
        md += f"| {cls} | {stats['source_occurrence']} | {stats['converted']} | {stats['skipped']} |\n"
        
    md += "\n## Conversion Errors\n"
    for err, count in report["errors"].items():
        md += f"- **{err}**: {count}\n"
        
    with open("results/conversion_report.md", "w") as f:
        f.write(md)
        
    print("Conversion complete.")

if __name__ == "__main__":
    main()
