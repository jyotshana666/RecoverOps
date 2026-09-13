import json
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw, ImageFont

def main():
    base_dir = Path("data/raw/docile")
    ann_dir = base_dir / "annotations"
    pdf_dir = base_dir / "pdfs"
    out_dir = Path("results/dataset_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_classes = {"amount_due", "date_due", "document_id", "date_issue", "vendor_name"}
    
    # Get all annotation files, sort for determinism
    ann_files = sorted(list(ann_dir.glob("*.json")))
    
    samples_generated = 0
    
    # Simple predefined colors
    colors = {
        "amount_due": "red",
        "date_due": "blue",
        "document_id": "green",
        "date_issue": "purple",
        "vendor_name": "orange"
    }
    
    for ann_file in ann_files:
        if samples_generated >= 20:
            break
            
        doc_id = ann_file.stem
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        
        if not pdf_path.exists():
            continue
            
        with open(ann_file, "r") as f:
            ann_data = json.load(f)
            
        target_fields = [f for f in ann_data.get("field_extractions", []) if f.get("fieldtype") in target_classes]
        
        # If no target fields exist, maybe skip or generate anyway? We will generate anyway to show actual samples.
        # But let's only pick documents that have at least ONE target field so the visualizer isn't just blank pages.
        if not target_fields:
            continue
            
        try:
            pdf_doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            print(f"Failed to open {pdf_path}: {e}")
            continue
            
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            draw = ImageDraw.Draw(img)
            
            # Fields for this page
            page_fields = [f for f in target_fields if f.get("page", 0) == page_idx]
            
            for field in page_fields:
                ft = field["fieldtype"]
                bbox = field["bbox"]
                
                # DocILE bboxes are [left, top, right, bottom] relative to page (0 to 1)
                left = bbox[0] * img.width
                top = bbox[1] * img.height
                right = bbox[2] * img.width
                bottom = bbox[3] * img.height
                
                color = colors.get(ft, "black")
                
                # Draw rectangle
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                
                # Draw label background and text
                text = f"{ft}: {field.get('text', '')}"
                draw.text((left, top - 15), text, fill=color)
                
            out_path = out_dir / f"{doc_id}_page_{page_idx}.png"
            img.save(out_path)
            
        pdf_doc.close()
        samples_generated += 1
        print(f"Generated sample {samples_generated}: {doc_id}")

if __name__ == "__main__":
    main()
