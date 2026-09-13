import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from config import REVERSE_CLASS_MAPPING

def visualize_converted():
    out_dir = Path("results/converted_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = Path("data/processed/invoice_detection")
    
    # Deterministic seed
    random.seed(42)
    
    colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "purple",
        4: "orange"
    }
    
    for split, count in [("train", 20), ("val", 10)]:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        
        if not img_dir.exists():
            continue
            
        images = sorted(list(img_dir.glob("*.png")))
        
        # We only want images with labels
        valid_images = []
        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                valid_images.append(img)
                
        samples = random.sample(valid_images, min(count, len(valid_images)))
        
        for img_path in samples:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"Failed to open image {img_path}: {e}")
                continue
                
            draw = ImageDraw.Draw(img)
            w, h = img.width, img.height
            
            with open(lbl_path, "r") as f:
                lines = f.read().strip().split("\n")
                
            for line in lines:
                if not line.strip(): continue
                parts = line.strip().split()
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                
                left = (xc - bw / 2) * w
                right = (xc + bw / 2) * w
                top = (yc - bh / 2) * h
                bottom = (yc + bh / 2) * h
                
                color = colors.get(cls_id, "black")
                draw.rectangle([left, top, right, bottom], outline=color, width=3)
                
                cls_name = REVERSE_CLASS_MAPPING.get(cls_id, str(cls_id))
                draw.text((left, top - 15), f"{cls_name} ({img_path.stem})", fill=color)
                
            img.save(out_dir / f"{split}_{img_path.name}")
            
    print("Visual verification complete.")

if __name__ == "__main__":
    visualize_converted()
