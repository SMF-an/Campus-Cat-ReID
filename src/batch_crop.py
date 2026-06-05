#!/usr/bin/env python3
"""
Batch crop cat images using YOLO detection.

Creates a new dataset folder with the same subdirectory (cat name) structure
and saves cropped images (one or multiple per original). Also writes a CSV
manifest with crop metadata.

Usage:
  python src/batch_crop.py --src data --dst data_crops

"""
from pathlib import Path
import argparse
import csv
from PIL import Image
import math

import torch
from ultralytics import YOLO


def load_yolo(model_path: str = None, device: str = "cpu"):
    if model_path:
        return YOLO(model_path).to(device)
    return YOLO("yolov8m.pt").to(device)


def detect_boxes(yolo, image_path):
    # Returns list of (x1,y1,x2,y2,conf,cls)
    if yolo is None:
        return []
    results = yolo(image_path)
    if len(results) == 0:
        return []
    r = results[0]
    boxes = []
    try:
        for box in r.boxes:
            try:
                cls = int(box.cls)
            except Exception:
                cls = int(box.cls.item())
            try:
                conf = float(box.conf)
            except Exception:
                conf = float(box.conf.item())
            # xyxy may be tensor with shape (1,4) or a flat list
            try:
                coords = box.xyxy[0].tolist()
            except Exception:
                coords = list(box.xyxy.tolist())
            x1, y1, x2, y2 = coords[:4]
            boxes.append((float(x1), float(y1), float(x2), float(y2), conf, cls))
    except Exception:
        return []
    return boxes


def padded_bbox(x1, y1, x2, y2, pad_ratio, img_w, img_h):
    w = x2 - x1
    h = y2 - y1
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    nx1 = max(0, math.floor(x1 - pad_w))
    ny1 = max(0, math.floor(y1 - pad_h))
    nx2 = min(img_w, math.ceil(x2 + pad_w))
    ny2 = min(img_h, math.ceil(y2 + pad_h))
    return int(nx1), int(ny1), int(nx2), int(ny2)


def ensure_min_short_edge(img: Image.Image, min_short_edge: int) -> Image.Image:
    if min_short_edge is None or min_short_edge <= 0:
        return img
    w, h = img.size
    short = min(w, h)
    if short >= min_short_edge:
        return img
    scale = min_short_edge / short
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def process_image(src_path: Path, dst_dir: Path, yolo, args, rel_cat: str, manifest_writer):
    img = Image.open(src_path).convert("RGB")
    img_w, img_h = img.size
    boxes = detect_boxes(yolo, str(src_path))

    # filter cat class (COCO class 15)
    boxes = [b for b in boxes if int(b[5]) == 15 and b[4] >= args.conf_threshold]

    saved = []
    if not boxes:
        # fallback: save original image (resized to min_short_edge if needed)
        out_dir = dst_dir / rel_cat
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src_path.name
        out_img = ensure_min_short_edge(img, args.min_short_edge)
        out_img.save(out_path)
        manifest_writer.writerow({
            "orig_path": str(src_path),
            "crop_path": str(out_path),
            "cat_id": rel_cat,
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "conf": "",
        })
        return 1

    # sort boxes by confidence descending
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    to_save = boxes if args.save_all else [boxes[0]]

    out_dir = dst_dir / rel_cat
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (x1, y1, x2, y2, conf, cls) in enumerate(to_save, start=1):
        nx1, ny1, nx2, ny2 = padded_bbox(x1, y1, x2, y2, args.padding, img_w, img_h)
        crop = img.crop((nx1, ny1, nx2, ny2))
        crop = ensure_min_short_edge(crop, args.min_short_edge)
        if args.save_all:
            out_name = f"{src_path.stem}_crop{idx}{src_path.suffix}"
        else:
            out_name = src_path.name
        out_path = out_dir / out_name
        crop.save(out_path)
        manifest_writer.writerow({
            "orig_path": str(src_path),
            "crop_path": str(out_path),
            "cat_id": rel_cat,
            "x1": nx1,
            "y1": ny1,
            "x2": nx2,
            "y2": ny2,
            "conf": conf,
        })
        saved.append(out_path)

    return len(saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="meowzart_scraper/output/cats")
    parser.add_argument("--dst", default="data_crops")
    parser.add_argument("--model", default="yolov8m.pt", help="Path to yolov8 weights (optional)")
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    parser.add_argument("--min-short-edge", type=int, default=224)
    parser.add_argument("--save-all", action="store_true", help="Save all detected cats per image (else save only top)" )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo = load_yolo(args.model, device=device)
    if yolo is None:
        print("ultralytics not available — falling back to copying originals.")

    manifest_path = dst_root / "manifest.csv"
    with open(manifest_path, "w", newline='', encoding='utf-8') as mf:
        fieldnames = ["orig_path", "crop_path", "cat_id", "x1", "y1", "x2", "y2", "conf"]
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()

        total = 0
        processed = 0
        # assume src_root contains per-cat subdirectories
        for cat_dir in sorted(src_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            rel_cat = cat_dir.name
            for img_path in sorted(cat_dir.rglob('*')):
                if not img_path.is_file():
                    continue
                # simple check for image suffix
                if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                    continue
                total += 1
                try:
                    n = process_image(img_path, dst_root, yolo, args, rel_cat, writer)
                    processed += n
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

    print(f"Done. scanned={total} crops_saved={processed} manifest={manifest_path}")


if __name__ == '__main__':
    main()
