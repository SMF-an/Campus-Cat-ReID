#!/usr/bin/env python3
"""Split baseline evaluation for Campus-Cat-ReID.

Scans a data directory (default data/cats) where each subfolder is one cat,
splits gallery/test per cat (8:2 by default), builds a per-image FAISS index
from gallery, queries with test images, and reports Top1 / Top3 accuracy.

Also supports reading from data/manifest.csv via --manifest mode.

Usage:
    PYTHONPATH=. uv run python scripts/evaluate_split_baseline.py
    PYTHONPATH=. uv run python scripts/evaluate_split_baseline.py --no-detect
    PYTHONPATH=. uv run python scripts/evaluate_split_baseline.py --data-dir data/reid
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ai.faiss_index import FaissIndexWrapper
from ai.feature_extractor import VisionFeatureExtractor
from src.data_loader import stratified_split_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# YOLO helper (load once, reuse)
# ---------------------------------------------------------------------------

def make_crop_fn(no_detect: bool):
    """Return a callable (PIL.Image -> PIL.Image) that optionally crops cats."""
    if no_detect:
        return lambda img: img

    from ultralytics import YOLO

    logger.info("Loading YOLO model …")
    yolo = YOLO("yolov8m.pt")

    def crop_cat(img: Image.Image) -> Image.Image:
        try:
            results = yolo(img)
            for box in results[0].boxes:
                if int(box.cls) == 15:  # COCO class 15 == cat
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    return img.crop((x1, y1, x2, y2))
        except Exception:
            pass
        return img

    return crop_cat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def scan_data_dir(data_dir: Path) -> list[dict]:
    """Scan a directory where each subfolder is one cat.

    Returns list of dicts with keys: cat_id, cat_name, image_path.
    cat_id and cat_name are both the folder name.
    """
    rows = []
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if cat_dir.name.startswith("."):
            continue
        cat_id = cat_dir.name
        cat_name = cat_dir.name
        for img_path in sorted(cat_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if img_path.name.startswith("."):
                continue
            rows.append({
                "cat_id": cat_id,
                "cat_name": cat_name,
                "image_path": str(img_path),
            })
    return rows


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load from manifest.csv, resolving image_path to actual files.

    If image_path doesn't exist, tries raw_path as fallback.
    Skips rows where no image can be found.
    """
    from ai.dataset_utils import read_manifest

    raw_rows = read_manifest(str(manifest_path))
    rows = []
    for r in raw_rows:
        img = Path(r["image_path"])
        if img.exists():
            rows.append({**r, "image_path": str(img)})
            continue
        # Fallback to raw_path
        raw = Path(r.get("raw_path", ""))
        if raw.exists():
            rows.append({**r, "image_path": str(raw)})
            continue
        logger.warning("Skipping row — image not found: %s", r["image_path"])
    return rows


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def stratified_split(
    rows: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[str]]:
    """Split rows into gallery / test using the same logic as training.

    Delegates to stratified_split_indices (src/data_loader.py) so the split
    is bit-identical to what train.py produced for the same seed and val_ratio.

    Returns (gallery_rows, test_rows, skipped_test_cats).
    """
    labels = [r["cat_id"] for r in rows]
    train_indices, val_indices = stratified_split_indices(labels, val_ratio=val_ratio, seed=seed)

    gallery = [rows[i] for i in train_indices]
    test = [rows[i] for i in val_indices]

    # Cats that ended up with zero test images
    test_cats = {r["cat_id"] for r in test}
    all_cats = {r["cat_id"] for r in rows}
    skipped = sorted(all_cats - test_cats)

    return gallery, test, skipped


# ---------------------------------------------------------------------------
# Image-level → Cat-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_cat_level(
    scores: list[float],
    ids: list[str],
    meta: dict[str, dict],
    top_k: int,
) -> list[dict]:
    """Deduplicate image-level FAISS results to cat-level.

    Returns list of dicts with keys: cat_id, cat_name, score, matched_image_path.
    Sorted by score descending, at most top_k unique cats.
    """
    seen_cats: set[str] = set()
    result: list[dict] = []

    for score, fid in zip(scores, ids):
        m = meta.get(fid, {})
        cat_id = m.get("cat_id", "")
        if not cat_id or cat_id in seen_cats:
            continue
        seen_cats.add(cat_id)
        result.append({
            "cat_id": cat_id,
            "cat_name": m.get("cat_name", ""),
            "score": round(float(score), 4),
            "matched_image_path": m.get("image_path", ""),
        })
        if len(result) >= top_k:
            break

    return result


# ---------------------------------------------------------------------------
# Process one image through the pipeline
# ---------------------------------------------------------------------------

def process_image(
    image_path: str,
    crop_fn,
    extractor: VisionFeatureExtractor,
) -> Optional[np.ndarray]:
    """Read image, optionally crop, extract feature vector."""
    p = Path(image_path)
    if not p.exists():
        logger.warning("Image not found: %s", image_path)
        return None
    try:
        img = Image.open(p).convert("RGB")
        img = crop_fn(img)
        vec = extractor.extract(img)
        return vec
    except Exception as exc:
        logger.warning("Failed to process %s: %s", image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Split baseline evaluation")
    parser.add_argument("--data-dir", default="data_crops", type=str,
                        help="Directory with per-cat subfolders (default: data_crops)")
    parser.add_argument("--manifest", default=None, type=str,
                        help="Use manifest.csv instead of --data-dir")
    parser.add_argument("--checkpoint", default="models/finetuned_best.pt", type=str,
                        help="Path to fine-tuned model checkpoint (e.g. models/finetuned_best.pt)")
    parser.add_argument("--gallery-ratio", default=None, type=float,
                        help="Deprecated; use --val-ratio instead (gallery_ratio = 1 - val_ratio)")
    parser.add_argument("--val-ratio", default=0.2, type=float,
                        help="Validation ratio for gallery/test split — must match training (default: 0.2)")
    parser.add_argument("--min-samples", default=2, type=int,
                        help="Drop cats with fewer than this many images — must match training --min-samples")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--out-dir", default="outputs/split_baseline", type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--no-detect", action="store_true",
                        help="Skip YOLO crop, use original images")
    args = parser.parse_args()

    # Backward compatibility: --gallery-ratio falls back to --val-ratio
    val_ratio = args.val_ratio
    if args.gallery_ratio is not None:
        val_ratio = 1.0 - args.gallery_ratio
        logger.warning("--gallery-ratio is deprecated; using val_ratio=%.2f (1 - %.2f)", val_ratio, args.gallery_ratio)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Split Baseline Evaluation ===")
    logger.info("val_ratio=%.2f  seed=%d  top_k=%d  device=%s  no_detect=%s  min_samples=%d",
                val_ratio, args.seed, args.top_k, device, args.no_detect, args.min_samples)

    # ------------------------------------------------------------------
    # Step 1: Load & Split
    # ------------------------------------------------------------------
    if args.manifest:
        logger.info("Loading from manifest: %s", args.manifest)
        rows = load_manifest(Path(args.manifest))
    else:
        data_dir = Path(args.data_dir)
        logger.info("Scanning data directory: %s", data_dir)
        rows = scan_data_dir(data_dir)

    logger.info("Loaded %d images across %d cats",
                len(rows), len(set(r["cat_id"] for r in rows)))

    # Apply min_samples filter to match training
    if args.min_samples > 1:
        cat_counts = Counter(r["cat_id"] for r in rows)
        keep_cats = {c for c, n in cat_counts.items() if n >= args.min_samples}
        n_before = len(rows)
        n_cats_before = len(cat_counts)
        rows = [r for r in rows if r["cat_id"] in keep_cats]
        logger.info("Filtered (min_samples=%d): %d → %d images, %d → %d cats",
                    args.min_samples, n_before, len(rows), n_cats_before, len(keep_cats))

    gallery_rows, test_rows, skipped_cats = stratified_split(
        rows, val_ratio, args.seed,
    )
    logger.info("Split: gallery=%d  test=%d  skipped_test_cats=%d",
                len(gallery_rows), len(test_rows), len(skipped_cats))

    # Write split_summary.json
    by_cat_gallery = Counter(r["cat_id"] for r in gallery_rows)
    by_cat_test = Counter(r["cat_id"] for r in test_rows)
    all_cats = sorted(set(by_cat_gallery) | set(by_cat_test))
    split_summary = {
        "val_ratio": val_ratio,
        "seed": args.seed,
        "total_rows": len(rows),
        "gallery_count": len(gallery_rows),
        "test_count": len(test_rows),
        "skipped_test_cats": skipped_cats,
        "per_cat": {
            cat: {
                "gallery": by_cat_gallery.get(cat, 0),
                "test": by_cat_test.get(cat, 0),
            }
            for cat in all_cats
        },
    }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out_dir / "split_summary.json")

    # ------------------------------------------------------------------
    # Step 2: Build Gallery Index
    # ------------------------------------------------------------------
    crop_fn = make_crop_fn(args.no_detect)
    extractor = VisionFeatureExtractor(device=device, checkpoint=args.checkpoint)
    logger.info("Feature extractor ready (dim=%d)%s", extractor.dim,
                f" (checkpoint={args.checkpoint})" if args.checkpoint else "")

    index = FaissIndexWrapper(dim=extractor.dim, path=str(out_dir / "index_data.npz"))

    logger.info("Building gallery index (%d images) …", len(gallery_rows))
    for i, row in enumerate(gallery_rows):
        vec = process_image(row["image_path"], crop_fn, extractor)
        if vec is None:
            continue
        stem = Path(row["image_path"]).stem
        fid = f"{row['cat_id']}::{stem}"
        index.add(
            id=fid,
            vector=vec,
            meta={
                "cat_id": row["cat_id"],
                "cat_name": row["cat_name"],
                "image_path": row["image_path"],
            },
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(gallery_rows):
            logger.info("  gallery indexed: %d / %d", i + 1, len(gallery_rows))

    index.build_index()
    index.save()
    logger.info("Gallery index saved: %d vectors → %s", len(index.ids), out_dir / "index_data.npz")

    if len(index.ids) == 0:
        logger.error("Gallery index is empty — nothing to search. Aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Query Test Set
    # ------------------------------------------------------------------
    predictions: list[dict] = []
    logger.info("Querying test set (%d images) …", len(test_rows))

    for i, row in enumerate(test_rows):
        vec = process_image(row["image_path"], crop_fn, extractor)
        if vec is None:
            continue

        # Search with a larger top_k to get enough unique cats after aggregation
        search_k = min(args.top_k * 5, len(index.ids))
        scores, ids = index.search(vec, top_k=search_k)

        # Aggregate to cat-level
        cat_results = aggregate_to_cat_level(scores, ids, index.meta, args.top_k)

        true_cat_id = row["cat_id"]
        true_cat_name = row["cat_name"]

        pred_cat_id = cat_results[0]["cat_id"] if cat_results else ""
        pred_cat_name = cat_results[0]["cat_name"] if cat_results else ""
        top1_score = cat_results[0]["score"] if cat_results else 0.0

        top1_correct = int(pred_cat_id == true_cat_id)
        top3_correct = int(any(c["cat_id"] == true_cat_id for c in cat_results))

        predictions.append({
            "query_path": row["image_path"],
            "true_cat_id": true_cat_id,
            "true_cat_name": true_cat_name,
            "pred_cat_id": pred_cat_id,
            "pred_cat_name": pred_cat_name,
            "top1_score": top1_score,
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
            "candidates": cat_results,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(test_rows):
            logger.info("  test queried: %d / %d", i + 1, len(test_rows))

    # ------------------------------------------------------------------
    # Step 4: Compute Metrics
    # ------------------------------------------------------------------
    total = len(predictions)
    top1_correct_count = sum(p["top1_correct"] for p in predictions)
    top3_correct_count = sum(p["top3_correct"] for p in predictions)
    top1_acc = top1_correct_count / total if total else 0.0
    top3_acc = top3_correct_count / total if total else 0.0

    # Per-cat accuracy
    per_cat: dict[str, dict] = {}
    by_cat_preds: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        by_cat_preds[p["true_cat_id"]].append(p)

    for cat_id in sorted(by_cat_preds):
        cat_preds = by_cat_preds[cat_id]
        n = len(cat_preds)
        t1 = sum(p["top1_correct"] for p in cat_preds)
        t3 = sum(p["top3_correct"] for p in cat_preds)
        cat_name = cat_preds[0]["true_cat_name"]
        per_cat[cat_id] = {
            "cat_name": cat_name,
            "n_test": n,
            "top1_correct": t1,
            "top1_accuracy": round(t1 / n, 4) if n else 0.0,
            "top3_correct": t3,
            "top3_accuracy": round(t3 / n, 4) if n else 0.0,
        }

    # Most confused pairs
    confused: Counter = Counter()
    for p in predictions:
        if not p["top1_correct"]:
            confused[(p["true_cat_id"], p["pred_cat_id"])] += 1

    most_confused = [
        {"true_cat_id": t, "pred_cat_id": pr, "count": c}
        for (t, pr), c in confused.most_common(20)
    ]

    # ------------------------------------------------------------------
    # Step 5: Write Outputs
    # ------------------------------------------------------------------
    report = {
        "total_test_queries": total,
        "top1_correct": top1_correct_count,
        "top1_accuracy": round(top1_acc, 4),
        "top3_correct": top3_correct_count,
        "top3_accuracy": round(top3_acc, 4),
        "per_cat_accuracy": per_cat,
        "skipped_test_cats": skipped_cats,
        "most_confused_pairs": most_confused,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", out_dir / "report.json")

    # predictions.csv
    csv_fields = [
        "query_path", "true_cat_id", "true_cat_name",
        "pred_cat_id", "pred_cat_name",
        "top1_score", "top1_correct", "top3_correct",
        "candidates",
    ]
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for p in predictions:
            row_out = {k: p[k] for k in csv_fields if k != "candidates"}
            row_out["candidates"] = json.dumps(p["candidates"], ensure_ascii=False)
            writer.writerow(row_out)
    logger.info("Wrote %s", out_dir / "predictions.csv")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info("Gallery: %d images | Test: %d images", len(gallery_rows), total)
    logger.info("Skipped test cats (too few images): %s",
                skipped_cats if skipped_cats else "none")
    logger.info("Top1 accuracy: %.4f (%d / %d)", top1_acc, top1_correct_count, total)
    logger.info("Top3 accuracy: %.4f (%d / %d)", top3_acc, top3_correct_count, total)
    if most_confused:
        logger.info("Most confused pairs:")
        for mc in most_confused[:5]:
            logger.info("  %s → %s  (%d times)",
                        mc["true_cat_id"], mc["pred_cat_id"], mc["count"])
    logger.info("Outputs written to: %s", out_dir)


if __name__ == "__main__":
    main()
