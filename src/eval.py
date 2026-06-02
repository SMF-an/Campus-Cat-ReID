from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ai.feature_extractor import VisionFeatureExtractor
from src.data_loader import build_split_samples


def compute_recall_at_k(labels: List[str], similarities: np.ndarray, ks: List[int]) -> Tuple[Dict[int, float], int]:
    """Compute Recall@k for image retrieval.

    Each image is used as a query, excluding itself from the ranked list.
    A query is counted only if it has at least one other image with the same label.
    """
    if similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarities must be a square matrix")

    valid_ks = sorted({int(k) for k in ks if int(k) > 0})
    if not valid_ks:
        raise ValueError("ks must contain at least one positive integer")

    max_k = min(max(valid_ks), similarities.shape[0] - 1)
    if max_k <= 0:
        return {k: 0.0 for k in valid_ks}, 0

    hits = {k: 0 for k in valid_ks}
    query_count = 0

    for i, query_label in enumerate(labels):
        ranked = np.argsort(-similarities[i])
        ranked = ranked[ranked != i]
        if not any(labels[j] == query_label for j in ranked):
            continue

        query_count += 1
        top_ranked = ranked[:max_k]
        for k in valid_ks:
            cutoff = min(k, len(top_ranked))
            if cutoff <= 0:
                continue
            if any(labels[j] == query_label for j in top_ranked[:cutoff]):
                hits[k] += 1

    recalls = {k: (hits[k] / query_count if query_count else 0.0) for k in valid_ks}
    return recalls, query_count


def compute_map(labels: List[str], similarities: np.ndarray) -> Tuple[float, int]:
    """Compute mean Average Precision (mAP) for image retrieval.

    Each image is used as a query, excluding itself from the ranked list.
    Only queries that have at least one other image with the same label are counted.
    """
    if similarities.shape[0] != similarities.shape[1]:
        raise ValueError("similarities must be a square matrix")

    ap_scores: List[float] = []

    for i, query_label in enumerate(labels):
        ranked = np.argsort(-similarities[i])
        ranked = ranked[ranked != i]

        relevant_total = sum(1 for j in ranked if labels[j] == query_label)
        if relevant_total == 0:
            continue

        hits = 0
        precision_sum = 0.0
        for rank, j in enumerate(ranked, start=1):
            if labels[j] != query_label:
                continue
            hits += 1
            precision_sum += hits / rank

        ap_scores.append(precision_sum / relevant_total)

    if not ap_scores:
        return 0.0, 0

    return float(np.mean(ap_scores)), len(ap_scores)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute cat image similarity by cat-name folders")
    parser.add_argument("--data-dir", default="data_crops", help="Root directory containing cropped cat images")
    parser.add_argument("--manifest", default=None, help="Optional CSV manifest with path,cat_id columns")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val", help="Which split to evaluate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio used for the stratified split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the stratified split")
    parser.add_argument("--output", default="outputs/similarity_hist.png", help="Output histogram path")
    parser.add_argument("--metrics-output", default="outputs/evaluation_metrics.json", help="Output JSON path for evaluation metrics")
    parser.add_argument("--checkpoint", default="models/finetuned_best.pt", help="Path to finetuned checkpoint to load for feature extraction")
    parser.add_argument("--recall-ks", nargs="*", type=int, default=[1, 5, 10], help="Recall@k values to report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    samples = build_split_samples(str(data_dir), args.manifest, args.split, args.val_ratio, args.seed)

    if not samples:
        raise SystemExit(f"No images found in {data_dir.resolve()} for split={args.split}")

    labels = [label for _, label in samples]
    image_paths = [path for path, _ in samples]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = VisionFeatureExtractor(device=device, checkpoint=args.checkpoint)

    print(f"Using device: {device}")
    print(f"Found {len(samples)} images across {len(set(labels))} cats")
    for cat_name in sorted(set(labels)):
        count = sum(1 for label in labels if label == cat_name)
        print(f"  {cat_name}: {count}")

    feats = []
    for path in image_paths:
        with Image.open(path) as image:
            img = image.convert("RGB")
        feats.append(extractor.extract(img))
    feats = np.vstack(feats)

    # Cosine similarities since features are L2-normalized.
    sims = feats @ feats.T

    n = sims.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sims[i, j])))

    intra_scores = []
    inter_scores = []
    for i, j, score in pairs:
        if labels[i] == labels[j]:
            intra_scores.append(score)
        else:
            inter_scores.append(score)

    os.makedirs(Path(args.output).parent, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if intra_scores:
        plt.hist(intra_scores, bins=30, alpha=0.55, label="intra-cat")
    if inter_scores:
        plt.hist(inter_scores, bins=30, alpha=0.55, label="inter-cat")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()

    print("Similarity summary:")
    print(f"  pairs total: {len(pairs)}")
    if intra_scores:
        intra = np.array(intra_scores)
        print(f"  intra mean={intra.mean():.4f} std={intra.std():.4f} count={len(intra)}")
    if inter_scores:
        inter = np.array(inter_scores)
        print(f"  inter mean={inter.mean():.4f} std={inter.std():.4f} count={len(inter)}")

    if intra_scores and inter_scores:
        threshold = float((np.mean(intra_scores) + np.mean(inter_scores)) / 2.0)
        print(f"Recommended threshold (midpoint) = {threshold:.4f}")
    else:
        print("Need at least two cat folders with images to estimate a threshold")

    recalls, query_count = compute_recall_at_k(labels, sims, args.recall_ks)
    mean_ap, map_query_count = compute_map(labels, sims)
    print("Retrieval summary:")
    print(f"  queries with positives: {query_count}")
    for k in sorted(recalls):
        print(f"  Recall@{k} = {recalls[k]:.4f}")
    print(f"  mAP = {mean_ap:.4f} (queries used: {map_query_count})")

    metrics = {
        "data_dir": str(data_dir.resolve()),
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "split": args.split,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_images": len(samples),
        "num_categories": len(set(labels)),
        "pairs_total": len(pairs),
        "intra_cat": {
            "mean": float(np.mean(intra_scores)) if intra_scores else None,
            "std": float(np.std(intra_scores)) if intra_scores else None,
            "count": len(intra_scores),
        },
        "inter_cat": {
            "mean": float(np.mean(inter_scores)) if inter_scores else None,
            "std": float(np.std(inter_scores)) if inter_scores else None,
            "count": len(inter_scores),
        },
        "recommended_threshold": float((np.mean(intra_scores) + np.mean(inter_scores)) / 2.0) if intra_scores and inter_scores else None,
        "retrieval": {
            "queries_with_positives": query_count,
            "recall_at_k": {f"Recall@{k}": float(recalls[k]) for k in sorted(recalls)},
            "mAP": float(mean_ap),
            "map_queries_used": map_query_count,
        },
    }

    metrics_path = Path(args.metrics_output)
    os.makedirs(metrics_path.parent, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics JSON saved to: {metrics_path.resolve()}")

    print(f"Histogram saved to: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
