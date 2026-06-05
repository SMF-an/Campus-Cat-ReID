"""Test runner: leave-one-out validation with confusion matrix and PR curve."""

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from ai.feature_extractor import VisionFeatureExtractor
from src.data_loader import build_split_samples


def load_samples(data_dir: Path, manifest: str | None, split: str, val_ratio: float, seed: int) -> list[tuple[Path, str]]:
    return build_split_samples(str(data_dir), manifest, split, val_ratio, seed)


def load_features(image_paths: list[Path], extractor: VisionFeatureExtractor) -> list[tuple[Path, object]]:
    items = []
    for image_path in image_paths:
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        feature = extractor.extract(image)
        items.append((image_path, feature))
    return items


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_precision_recall_curve(precisions: np.ndarray, recalls: np.ndarray, out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a leave-one-out gallery test for cat recognition")
    parser.add_argument("--data-dir", default="data_crops", help="Directory containing cropped sample images")
    parser.add_argument("--manifest", default=None, help="Optional CSV manifest with path,cat_id columns")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val", help="Which split to evaluate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio used for the stratified split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the stratified split")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to report")
    parser.add_argument("--threshold", type=float, default=0.15, help="Cosine similarity threshold to accept a match")
    parser.add_argument("--checkpoint", default="models/finetuned_best.pt", help="Path to finetuned checkpoint to load for feature extraction")
    parser.add_argument("--cm-out", default="outputs/confusion_matrix.png", help="Confusion matrix image path")
    parser.add_argument("--pr-out", default="outputs/precision_recall.png", help="Precision-recall curve image path")
    parser.add_argument("--metrics-out", default="outputs/test_metrics.json", help="JSON file to write test metrics")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    samples = load_samples(data_dir, args.manifest, args.split, args.val_ratio, args.seed)
    if len(samples) < 2:
        raise SystemExit(f"Need at least 2 sample images in {data_dir.resolve()} for split={args.split}, found {len(samples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = VisionFeatureExtractor(device=device, checkpoint=args.checkpoint)

    print(f"Using device: {device}")
    print(f"Found {len(samples)} sample images in {data_dir.resolve()} (split={args.split})")
    print("Precomputing features...")
    paths = [p for p, _ in samples]
    labels = [lbl for _, lbl in samples]
    feats_items = load_features(paths, extractor)
    # feats_items: list of (path, feature)
    features = np.vstack([f for _, f in feats_items])

    # Build pairwise scores for PR curve
    y_true = []
    y_scores = []

    # For confusion matrix, we will predict top-1 with threshold
    unique_labels = sorted(set(labels))
    unknown_label = "__unknown__"
    pred_labels = unique_labels + [unknown_label]
    label_to_idx = {lbl: i for i, lbl in enumerate(pred_labels)}
    cm = np.zeros((len(pred_labels), len(pred_labels)), dtype=int)

    n = features.shape[0]
    for i in range(n):
        q_feat = features[i]
        sims = features @ q_feat
        # build pairwise lists for PR
        for j in range(n):
            if i == j:
                continue
            y_scores.append(float(sims[j]))
            y_true.append(1 if labels[i] == labels[j] else 0)

        # leave-one-out gallery for prediction
        # find top-1 excluding self
        ranked = np.argsort(-sims)
        ranked = ranked[ranked != i]
        best_idx = ranked[0]
        best_sim = float(sims[best_idx])
        best_label = labels[best_idx]
        if best_sim >= args.threshold:
            predicted = best_label
        else:
            predicted = unknown_label

        true_label = labels[i]
        cm[label_to_idx[true_label], label_to_idx[predicted]] += 1

    # Compute precision-recall curve
    if y_scores:
        order = np.argsort(-np.array(y_scores))
        y_sorted = np.array(y_true)[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        precisions = tp / (tp + fp)
        recalls = tp / max(1, np.sum(y_sorted == 1))
    else:
        precisions = np.array([])
        recalls = np.array([])

    # save outputs
    out_cm = Path(args.cm_out)
    out_pr = Path(args.pr_out)
    out_metrics = Path(args.metrics_out)
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    out_pr.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(cm, pred_labels, out_cm)
    if precisions.size and recalls.size:
        plot_precision_recall_curve(precisions, recalls, out_pr)

    # basic accuracy
    correct = sum(cm[i, i] for i in range(len(unique_labels)))
    total = len(samples)
    accuracy = correct / max(1, total)

    metrics = {
        "num_images": len(samples),
        "num_categories": len(unique_labels),
        "split": args.split,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "threshold": args.threshold,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "labels": pred_labels,
    }
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Confusion matrix saved to: {out_cm.resolve()}")
    print(f"Precision-recall curve saved to: {out_pr.resolve()}")
    print(f"Metrics JSON saved to: {out_metrics.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
