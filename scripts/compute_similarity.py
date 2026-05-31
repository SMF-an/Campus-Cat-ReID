from pathlib import Path
import argparse
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt
import torch

from ai.detector import crop_cat_from_bytes
from ai.feature_extractor import CLIPFeatureExtractor


def find_images(data_dir: Path) -> List[Path]:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def parse_groups(groups_str: str):
    # groups_str like "1,2,4;3,5"
    groups = []
    if not groups_str:
        return groups
    for part in groups_str.split(";"):
        ids = [s.strip() for s in part.split(",") if s.strip()]
        groups.append(ids)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--groups", default="1,2,4;3;5", help="Semicolon-separated groups of basenames, e.g. '1,2,4;3,5'")
    parser.add_argument("--out", default="outputs/similarity_hist.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_paths = find_images(data_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = CLIPFeatureExtractor(device=device)

    names = [p.name for p in image_paths]
    feats = []
    for p in image_paths:
        img = crop_cat_from_bytes(p.read_bytes())
        feats.append(extractor.extract(img))
    feats = np.vstack(feats)

    # Cosine similarities since features are L2-normalized
    sims = feats @ feats.T

    # Collect upper-triangle (excluding diagonal)
    n = sims.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sims[i, j])))

    groups = parse_groups(args.groups)
    intra_scores = []
    inter_scores = []
    if groups:
        # map basename (without ext or with) to index
        def match_name(token, name):
            # token may be '1' and name '1.webp' -> startswith
            return name.startswith(token)

        for i, j, s in pairs:
            name_i = names[i]
            name_j = names[j]
            same_group = False
            for g in groups:
                if any(match_name(t, name_i) for t in g) and any(match_name(t, name_j) for t in g):
                    same_group = True
                    break
            if same_group:
                intra_scores.append(s)
            else:
                inter_scores.append(s)

    # Plotting
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(8, 4))
    if intra_scores:
        plt.hist(intra_scores, bins=30, alpha=0.5, label="intra-group")
    if inter_scores:
        plt.hist(inter_scores, bins=30, alpha=0.5, label="inter-group")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()

    # Summary and threshold recommendation
    print("Similarity summary:")
    if intra_scores:

        intra = np.array(intra_scores)
        inter = np.array(inter_scores)
        print(f"  intra mean={intra.mean():.4f} std={intra.std():.4f} count={len(intra)}")
        print(f"  inter mean={inter.mean():.4f} std={inter.std():.4f} count={len(inter)}")
        # recommend threshold midway between intra mean and inter mean
        thresh = float((intra.mean() + inter.mean()) / 2.0)
        print(f"Recommended threshold (midpoint) = {thresh:.4f}")
    else:
        print("No groups provided — returning default thresholds: 0.80 (confirmed), 0.50 (uncertain)")


if __name__ == "__main__":
    main()
