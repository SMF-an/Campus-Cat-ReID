import argparse
from pathlib import Path

import torch

from ai.detector import crop_cat_from_bytes
from ai.feature_extractor import CLIPFeatureExtractor
from ai.faiss_index import FaissIndexWrapper


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(data_dir: Path) -> list[Path]:
    return sorted([path for path in data_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def load_features(image_paths: list[Path], extractor: CLIPFeatureExtractor) -> list[tuple[Path, object]]:
    items = []
    for image_path in image_paths:
        image_bytes = image_path.read_bytes()
        image = crop_cat_from_bytes(image_bytes)
        feature = extractor.extract(image)
        items.append((image_path, feature))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a leave-one-out gallery test for cat recognition")
    parser.add_argument("--data-dir", default="data", help="Directory containing sample images")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_paths = find_images(data_dir)
    if len(image_paths) < 2:
        raise SystemExit(f"Need at least 2 sample images in {data_dir.resolve()}, found {len(image_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = CLIPFeatureExtractor(device=device)

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} sample images in {data_dir.resolve()}")
    print("Precomputing features...")
    samples = load_features(image_paths, extractor)

    print("\nLeave-one-out results:")
    for query_path, query_feature in samples:
        index = FaissIndexWrapper(dim=extractor.dim)

        for gallery_index, (gallery_path, gallery_feature) in enumerate(samples, start=1):
            if gallery_path == query_path:
                continue
            cat_id = f"sample_{gallery_index:03d}"
            index.add(cat_id, gallery_feature, meta={"path": str(gallery_path)})

        scores, ids = index.search(query_feature, top_k=args.top_k)

        print(f"query={query_path.resolve()}")
        for rank, (cat_id, score) in enumerate(zip(ids, scores), start=1):
            candidate_path = index.meta.get(cat_id, {}).get("path", "<unknown>")
            print(f"  top{rank}: {cat_id} score={score:.4f} path={candidate_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
