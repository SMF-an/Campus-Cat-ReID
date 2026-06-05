from pathlib import Path
from typing import Any, Optional, List, Sequence, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torch
import csv
import random
from collections import defaultdict


class CatDataset(Dataset):
    """Dataset for nested cat folders.

    Expected layout:
    root/
        Amber/
            img1.jpg
            img2.jpg
        Nana/
            xxx.png

    If no manifest is provided, the immediate parent folder name is used as cat_id.
    A CSV manifest is still supported with columns: path,cat_id.
    """

    def __init__(self, root: str, manifest: Optional[str] = None, transform=None,
                 min_samples_per_class: int = 1):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, str]] = []
        self._image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

        if manifest:
            with open(manifest, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    p = Path(r['path'])
                    if not p.is_absolute():
                        p = self.root / p
                    self.samples.append((p, r['cat_id']))
        else:
            for p in sorted(self.root.rglob('*')):
                if p.is_file() and p.suffix.lower() in self._image_extensions:
                    cat_id = self._infer_cat_id(p)
                    self.samples.append((p, cat_id))

        # Filter out cats with too few samples
        if min_samples_per_class > 1:
            cat_counts: dict[str, int] = {}
            for _, cat in self.samples:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            keep_cats = {c for c, n in cat_counts.items() if n >= min_samples_per_class}
            n_before = len(self.samples)
            self.samples = [(p, c) for p, c in self.samples if c in keep_cats]
            n_dropped = n_before - len(self.samples)
            if n_dropped > 0:
                import logging
                _log = logging.getLogger("cat-train")
                _log.info("Filtered out %d samples from %d cats (min_samples_per_class=%d); %d samples remain",
                          n_dropped, len(cat_counts) - len(keep_cats), min_samples_per_class, len(self.samples))

        # build mapping to numeric labels
        cats = sorted({cat for (_, cat) in self.samples})
        self.cat2idx = {c: i for i, c in enumerate(cats)}

    def _infer_cat_id(self, path: Path) -> str:
        """Infer cat id from directory structure.

        Prefer the immediate parent folder name. If the file lives directly under
        root, fall back to the filename stem.
        """
        try:
            rel_parent = path.parent.relative_to(self.root)
            if str(rel_parent) != '.':
                return rel_parent.parts[0]
        except ValueError:
            pass

        stem = path.stem
        if '_' in stem:
            return stem.split('_', 1)[0]
        return stem

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat = self.samples[idx]
        with Image.open(path) as image:
            img = image.convert('RGB')

            if self.transform:
                img = self.transform(img)
            else:
                # default transform: to tensor, resize to 224
                import torchvision.transforms as T

                t = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
                img = t(img)
        label = self.cat2idx[cat]
        return img, label, str(path)


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths = [b[2] for b in batch]
    return imgs, labels, paths


def stratified_split_indices(labels: Sequence[Any], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    label_to_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for indices in label_to_indices.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_indices.extend(shuffled)
            continue

        val_count = int(round(len(shuffled) * val_ratio))
        val_count = max(1, min(val_count, len(shuffled) - 1))

        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_split_samples(
    root: str,
    manifest: Optional[str],
    split: str,
    val_ratio: float,
    seed: int,
) -> List[Tuple[Path, str]]:
    dataset = CatDataset(root, manifest=manifest)
    samples = list(dataset.samples)
    labels = [cat for _, cat in samples]
    train_indices, val_indices = stratified_split_indices(labels, val_ratio=val_ratio, seed=seed)

    if split == "train":
        indices = train_indices
    elif split == "val":
        indices = val_indices
    else:
        indices = list(range(len(samples)))

    return [samples[i] for i in indices]
