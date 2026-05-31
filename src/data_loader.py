from pathlib import Path
from typing import Optional, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import torch
import csv


class CatDataset(Dataset):
    """Dataset that reads images and labels. Prefer a CSV manifest with columns: path,cat_id.
    If no manifest is provided, filenames are used: prefix before first '_' or the stem as cat_id."""

    def __init__(self, root: str, manifest: Optional[str] = None, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, str]] = []

        if manifest:
            with open(manifest, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    p = Path(r['path'])
                    if not p.is_absolute():
                        p = self.root / p
                    self.samples.append((p, r['cat_id']))
        else:
            exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            for p in sorted(self.root.iterdir()):
                if p.suffix.lower() in exts:
                    stem = p.stem
                    if '_' in stem:
                        cat_id = stem.split('_', 1)[0]
                    else:
                        cat_id = stem
                    self.samples.append((p, cat_id))

        # build mapping to numeric labels
        cats = sorted({cat for (_, cat) in self.samples})
        self.cat2idx = {c: i for i, c in enumerate(cats)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cat = self.samples[idx]
        img = Image.open(path).convert('RGB')
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
