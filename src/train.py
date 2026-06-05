import argparse
import logging
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset, Sampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ai.feature_extractor import build_backbone
from src.data_loader import CatDataset, collate_fn, stratified_split_indices


def setup_logger(log_file: str):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cat-train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def make_loaders(root: str, manifest: str, batch_size: int, val_ratio: float, seed: int,
                 train_transform=None, val_transform=None, pk: tuple = None,
                 min_samples_per_class: int = 1):
    # Build a dataset to enumerate labels, then create per-split datasets with transforms
    dataset_all = CatDataset(root, manifest=manifest, min_samples_per_class=min_samples_per_class)
    labels = [dataset_all.cat2idx[cat] for _, cat in dataset_all.samples]
    train_indices, val_indices = stratified_split_indices(labels, val_ratio=val_ratio, seed=seed)

    train_ds = CatDataset(root, manifest=manifest, transform=train_transform,
                          min_samples_per_class=min_samples_per_class)
    val_ds = CatDataset(root, manifest=manifest, transform=val_transform,
                        min_samples_per_class=min_samples_per_class) if val_indices else None

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices) if val_indices else None

    # if pk sampler requested, construct sampler for balanced P x K batches
    train_loader = None
    if pk is not None:
        P, K = pk
        class PKSampler(Sampler):
            def __init__(self, labels, P, K, seed=42):
                self.labels = labels
                self.P = P
                self.K = K
                self.seed = seed
                self.label2idx = defaultdict(list)
                for idx, lbl in enumerate(labels):
                    self.label2idx[int(lbl)].append(idx)
                self.rng = random.Random(seed)
                # precompute batches
                target_batches = max(1, len(labels) // (P * K))
                self.batches = []
                all_labels = list(self.label2idx.keys())
                if len(all_labels) < P:
                    raise ValueError(f"Not enough identities ({len(all_labels)}) for P={P}")
                for _ in range(target_batches):
                    chosen = self.rng.sample(all_labels, P)
                    batch = []
                    for c in chosen:
                        idxs = self.label2idx[c]
                        if len(idxs) >= K:
                            batch.extend(self.rng.sample(idxs, K))
                        else:
                            batch.extend([self.rng.choice(idxs) for _ in range(K)])
                    self.batches.append(batch)

            def __iter__(self):
                for b in self.batches:
                    for idx in b:
                        yield idx

            def __len__(self):
                return len(self.batches) * self.P * self.K

        # build labels only for the training subset so sampler indices match subset indices
        labels_train = [dataset_all.cat2idx[dataset_all.samples[i][1]] for i in train_indices]
        sampler = PKSampler(labels_train, P, K, seed)
        train_loader = DataLoader(train_subset, batch_size=P * K, sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = None
    if val_subset is not None:
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, len(train_indices), len(val_indices), len(dataset_all.cat2idx)


def build_transforms(img_size: int = 224, use_augment: bool = True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not use_augment:
        comp = A.Compose([
            A.Resize(size=(img_size, img_size)),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

        def train_transform(img):
            arr = np.array(img)
            return comp(image=arr)['image']

        def val_transform(img):
            arr = np.array(img)
            return comp(image=arr)['image']

        return train_transform, val_transform

    train_comp = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    def train_transform(img):
        arr = np.array(img)
        return train_comp(image=arr)['image']

    val_comp = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    def val_transform(img):
        arr = np.array(img)
        return val_comp(image=arr)['image']

    return train_transform, val_transform


def batch_hard_miner(embeddings: torch.Tensor, labels: torch.Tensor):
    """Return anchor/positive/negative tensors using batch-hard mining.

    embeddings: (N, D), labels: (N,)
    For each anchor i, find hardest positive (max dist within same label) and hardest negative (min dist across other labels).
    """
    if embeddings.size(0) < 2:
        return None, None, None
    # pairwise distances
    dists = torch.cdist(embeddings, embeddings, p=2)
    N = embeddings.size(0)
    anchors = []
    positives = []
    negatives = []

    for i in range(N):
        label = labels[i].item()
        mask_pos = (labels == label)
        mask_pos[i] = False
        if mask_pos.sum() == 0:
            continue
        pos_dists = dists[i].clone()
        pos_dists[~mask_pos] = -1.0  # ignore non-positives for argmax
        hard_pos_idx = int(torch.argmax(pos_dists).item())

        mask_neg = (labels != label)
        if mask_neg.sum() == 0:
            continue
        neg_dists = dists[i].clone()
        neg_dists[~mask_neg] = float('inf')
        hard_neg_idx = int(torch.argmin(neg_dists).item())

        anchors.append(embeddings[i])
        positives.append(embeddings[hard_pos_idx])
        negatives.append(embeddings[hard_neg_idx])

    if not anchors:
        return None, None, None

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def mine_triplets(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor):
    feats = model.backbone(imgs)
    emb_all = model.emb_head(feats)
    return batch_hard_miner(emb_all, labels)


def build_model(device, mode='head', num_classes=0, embed_dim=256, pretrained_backbone=True, img_size=224):
    backbone_module, backbone_dim, _ = build_backbone(
        device=device,
        pretrained=pretrained_backbone,
        image_size=img_size,
    )

    if mode == 'head':
        head = nn.Linear(backbone_dim, num_classes)
        model = nn.Module()
        model.backbone = backbone_module
        model.head = head
        return model.to(device)

    elif mode == 'triplet':
        class EmbeddingHead(nn.Module):
            def __init__(self, in_dim: int, out_dim: int = 256, normalize: bool = True):
                super().__init__()
                self.proj = nn.Linear(in_dim, out_dim)
                self.normalize = normalize

            def forward(self, x):
                x = self.proj(x)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
                return x
            
        emb_head = EmbeddingHead(backbone_dim, out_dim=embed_dim)
        model = nn.Module()
        model.backbone = backbone_module
        model.emb_head = emb_head
        return model.to(device)

    else:
        raise ValueError('Unknown mode')


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def build_triplet_optimizer(model: nn.Module, lr: float, backbone_lr_mult: float, backbone_trainable: bool):
    if backbone_trainable:
        return torch.optim.AdamW(
            [
                {'params': model.backbone.parameters(), 'lr': lr * backbone_lr_mult},
                {'params': model.emb_head.parameters(), 'lr': lr},
            ]
        )

    return torch.optim.AdamW(model.emb_head.parameters(), lr=lr)


def train_head(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = setup_logger(args.log_file)
    train_t, val_t = build_transforms(img_size=getattr(args, 'img_size', 224), use_augment=getattr(args, 'augment', True))
    train_loader, val_loader, train_size, val_size, num_classes = make_loaders(
        args.data_dir, args.manifest, args.batch_size, args.val_ratio, args.seed,
        train_transform=train_t, val_transform=val_t,
        min_samples_per_class=getattr(args, 'min_samples', 1),
    )
    model = build_model(
        device,
        mode='head',
        num_classes=num_classes,
        pretrained_backbone=not args.no_pretrained_backbone,
        img_size=args.img_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_metric = float('-inf')
    best_path = Path(args.best_output)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Head training started | device=%s | train=%d | val=%d | classes=%d",
        device,
        train_size,
        val_size,
        num_classes,
    )

    for epoch in range(args.epochs):
        model.train()
        total = 0
        correct = 0
        train_loss_total = 0.0
        for imgs, labels, paths in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            feats = model.backbone(imgs)
            logits = model.head(feats)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            train_loss_total += loss.item() * labels.size(0)

        train_acc = correct / max(total, 1)
        train_loss = train_loss_total / max(total, 1)

        val_acc = float('nan')
        val_loss = float('nan')
        if val_loader is not None:
            val_loss, val_acc = evaluate_head(model, val_loader, loss_fn, device)

        metric_value = val_acc if val_loader is not None else train_acc
        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(model.state_dict(), best_path)
            logger.info("Epoch %d: saved best checkpoint to %s", epoch + 1, best_path)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f | best=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            best_metric,
        )

    torch.save(model.state_dict(), args.output)
    logger.info("Last checkpoint saved to %s", args.output)


def train_triplet(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = setup_logger(args.log_file)
    train_t, val_t = build_transforms(img_size=getattr(args, 'img_size', 224), use_augment=getattr(args, 'augment', True))
    pk = None
    if getattr(args, 'p', None) and getattr(args, 'k', None):
        pk = (int(args.p), int(args.k))
    train_loader, val_loader, train_size, val_size, num_classes = make_loaders(
        args.data_dir, args.manifest, args.batch_size, args.val_ratio, args.seed,
        train_transform=train_t, val_transform=val_t, pk=pk,
        min_samples_per_class=getattr(args, 'min_samples', 2),
    )
    model = build_model(
        device,
        mode='triplet',
        embed_dim=args.embed_dim,
        pretrained_backbone=not args.no_pretrained_backbone,
        img_size=args.img_size,
    )
    loss_fn = torch.nn.TripletMarginLoss(margin=args.margin)
    freeze_backbone_epochs = max(0, int(getattr(args, 'freeze_backbone_epochs', 5)))
    backbone_lr_mult = float(getattr(args, 'backbone_lr_mult', 0.1))

    set_module_trainable(model.backbone, False)
    set_module_trainable(model.emb_head, True)
    optimizer = build_triplet_optimizer(model, args.lr, backbone_lr_mult, backbone_trainable=False)

    best_metric = float('inf')
    best_path = Path(args.best_output)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Triplet training started | device=%s | train=%d | val=%d | classes=%d | freeze_backbone_epochs=%d | backbone_lr_mult=%.3f",
        device,
        train_size,
        val_size,
        num_classes,
        freeze_backbone_epochs,
        backbone_lr_mult,
    )

    for epoch in range(args.epochs):
        if epoch == freeze_backbone_epochs:
            set_module_trainable(model.backbone, True)
            set_module_trainable(model.emb_head, True)
            optimizer = build_triplet_optimizer(model, args.lr, backbone_lr_mult, backbone_trainable=True)
            logger.info(
                "Epoch %d: backbone unfrozen; switching to full fine-tuning with backbone_lr=%.6f",
                epoch + 1,
                args.lr * backbone_lr_mult,
            )

        model.train()
        if epoch < freeze_backbone_epochs:
            model.backbone.eval()
        train_loss_total = 0.0
        train_triplets = 0

        for imgs, labels, paths in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            anchors, positives, negatives = mine_triplets(model, imgs, labels)
            if anchors is None:
                continue

            loss = loss_fn(anchors, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * anchors.size(0)
            train_triplets += anchors.size(0)

        train_loss = train_loss_total / max(train_triplets, 1)

        val_loss = float('nan')
        if val_loader is not None:
            val_loss = evaluate_triplet(model, val_loader, loss_fn, device)

        metric_value = val_loss if val_loader is not None else train_loss
        if metric_value < best_metric:
            best_metric = metric_value
            torch.save(model.state_dict(), best_path)
            logger.info("Epoch %d: saved best checkpoint to %s", epoch + 1, best_path)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | train_triplets=%d | val_loss=%.4f | best=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            train_triplets,
            val_loss,
            best_metric,
        )

    torch.save(model.state_dict(), args.output)
    logger.info("Last checkpoint saved to %s", args.output)


@torch.no_grad()
def evaluate_head(model, loader, loss_fn, device):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0

    for imgs, labels, paths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        feats = model.backbone(imgs)
        logits = model.head(feats)
        loss = loss_fn(logits, labels)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        loss_total += loss.item() * labels.size(0)

    return loss_total / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_triplet(model, loader, loss_fn, device):
    model.eval()
    loss_total = 0.0
    triplet_count = 0

    for imgs, labels, paths in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        anchors, positives, negatives = mine_triplets(model, imgs, labels)
        if anchors is None:
            continue

        loss = loss_fn(anchors, positives, negatives)
        loss_total += loss.item() * anchors.size(0)
        triplet_count += anchors.size(0)

    return loss_total / max(triplet_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['head', 'triplet'], default='triplet')
    parser.add_argument('--data-dir', default='data_crops')
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--no-pretrained-backbone', action='store_true', help='Initialize the backbone without pretrained weights')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', default='models/finetuned.pt')
    parser.add_argument('--best-output', default='models/finetuned_best.pt')
    parser.add_argument('--log-file', default='logs/train.log')
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=8, help='Freeze DINOv3 backbone for the first N triplet epochs')
    parser.add_argument('--backbone-lr-mult', type=float, default=0.1, help='Backbone learning-rate multiplier after unfreezing')
    parser.add_argument('--p', type=int, default=16, help='P: identities per batch (for PK sampler)')
    parser.add_argument('--k', type=int, default=4, help='K: images per identity (for PK sampler)')
    parser.add_argument('--min-samples', type=int, default=2, help='Drop cat classes with fewer than this many images (2+ required for triplet; 4+ recommended)')
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'head':
        train_head(args)
    else:
        train_triplet(args)


if __name__ == '__main__':
    main()
