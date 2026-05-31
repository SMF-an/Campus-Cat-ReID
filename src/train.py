import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel

from src.data_loader import CatDataset, collate_fn
from src.metric_learning import EmbeddingHead


def build_model(device, mode='head', num_classes=0, embed_dim=256):
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').vision_model
    clip.eval()
    # Use the pooled output if available; else use last_hidden_state mean
    class FeatureExtractor(nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, x):
            out = self.clip(pixel_values=x)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                feat = out.pooler_output
            else:
                feat = out.last_hidden_state.mean(dim=1)
            return feat

    backbone = FeatureExtractor(clip)

    if mode == 'head':
        head = nn.Linear(backbone.clip.config.hidden_size, num_classes)
        model = nn.Module()
        model.backbone = backbone
        model.head = head
        return model.to(device)

    elif mode == 'triplet':
        emb_head = EmbeddingHead(backbone.clip.config.hidden_size, out_dim=embed_dim)
        model = nn.Module()
        model.backbone = backbone
        model.emb_head = emb_head
        return model.to(device)

    else:
        raise ValueError('Unknown mode')


def train_head(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = CatDataset(args.data_dir, manifest=args.manifest)
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = build_model(device, mode='head', num_classes=len(ds.cat2idx))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total = 0
        correct = 0
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
        print(f'Epoch {epoch+1}/{args.epochs} acc={correct/total:.4f}')
    torch.save(model.state_dict(), args.output)


def train_triplet(args):
    # Placeholder: user can replace with a proper P×K sampler implementation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = CatDataset(args.data_dir, manifest=args.manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = build_model(device, mode='triplet', embed_dim=args.embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.TripletMarginLoss(margin=args.margin)

    model.train()
    for epoch in range(args.epochs):
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            # naive triplet: take first as anchor, second as positive, third as negative when possible
            if imgs.size(0) < 3:
                continue
            a = imgs[0::3]
            p = imgs[1::3]
            n = imgs[2::3]
            a_f = model.backbone(a)
            p_f = model.backbone(p)
            n_f = model.backbone(n)
            a_e = model.emb_head(a_f)
            p_e = model.emb_head(p_f)
            n_e = model.emb_head(n_f)
            loss = loss_fn(a_e, p_e, n_e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{args.epochs} done')
    torch.save(model.state_dict(), args.output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['head', 'triplet'], default='head')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', default='models/finetuned.pt')
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--margin', type=float, default=0.2)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'head':
        train_head(args)
    else:
        train_triplet(args)


if __name__ == '__main__':
    main()
