import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
        self.loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)
