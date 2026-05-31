import os
import numpy as np
from typing import List, Tuple, Dict
import faiss


class FaissIndexWrapper:
    def __init__(self, dim: int = 512, path: str = "d:/code/python/Cat/ai/index_data.npz"):
        self.dim = dim
        self.path = path
        self.ids: List[str] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.meta: Dict[str, dict] = {}
        self.faiss = faiss
        self.index = None

    def build_index(self):
        self.index = self.faiss.IndexFlatIP(self.dim)
        if self.vectors.shape[0] > 0:
            self.index.add(self.vectors)

    def add(self, id: str, vector: np.ndarray, meta: dict = None):
        vector = vector.astype(np.float32).reshape(1, -1)
        self.ids.append(id)
        self.vectors = np.vstack([self.vectors, vector])
        self.meta[id] = meta or {}
        if self.index is not None:
            self.index.add(vector)

    def search(self, query: np.ndarray, top_k: int = 3) -> Tuple[List[float], List[str]]:
        query = query.astype(np.float32).reshape(1, -1)
        if self.index is not None:
            scores, indices = self.index.search(query, top_k)
            scores = scores[0].tolist()
            ids = [self.ids[i] for i in indices[0] if i != -1]
            return scores, ids

        # Fallback brute-force
        if self.vectors.shape[0] == 0:
            return [], []
        sims = (self.vectors @ query.T).squeeze(1)
        order = np.argsort(-sims)[:top_k]
        scores = sims[order].tolist()
        ids = [self.ids[i] for i in order]
        return scores, ids

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez(self.path, ids=np.array(self.ids), vectors=self.vectors, meta=np.array([self.meta]))

    def load(self):
        if not os.path.exists(self.path):
            return
        data = np.load(self.path, allow_pickle=True)
        self.ids = data["ids"].tolist()
        self.vectors = data["vectors"]
        try:
            self.meta = data["meta"].tolist()[0]
        except Exception:
            self.meta = {}
        self.build_index()
