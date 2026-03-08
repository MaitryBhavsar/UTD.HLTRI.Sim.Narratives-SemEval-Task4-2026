"""MiniLM embedder with cache and safe text preprocessor."""

import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def preprocess_text(text: str) -> str:
    """Strip leading/trailing whitespace; collapse multiple spaces."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


class MiniLMEmbedder:
    """Sentence embedder using all-MiniLM-L6-v2 with optional cache."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self._cache: dict[str, np.ndarray] = {}

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings. Uses cache for duplicates.
        If normalize=True, L2-normalize so cosine sim = dot product.
        """
        preprocessed = [preprocess_text(t) for t in texts]
        out = np.zeros((len(texts), self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        to_compute: List[tuple[int, str]] = []
        for i, t in enumerate(preprocessed):
            if normalize and t in self._cache:
                out[i] = self._cache[t]
            else:
                to_compute.append((i, t))

        if to_compute:
            indices = [x[0] for x in to_compute]
            strings = [x[1] for x in to_compute]
            emb = self.model.encode(
                strings,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            if not normalize and emb.dtype != np.float32:
                emb = emb.astype(np.float32)
            for j, (idx, s) in enumerate(to_compute):
                out[idx] = emb[j]
                if normalize:
                    self._cache[s] = emb[j].copy()

        return out
