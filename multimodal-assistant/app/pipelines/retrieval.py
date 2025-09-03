from typing import List, Dict
import numpy as np
from app.models.registry import get_sbert

class SemanticRetriever:
    def __init__(self):
        self.model = get_sbert()

    def _embed(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def search(self, query: str, corpus: List[str], top_k: int = 3) -> List[Dict]:
        corpus = corpus or []
        if not corpus:
            return []
        q = self._embed([query])[0]
        C = self._embed(corpus)
        sims = (C @ q)
        idx = np.argsort(-sims)[:top_k]
        return [{"text": corpus[i], "score": float(sims[i]), "idx": int(i)} for i in idx]
