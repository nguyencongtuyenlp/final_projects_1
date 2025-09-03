from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json, os, uuid
import numpy as np
import faiss

from app.models.registry import get_sbert

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

class RAGStore:
    def __init__(self, base_dir: str | Path = "storage/rag"):
        self.base_dir = Path(base_dir)
        _ensure_dir(self.base_dir)
        self.index_path = self.base_dir / "index.faiss"
        self.meta_path = self.base_dir / "meta.jsonl"
        self.model = get_sbert()
        self.index = None
        self.dim = None
        self._load()

    def _load(self):
        if self.meta_path.exists() and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dim = self.index.d
        else:
            self.index = None
            self.dim = None

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
            self.dim = dim

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def add_documents(self, docs: List[Dict[str, Any]]) -> int:
        """docs: List[{id?, text, meta?}]"""
        metas = []
        texts = []
        for d in docs:
            did = d.get("id") or str(uuid.uuid4())
            meta = d.get("meta", {})
            txt = d.get("text", "").strip()
            if not txt:
                continue
            metas.append({"id": did, "meta": meta, "text": txt})
            texts.append(txt)
        if not texts:
            return 0
        embs = self._embed(texts)
        self._ensure_index(embs.shape[1])
        self.index.add(embs)
        # append metas
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        faiss.write_index(self.index, str(self.index_path))
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        q = self._embed([query])
        D, I = self.index.search(q, top_k)
        I = I[0].tolist()
        D = D[0].tolist()
        # read metas sequentially (small scale demo; for large scale keep sidecar sqlite/parquet)
        results = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for rank, (idx, score) in enumerate(zip(I, D)):
            if idx < 0 or idx >= len(lines):
                continue
            m = json.loads(lines[idx])
            m.update({"score": float(score), "rank": rank})
            results.append(m)
        return results
