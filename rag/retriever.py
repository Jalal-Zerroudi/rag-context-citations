from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .indexer import RAGIndex, Chunk

@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    ref_id: int  # 1..K (pour citations)

class Retriever:
    def __init__(self, rag_index: RAGIndex, embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.rag_index = rag_index
        self.embedder = SentenceTransformer(embedder_name)

    def search(self, query: str, topk: int = 6) -> List[Retrieved]:
        if self.rag_index.index.ntotal == 0:
            return []
        q = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.rag_index.index.search(q, topk)
        out: List[Retrieved] = []
        for rank, (idx, sc) in enumerate(zip(ids[0].tolist(), scores[0].tolist()), start=1):
            if idx < 0 or idx >= len(self.rag_index.chunks):
                continue
            out.append(Retrieved(chunk=self.rag_index.chunks[idx], score=float(sc), ref_id=rank))
        return out
