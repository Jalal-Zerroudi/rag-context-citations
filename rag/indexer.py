from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer

from .loaders import Document, iter_files, load_any, sha256_file

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]  # path, page, chunk_id

@dataclass
class RAGIndex:
    index: faiss.Index
    chunks: List[Chunk]         # position i -> chunk
    dim: int

def _ensure_dirs(cache_dir: Path):
    (cache_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (cache_dir / "embeddings").mkdir(parents=True, exist_ok=True)

def build_or_load_index(
    data_dir: Path | List[Path],
    cache_dir: Path,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 900,
    overlap: int = 150,
) -> RAGIndex:
    """
    Cache par fichier:
    - cache/chunks/<hash>.jsonl
    - cache/embeddings/<hash>.npy
    + cache/file_hashes.json (mapping path -> hash)
    On rebuild l'index FAISS à partir des caches.
    """
    _ensure_dirs(cache_dir)

    hashes_path = cache_dir / "file_hashes.json"
    old_hashes: Dict[str, str] = {}
    if hashes_path.exists():
        old_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))

    new_hashes: Dict[str, str] = {}
    files = sorted(list(iter_files(data_dir)))

    # Embedder
    embedder = SentenceTransformer(embedding_model_name)

    all_chunks: List[Chunk] = []
    all_embs: List[np.ndarray] = []

    for fp in tqdm(files, desc="Scan & Cache"):
        file_hash = sha256_file(fp)
        new_hashes[str(fp)] = file_hash

        chunks_file = cache_dir / "chunks" / f"{file_hash}.jsonl"
        emb_file = cache_dir / "embeddings" / f"{file_hash}.npy"

        need_recompute = (old_hashes.get(str(fp)) != file_hash) or (not chunks_file.exists()) or (not emb_file.exists())

        if need_recompute:
            docs: List[Document] = load_any(fp)
            file_chunks: List[Chunk] = []
            for d in docs:
                pieces = chunk_text(d.text, chunk_size=chunk_size, overlap=overlap)
                for j, ch in enumerate(pieces):
                    meta = dict(d.meta)
                    meta["chunk_id"] = j
                    file_chunks.append(Chunk(text=ch, meta=meta))

            # save chunks jsonl
            with chunks_file.open("w", encoding="utf-8") as f:
                for c in file_chunks:
                    f.write(json.dumps({"text": c.text, "meta": c.meta}, ensure_ascii=False) + "\n")

            # embed + normalize for cosine
            if file_chunks:
                embs = embedder.encode([c.text for c in file_chunks], show_progress_bar=False, convert_to_numpy=True)
                embs = embs.astype("float32")
                faiss.normalize_L2(embs)
            else:
                embs = np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype="float32")

            np.save(emb_file, embs)
        # load cached
        file_chunks_loaded: List[Chunk] = []
        with chunks_file.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                file_chunks_loaded.append(Chunk(text=obj["text"], meta=obj["meta"]))
        embs_loaded = np.load(emb_file).astype("float32")

        # align safety
        if len(file_chunks_loaded) != embs_loaded.shape[0]:
            # fallback: recompute quickly (rare)
            docs = load_any(fp)
            file_chunks_loaded = []
            for d in docs:
                pieces = chunk_text(d.text, chunk_size=chunk_size, overlap=overlap)
                for j, ch in enumerate(pieces):
                    meta = dict(d.meta)
                    meta["chunk_id"] = j
                    file_chunks_loaded.append(Chunk(text=ch, meta=meta))
            if file_chunks_loaded:
                embs_loaded = embedder.encode([c.text for c in file_chunks_loaded], show_progress_bar=False, convert_to_numpy=True).astype("float32")
                faiss.normalize_L2(embs_loaded)
            else:
                embs_loaded = np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype="float32")

        all_chunks.extend(file_chunks_loaded)
        if embs_loaded.size:
            all_embs.append(embs_loaded)

    # save hashes
    hashes_path.write_text(json.dumps(new_hashes, ensure_ascii=False, indent=2), encoding="utf-8")

    dim = embedder.get_sentence_embedding_dimension()
    if all_embs:
        mat = np.vstack(all_embs).astype("float32")
    else:
        mat = np.zeros((0, dim), dtype="float32")

    # FAISS cosine (IP sur vecteurs normalisés)
    index = faiss.IndexFlatIP(dim)
    if mat.shape[0] > 0:
        index.add(mat)

    return RAGIndex(index=index, chunks=all_chunks, dim=dim)
