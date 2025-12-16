from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any
import hashlib

from pypdf import PdfReader

@dataclass
class Document:
    text: str
    meta: Dict[str, Any]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_files(data_dir: Path) -> Iterator[Path]:
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}:
            yield p

def load_txt(path: Path) -> List[Document]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(text=txt, meta={"path": str(path), "page": None})]

def load_pdf(path: Path) -> List[Document]:
    docs: List[Document] = []
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            docs.append(Document(text=text, meta={"path": str(path), "page": i + 1}))
    return docs

def load_any(path: Path) -> List[Document]:
    if path.suffix.lower() == ".txt":
        return load_txt(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    return []
