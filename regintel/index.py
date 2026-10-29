"""Sparse (BM25) and dense (sentence-transformer embeddings + FAISS) indexes."""
from __future__ import annotations

import json
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Config
from .ingest import Chunk, save_chunks

# --------------------------------------------------------------------------- tokenisation
# Deliberately NOT removing "shall", "may", "must", "not": they carry obligation semantics in regulation.
STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have if in into is it its of on or such "
    "than that the their then there these they this to was were which will with".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

try:
    from nltk.stem import PorterStemmer

    _stemmer = PorterStemmer()

    @lru_cache(maxsize=200_000)
    def _stem(tok: str) -> str:
        return _stemmer.stem(tok)

except ImportError:  # pragma: no cover - nltk is in requirements.txt

    def _stem(tok: str) -> str:
        return tok


def tokenize(text: str) -> list[str]:
    """lowercase -> alnum tokens (hyphens split: 'high-risk' -> high, risk) -> stopword filter -> stem."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


# --------------------------------------------------------------------------- BM25
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.bm25 = None

    def build(self, texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=self.k1, b=self.b)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = np.asarray(self.bm25.get_scores(toks))
        return [(int(i), float(scores[i])) for i in _top_k(scores, k) if scores[i] > 0]

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            blob = pickle.load(f)  # only load index files you built yourself
        obj = cls(blob["k1"], blob["b"])
        obj.bm25 = blob["bm25"]
        return obj


# --------------------------------------------------------------------------- dense / FAISS
class DenseIndex:
    """Bi-encoder embeddings, L2-normalised, searched by inner product (= cosine similarity)."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 64):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = None
        self.index = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, texts: list[str]) -> None:
        import faiss

        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar="idlelib" not in sys.modules,   # progress bars render badly in IDLE
            convert_to_numpy=True,
        ).astype("float32")
        # Exact search. For > ~1M chunks switch to faiss.IndexHNSWFlat or IndexIVFFlat.
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, directory: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(directory / "faiss.index"))
        (directory / "dense_meta.json").write_text(
            json.dumps({"model_name": self.model_name, "query_prefix": self.query_prefix})
        )

    @classmethod
    def load(cls, directory: Path, batch_size: int = 64) -> "DenseIndex":
        import faiss

        meta = json.loads((directory / "dense_meta.json").read_text())
        obj = cls(meta["model_name"], meta["query_prefix"], batch_size)
        obj.index = faiss.read_index(str(directory / "faiss.index"))
        return obj


# --------------------------------------------------------------------------- build all
def build_all(cfg: Config, chunks: list[Chunk]) -> None:
    cfg.ensure_dirs()
    texts = [c.text for c in chunks]

    save_chunks(chunks, cfg.chunks_path)

    print("[index] building BM25 ...")
    bm25 = BM25Index(cfg.bm25_k1, cfg.bm25_b)
    bm25.build(texts)
    bm25.save(cfg.bm25_path)

    print(f"[index] embedding {len(texts)} chunks with {cfg.embed_model} ...")
    dense = DenseIndex(cfg.embed_model, cfg.embed_query_prefix, cfg.embed_batch_size)
    dense.build(texts)
    dense.save(cfg.index_dir)
    print(f"[index] done -> {cfg.index_dir}")
