"""Hybrid retrieval: BM25 (lexical) + FAISS (semantic), fused with RRF or a weighted blend."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .index import BM25Index, DenseIndex
from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float                      # score from the stage that produced this ordering
    bm25_rank: int | None = None      # 1-based rank in BM25 results (None if not retrieved by BM25)
    dense_rank: int | None = None     # 1-based rank in dense results
    rerank_score: float | None = None # cross-encoder score (set by the reranker)


def rrf_fuse(
    ranked_lists: list[list[int]], weights: list[float], k: int = 60
) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)). Rank-based, so no score calibration needed."""
    fused: dict[int, float] = {}
    for ids, w in zip(ranked_lists, weights):
        for rank, i in enumerate(ids, start=1):
            fused[i] = fused.get(i, 0.0) + w / (k + rank)
    return fused


def minmax(pairs: list[tuple[int, float]]) -> dict[int, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {i: 1.0 for i, _ in pairs}
    return {i: (s - lo) / (hi - lo) for i, s in pairs}


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], bm25: BM25Index, dense: DenseIndex, cfg: Config):
        self.chunks, self.bm25, self.dense, self.cfg = chunks, bm25, dense, cfg

    # ---- single-retriever searches (used for ablations) ----------------------
    def search_bm25(self, query: str, k: int) -> list[Hit]:
        return [
            Hit(self.chunks[i], s, bm25_rank=r)
            for r, (i, s) in enumerate(self.bm25.search(query, k), start=1)
        ]

    def search_dense(self, query: str, k: int) -> list[Hit]:
        return [
            Hit(self.chunks[i], s, dense_rank=r)
            for r, (i, s) in enumerate(self.dense.search(query, k), start=1)
        ]

    # ---- hybrid ---------------------------------------------------------------
    def search_hybrid(self, query: str, k: int) -> list[Hit]:
        n = self.cfg.candidates_per_retriever
        bm = self.bm25.search(query, n)
        de = self.dense.search(query, n)
        bm_rank = {i: r for r, (i, _) in enumerate(bm, start=1)}
        de_rank = {i: r for r, (i, _) in enumerate(de, start=1)}

        if self.cfg.fusion == "rrf":
            fused = rrf_fuse(
                [[i for i, _ in bm], [i for i, _ in de]],
                [self.cfg.bm25_weight, self.cfg.dense_weight],
                self.cfg.rrf_k,
            )
        elif self.cfg.fusion == "weighted":
            nb, nd, a = minmax(bm), minmax(de), self.cfg.dense_alpha
            fused = {i: a * nd.get(i, 0.0) + (1 - a) * nb.get(i, 0.0) for i in set(nb) | set(nd)}
        else:
            raise ValueError(f"unknown fusion method: {self.cfg.fusion!r}")

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Hit(self.chunks[i], s, bm25_rank=bm_rank.get(i), dense_rank=de_rank.get(i))
            for i, s in top
        ]
