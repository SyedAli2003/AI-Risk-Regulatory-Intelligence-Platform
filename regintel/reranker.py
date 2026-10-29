"""Cross-encoder reranking.

A bi-encoder (FAISS stage) embeds query and passage *separately*, which is fast but coarse.
A cross-encoder reads (query, passage) *together* through full attention, which is far more
accurate but too slow to run over the whole corpus, so it only re-scores the fused shortlist.
"""
from __future__ import annotations

from dataclasses import replace

from .retrieval import Hit


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(
        self,
        query: str,
        hits: list[Hit],
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        reranked = [replace(h, rerank_score=float(s)) for h, s in zip(hits, scores)]
        reranked.sort(key=lambda h: h.rerank_score, reverse=True)
        if min_score is not None:
            reranked = [h for h in reranked if h.rerank_score >= min_score]
        return reranked[:top_k] if top_k else reranked
