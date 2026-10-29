"""End-to-end pipeline: retrieve -> (rerank) -> grounded generation -> validated citations."""
from __future__ import annotations

import re
import textwrap
from dataclasses import asdict, dataclass, field

from .config import MODES, Config
from .index import BM25Index, DenseIndex
from .ingest import load_chunks
from .llm import LLM, get_llm
from .reranker import CrossEncoderReranker
from .retrieval import Hit, HybridRetriever

NO_ANSWER = "The provided documents do not contain enough information to answer this question."

SYSTEM_PROMPT = f"""You are a regulatory and risk-compliance research assistant. \
You answer questions using ONLY the numbered context passages supplied by the user.

Rules:
1. Ground every statement in the passages. Do not use outside knowledge and do not guess.
2. After each claim, cite the supporting passage number(s) in square brackets, e.g. [2] or [1][3]. \
Only cite numbers that appear in the context.
3. If the passages do not contain enough information, reply with exactly: "{NO_ANSWER}" \
followed by one sentence saying what is missing.
4. Never invent article or section numbers, dates, thresholds, fines or definitions that are not in the passages.
5. Preserve normative strength: "shall"/"must" = obligation, "should" = recommendation, "may" = permission.
6. If passages from different documents differ or conflict, say so and attribute each view to its source.
7. Be concise and well organised."""

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


# --------------------------------------------------------------------------- result types
@dataclass
class Citation:
    ref: int
    doc_id: str
    title: str
    pages: str
    url: str
    chunk_idx: int
    rerank_score: float | None
    snippet: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[Citation]                      # only sources the answer actually cites
    context: list[Hit] = field(default_factory=list)  # everything the LLM saw, in prompt order
    invalid_refs: list[int] = field(default_factory=list)
    abstained: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["context"] = [
            {
                "ref": i,
                "chunk_idx": h.chunk.idx,
                "citation": h.chunk.citation,
                "rerank_score": h.rerank_score,
                "text": h.chunk.text,
            }
            for i, h in enumerate(self.context, start=1)
        ]
        return d

    def pretty(self, show_context: bool = False) -> str:
        lines = [f"Q: {self.question}", "", "A: " + self.answer, ""]
        if self.citations:
            lines.append("Sources cited:")
            for c in self.citations:
                score = f"  (rerank {c.rerank_score:.2f})" if c.rerank_score is not None else ""
                lines.append(f"  [{c.ref}] {c.title}, {c.pages}{score}")
                if c.url:
                    lines.append(f"      {c.url}")
                lines.append("      " + textwrap.shorten(c.snippet, width=170, placeholder=" ..."))
        for w in self.warnings:
            lines.append(f"\n! {w}")
        if show_context:
            lines.append("\n--- retrieved context ---")
            for i, h in enumerate(self.context, start=1):
                lines.append(f"[{i}] {h.chunk.citation}\n{h.chunk.text}\n")
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers
def build_user_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [
        f"[{i}] Source: {h.chunk.title} ({h.chunk.pages})\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    ]
    return (
        "Context passages:\n\n"
        + "\n\n".join(blocks)
        + f"\n\nQuestion: {question}\n\nAnswer using only the context above, with bracketed citations."
    )


def extract_refs(answer: str, n_context: int) -> tuple[list[int], list[int]]:
    """Return (valid_refs_in_order_of_first_use, invalid_refs). Handles [1], [1][3] and [1, 3]."""
    valid: list[int] = []
    invalid: list[int] = []
    for m in _CITE_RE.finditer(answer):
        for tok in re.split(r"[,;]", m.group(1)):
            r = int(tok)
            bucket = valid if 1 <= r <= n_context else invalid
            if r not in bucket:
                bucket.append(r)
    return valid, invalid


# --------------------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(self, cfg: Config | None = None, llm: LLM | None = None):
        self.cfg = cfg or Config()
        if not self.cfg.chunks_path.exists():
            raise FileNotFoundError(
                f"{self.cfg.chunks_path} not found. Build the index first (menu option 3, or `python main.py index`)."
            )
        self.chunks = load_chunks(self.cfg.chunks_path)
        self.bm25 = BM25Index.load(self.cfg.bm25_path)
        self.dense = DenseIndex.load(self.cfg.index_dir, self.cfg.embed_batch_size)
        self.retriever = HybridRetriever(self.chunks, self.bm25, self.dense, self.cfg)
        self._reranker: CrossEncoderReranker | None = None
        self._llm = llm

    # lazy so that `eval` of bm25/dense/hybrid never loads the reranker, and retrieval-only runs never need an LLM key
    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                self.cfg.rerank_model, self.cfg.rerank_batch_size, self.cfg.rerank_max_length
            )
        return self._reranker

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = get_llm(self.cfg)
        return self._llm

    # ---- retrieval ----------------------------------------------------------
    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[Hit]:
        if mode == "bm25":
            return self.retriever.search_bm25(query, k)
        if mode == "dense":
            return self.retriever.search_dense(query, k)
        if mode == "hybrid":
            return self.retriever.search_hybrid(query, k)
        if mode == "hybrid_rerank":
            candidates = self.retriever.search_hybrid(query, max(self.cfg.rerank_candidates, k))
            return self.reranker.rerank(query, candidates, top_k=k, min_score=self.cfg.min_rerank_score)
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    # ---- generation ---------------------------------------------------------
    def ask(self, question: str, k: int | None = None, mode: str = "hybrid_rerank") -> RAGAnswer:
        k = k or self.cfg.top_k_context
        hits = self.retrieve(question, mode=mode, k=k)
        if not hits:
            return RAGAnswer(
                question,
                NO_ANSWER + " No sufficiently relevant passages were retrieved.",
                citations=[],
                abstained=True,
            )

        raw = self.llm.complete(
            SYSTEM_PROMPT, build_user_prompt(question, hits), max_tokens=self.cfg.max_answer_tokens
        )
        abstained = NO_ANSWER.lower() in raw.lower()
        valid, invalid = extract_refs(raw, len(hits))

        citations = [
            Citation(
                ref=r,
                doc_id=hits[r - 1].chunk.doc_id,
                title=hits[r - 1].chunk.title,
                pages=hits[r - 1].chunk.pages,
                url=hits[r - 1].chunk.url,
                chunk_idx=hits[r - 1].chunk.idx,
                rerank_score=hits[r - 1].rerank_score,
                snippet=hits[r - 1].chunk.text,
            )
            for r in valid
        ] if not abstained else []

        warnings: list[str] = []
        if invalid:
            warnings.append(f"Answer cites passage numbers that were not in the context: {invalid}.")
        if not abstained and not citations:
            warnings.append("Answer contains no valid citations; treat it as unverified.")

        return RAGAnswer(question, raw, citations, hits, invalid, abstained, warnings)
