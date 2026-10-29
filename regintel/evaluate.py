"""Retrieval evaluation: Precision@K, Recall@K (+ Hit-rate@K and MRR) across retrieval modes.

Ground truth format (JSONL, one object per line):

    {"id": "q001",
     "question": "Who must keep logs for high-risk AI systems?",
     "evidence": ["Providers of high-risk AI systems shall keep the logs automatically generated"],
     "doc_id": "eu_ai_act_2024_1689"}        # optional: restricts matches to that document

`evidence` is a list of short verbatim snippets (8-30 words) that a correct answer depends on.
A retrieved chunk is RELEVANT if it contains (after lowercasing / stripping punctuation) at least one
evidence snippet. Labelling with text snippets instead of chunk ids means your gold set survives
re-chunking, new chunk sizes and new embedding models.

    Precision@K = (# relevant chunks in top K) / K
    Recall@K    = (# distinct evidence snippets covered by the top K chunks) / (# evidence snippets)
    Hit@K       = 1 if any relevant chunk is in the top K
    MRR         = mean of 1 / (rank of first relevant chunk)

Note: if a question has a single relevant chunk, Precision@K can never exceed 1/K. Low-looking
Precision@5 is normal; compare modes against each other rather than against 1.0.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .ingest import Chunk
from .pipeline import MODES, RAGPipeline


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class EvalItem:
    id: str
    question: str
    evidence: list[str]
    doc_id: str | None = None


def load_evalset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            d = json.loads(line)
            ev = d["evidence"] if isinstance(d["evidence"], list) else [d["evidence"]]
            items.append(EvalItem(d.get("id", f"q{n:03d}"), d["question"], ev, d.get("doc_id")))
    return items


def covered_evidence(chunk: Chunk, chunk_norm: str, item_ev_norm: list[str], item: EvalItem) -> set[int]:
    if item.doc_id and chunk.doc_id != item.doc_id:
        return set()
    return {j for j, ev in enumerate(item_ev_norm) if ev and ev in chunk_norm}


def validate_evalset(items: list[EvalItem], chunks: list[Chunk]) -> list[str]:
    """Flag evidence snippets that match NO chunk in the corpus (typo, hyphenation, wrong doc ...).
    Such snippets make recall impossible and silently depress every metric."""
    norms = [norm(c.text) for c in chunks]
    problems: list[str] = []
    for it in items:
        for j, ev in enumerate(it.evidence):
            ev_n = norm(ev)
            ok = any(
                ev_n in cn and (not it.doc_id or c.doc_id == it.doc_id)
                for c, cn in zip(chunks, norms)
            )
            if not ok:
                problems.append(f"{it.id} evidence[{j}] not found in any chunk: {ev[:80]!r}")
    return problems


def evaluate_mode(
    pipe: RAGPipeline,
    items: list[EvalItem],
    mode: str,
    ks: list[int],
    chunk_norms: dict[int, str],
) -> dict:
    max_k = max(ks)
    prec = {k: [] for k in ks}
    rec = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    rr: list[float] = []
    failures: list[str] = []

    for it in items:
        ev_norm = [norm(e) for e in it.evidence]
        hits = pipe.retrieve(it.question, mode=mode, k=max_k)
        cov = [covered_evidence(h.chunk, chunk_norms[h.chunk.idx], ev_norm, it) for h in hits]

        first = next((r for r, c in enumerate(cov, start=1) if c), None)
        rr.append(1.0 / first if first else 0.0)
        if first is None:
            failures.append(it.id)

        for k in ks:
            top = cov[:k]
            n_rel = sum(1 for c in top if c)
            got = set().union(*top) if top else set()
            prec[k].append(n_rel / k)
            rec[k].append(len(got) / len(ev_norm))
            hit[k].append(1.0 if n_rel else 0.0)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return {
        "mode": mode,
        "n_questions": len(items),
        "precision": {k: mean(v) for k, v in prec.items()},
        "recall": {k: mean(v) for k, v in rec.items()},
        "hit": {k: mean(v) for k, v in hit.items()},
        "mrr": mean(rr),
        "failures_at_max_k": failures,
    }


def run_evaluation(
    pipe: RAGPipeline,
    evalset_path: Path,
    ks: list[int],
    modes: list[str] | None = None,
    out_path: Path | None = None,
) -> list[dict]:
    items = load_evalset(evalset_path)
    if not items:
        raise ValueError(f"No questions in {evalset_path}")

    problems = validate_evalset(items, pipe.chunks)
    if problems:
        print(f"[eval] WARNING: {len(problems)} evidence snippet(s) match no chunk. First few:")
        for p in problems[:5]:
            print("   ", p)
        print("       Fix or remove them; they cap achievable recall.\n")

    chunk_norms = {c.idx: norm(c.text) for c in pipe.chunks}
    modes = modes or list(MODES)
    results = []
    for m in modes:
        print(f"[eval] {m} ...")
        results.append(evaluate_mode(pipe, items, m, ks, chunk_norms))

    print_table(results, ks)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved {out_path}")
    return results


def print_table(results: list[dict], ks: list[int]) -> None:
    n = results[0]["n_questions"]
    head = f"{'mode':<15}" + "".join(f"P@{k:<5}" for k in ks) + "| " + "".join(f"R@{k:<5}" for k in ks) + "| MRR"
    print(f"\nRetrieval quality on {n} questions\n" + head + "\n" + "-" * len(head))
    for r in results:
        row = f"{r['mode']:<15}"
        row += "".join(f"{r['precision'][k]:<7.3f}" for k in ks) + "| "
        row += "".join(f"{r['recall'][k]:<7.3f}" for k in ks) + "| "
        row += f"{r['mrr']:.3f}"
        print(row)
