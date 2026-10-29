"""Bootstrap an evaluation set by asking an LLM to write a question + verbatim evidence sentence per sampled chunk.

Caveat (important, and worth saying in an interview): questions generated *from* a passage tend to share
vocabulary with it, which flatters BM25. Use this to get a quick 50-100 question baseline, then hand-write
or hand-edit 30+ questions in analysts' own wording for the numbers you report.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .evaluate import norm
from .ingest import Chunk, load_chunks
from .llm import LLM

GEN_SYSTEM = (
    "You write evaluation questions for a regulatory and risk-compliance search engine. "
    "Respond with a single JSON object and nothing else."
)

GEN_USER = """Document: "{title}"

Passage:
\"\"\"{text}\"\"\"

Write ONE realistic question a risk or compliance analyst might ask that this passage answers.
Requirements:
- The question must be self-contained: never say "the passage", "this document" or "the text".
- Prefer an analyst's natural wording over copying phrases from the passage.
- Also copy VERBATIM one sentence or clause (10-35 words) from the passage that contains the answer.

Return exactly: {{"question": "...", "evidence": "..."}}"""


def _stratified_sample(chunks: list[Chunk], n: int, rng: random.Random, min_words: int = 100) -> list[Chunk]:
    """Round-robin across documents so one large document doesn't dominate the eval set."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if len(c.text.split()) >= min_words:
            by_doc[c.doc_id].append(c)
    for lst in by_doc.values():
        rng.shuffle(lst)
    docs = sorted(by_doc)
    picked: list[Chunk] = []
    while len(picked) < n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(picked) < n:
                picked.append(by_doc[d].pop())
    return picked


def _parse_json(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def make_evalset(
    chunks_path: Path,
    llm: LLM,
    out_path: Path,
    n: int = 60,
    seed: int = 42,
) -> int:
    chunks = load_chunks(chunks_path)
    rng = random.Random(seed)
    # oversample: some generations will fail validation
    candidates = _stratified_sample(chunks, int(n * 1.5), rng)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            if kept >= n:
                break
            raw = llm.complete(
                GEN_SYSTEM,
                GEN_USER.format(title=c.title, text=c.text),
                max_tokens=300,
                temperature=0.2,
            )
            d = _parse_json(raw)
            if not d or not isinstance(d.get("question"), str) or not isinstance(d.get("evidence"), str):
                continue
            q, ev = d["question"].strip(), d["evidence"].strip()
            if len(q) < 15 or len(ev.split()) < 6:
                continue
            if norm(ev) not in norm(c.text):      # evidence must really be in the chunk (anti-hallucination)
                continue
            kept += 1
            f.write(
                json.dumps(
                    {"id": f"syn-{kept:03d}", "question": q, "evidence": [ev], "doc_id": c.doc_id},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"[evalset] {kept}/{n}: {q}")
    print(f"[evalset] wrote {kept} questions -> {out_path}")
    return kept
