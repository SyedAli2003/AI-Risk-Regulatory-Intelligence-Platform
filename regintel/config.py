"""Central configuration. Every knob lives here so ablations are one-line changes."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- .env loading
# Built in (no python-dotenv needed) so a missing package can never silently skip your settings.
def parse_env_text(raw: bytes) -> dict[str, str]:
    """Parse KEY=VALUE lines. Handles UTF-8 BOM and UTF-16 (Windows Notepad/PowerShell), `export `,
    quotes, blank lines, full-line comments and inline ' # comments'."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key:
            continue
        if val[:1] in {"'", '"'} and val.find(val[0], 1) != -1:
            val = val[1 : val.index(val[0], 1)]            # quoted: keep everything inside the quotes
        else:
            val = re.sub(r"\s+#.*$", "", val).strip()       # unquoted: drop an inline comment
        values[key] = val
    return values


def load_env(root: Path = ROOT) -> dict:
    """Load <root>/.env into os.environ (real environment variables win). Returns diagnostics."""
    info: dict = {"path": None, "keys": [], "note": f"no .env file found in {root}"}
    for name in (".env", ".env.txt"):          # Notepad on Windows often saves ".env" as ".env.txt"
        path = root / name
        if not path.is_file():
            continue
        try:
            values = parse_env_text(path.read_bytes())
        except OSError as e:
            info["note"] = f"could not read {path}: {e}"
            continue
        for k, v in values.items():
            os.environ.setdefault(k, v)
        info.update(path=str(path), keys=sorted(values), note="")
        if name == ".env.txt":
            info["note"] = "loaded .env.txt; rename it to exactly .env (Notepad adds .txt)"
        return info
    return info


ENV_INFO = load_env()

# Retrieval strategies, kept here (no heavy imports) so the CLI/menu can list them before dependencies are installed.
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Config:
    # ---- paths -------------------------------------------------------------
    raw_dir: Path = ROOT / "data" / "raw"
    index_dir: Path = ROOT / "data" / "index"
    eval_dir: Path = ROOT / "data" / "eval"

    # ---- chunking (measured in whitespace-separated words) -----------------
    chunk_words: int = 220        # ~300 tokens: fits comfortably in the cross-encoder's 512-token window
    chunk_overlap: int = 40       # overlap so clauses split at a boundary stay retrievable

    # ---- dense retrieval ---------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # BGE models are trained with an instruction prefix on the *query* side only.
    # Set to "" if you switch to a model that doesn't use one (e.g. all-MiniLM-L6-v2).
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 64

    # ---- BM25 --------------------------------------------------------------
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- hybrid fusion -----------------------------------------------------
    candidates_per_retriever: int = 50   # depth pulled from BM25 and from FAISS before fusion
    fusion: str = "rrf"                  # "rrf" (reciprocal rank fusion) or "weighted" (min-max score blend)
    rrf_k: int = 60                      # standard RRF constant
    bm25_weight: float = 1.0             # RRF per-retriever weights
    dense_weight: float = 1.0
    dense_alpha: float = 0.5             # weight on dense score when fusion == "weighted"

    # ---- cross-encoder reranker -------------------------------------------
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # alt: "BAAI/bge-reranker-base" (stronger, slower)
    rerank_candidates: int = 30          # how many fused candidates get re-scored
    rerank_batch_size: int = 32
    rerank_max_length: int = 512
    # Optional abstention gate: if the best passage scores below this, don't call the LLM.
    # Score scale depends on the model/activation, so calibrate on your own eval set before using.
    min_rerank_score: float | None = None

    # ---- generation --------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    hf_llm_model: str = field(default_factory=lambda: os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    top_k_context: int = 5               # passages handed to the LLM
    max_answer_tokens: int = 700

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.index_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"
