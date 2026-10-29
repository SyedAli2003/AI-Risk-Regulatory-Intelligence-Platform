"""Load PDFs/text, clean them, and split into overlapping, page-tracked chunks."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config


@dataclass
class Chunk:
    idx: int            # row id: position in the BM25 corpus and the FAISS index
    doc_id: str         # file stem
    title: str          # human-readable document title (from sources.json when available)
    page_start: int     # 1-based PDF page index of the first word
    page_end: int       # 1-based PDF page index of the last word
    text: str
    url: str = ""

    @property
    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    @property
    def citation(self) -> str:
        return f"{self.title}, {self.pages}"


# --------------------------------------------------------------------------- reading
def read_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # corrupt page: keep page numbering aligned
            pages.append("")
    return pages


def read_text_pages(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.split("\f") if "\f" in text else [text]


# --------------------------------------------------------------------------- cleaning
_PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_DIGITS_RE = re.compile(r"\d+")
# lines like "(#)", "(a)", "#" are list markers / page numbers, never running headers
_TRIVIAL_RE = re.compile(r"^\(?[#a-z]{1,3}\)?\.?$")


def clean_pages(pages: list[str]) -> list[str]:
    """Strip running headers/footers and page numbers, fix hyphenation, normalise whitespace.

    Running headers are detected statistically: a short line (digits masked) that appears on
    >= 50% of pages in a document of >= 6 pages is treated as boilerplate.
    """
    pages = [unicodedata.normalize("NFKC", p) for p in pages]  # also expands ligatures (ﬁ -> fi)
    n = len(pages)

    counts: Counter[str] = Counter()
    for p in pages:
        seen: set[str] = set()
        for line in p.splitlines():
            s = line.strip()
            if not s or len(s) > 100:
                continue
            norm = _DIGITS_RE.sub("#", s.lower())
            if _TRIVIAL_RE.match(norm):
                continue
            seen.add(norm)
        counts.update(seen)
    boilerplate = {k for k, c in counts.items() if c / n >= 0.5} if n >= 6 else set()

    cleaned: list[str] = []
    for p in pages:
        kept: list[str] = []
        for line in p.splitlines():
            s = line.strip()
            if not s:
                kept.append("")
                continue
            if _PAGE_NUM_RE.match(s):
                continue
            if _DIGITS_RE.sub("#", s.lower()) in boilerplate:
                continue
            kept.append(s)
        text = "\n".join(kept)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # "regu-\nlation" -> "regulation"
        text = re.sub(r"\s*\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        cleaned.append(text)
    return cleaned


# --------------------------------------------------------------------------- chunking
def chunk_pages(pages: list[str], chunk_words: int, overlap: int) -> list[tuple[str, int, int]]:
    """Sliding word window over the whole document (so clauses can span pages).

    Returns (text, page_start, page_end). Windows prefer to end on a sentence boundary
    within the last 25% of the window, which keeps chunks readable for the LLM.
    """
    words: list[str] = []
    word_page: list[int] = []
    for pno, text in enumerate(pages, start=1):
        for w in text.split():
            words.append(w)
            word_page.append(pno)
    n = len(words)
    if n == 0:
        return []

    out: list[tuple[str, int, int]] = []
    start = 0
    while start < n:
        hard_end = min(start + chunk_words, n)
        end = hard_end
        if hard_end < n:
            lo = start + int(chunk_words * 0.75)
            for j in range(hard_end, lo, -1):          # j = exclusive end index
                if words[j - 1].endswith((".", ";", "?", "!")):
                    end = j
                    break
        out.append((" ".join(words[start:end]), word_page[start], word_page[end - 1]))
        if end >= n:
            break
        start = max(end - overlap, start + 1)           # always make forward progress
    return out


# --------------------------------------------------------------------------- corpus
def build_chunks(cfg: Config | None = None) -> list[Chunk]:
    cfg = cfg or Config()
    meta_path = cfg.raw_dir / "sources.json"
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    files = sorted(p for p in cfg.raw_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt"})
    if not files:
        raise FileNotFoundError(
            f"No .pdf/.txt files in {cfg.raw_dir}. Run `python main.py download` or add your own documents."
        )

    chunks: list[Chunk] = []
    for path in files:
        pages = read_pdf_pages(path) if path.suffix.lower() == ".pdf" else read_text_pages(path)
        pages = clean_pages(pages)
        info = meta.get(path.name, {})
        title = info.get("title") or path.stem.replace("_", " ").replace("-", " ")
        url = info.get("url", "")
        pieces = chunk_pages(pages, cfg.chunk_words, cfg.chunk_overlap)
        for text, p0, p1 in pieces:
            chunks.append(Chunk(len(chunks), path.stem, title, p0, p1, text, url))
        print(f"[ingest] {path.name}: {len(pages)} pages -> {len(pieces)} chunks")
    print(f"[ingest] total chunks: {len(chunks)}")
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
