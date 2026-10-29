"""Model-free unit tests: run with `python -m unittest discover -s tests` (no GPU, no downloads)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regintel.config import load_env, parse_env_text  # noqa: E402
from regintel.evaluate import EvalItem, evaluate_mode, norm, validate_evalset  # noqa: E402
from regintel.index import tokenize  # noqa: E402
from regintel.ingest import Chunk, chunk_pages, clean_pages  # noqa: E402
from regintel.pipeline import extract_refs  # noqa: E402
from regintel.retrieval import Hit, minmax, rrf_fuse  # noqa: E402


class TestEnvLoading(unittest.TestCase):
    def test_inline_comment_quotes_export_blank_lines(self):
        raw = b"LLM_PROVIDER=hf\n\n# note\nexport A=1\nHF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct   # any model\nQ=\"x # y\"\n"
        v = parse_env_text(raw)
        self.assertEqual(v["LLM_PROVIDER"], "hf")
        self.assertEqual(v["HF_LLM_MODEL"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(v["A"], "1")
        self.assertEqual(v["Q"], "x # y")

    def test_windows_encodings(self):
        self.assertEqual(parse_env_text(b"\xef\xbb\xbfLLM_PROVIDER=hf\r\n")["LLM_PROVIDER"], "hf")          # UTF-8 BOM
        self.assertEqual(parse_env_text("LLM_PROVIDER=hf\r\n".encode("utf-16"))["LLM_PROVIDER"], "hf")        # PowerShell

    def test_load_env_files_and_precedence(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env.txt").write_text("ZZ_TEST_A=from_file\nZZ_TEST_B=from_file\n")
            os.environ["ZZ_TEST_B"] = "from_os"
            os.environ.pop("ZZ_TEST_A", None)
            info = load_env(Path(d))
            self.assertEqual(os.environ["ZZ_TEST_A"], "from_file")
            self.assertEqual(os.environ["ZZ_TEST_B"], "from_os")     # real env vars win
            self.assertIn("rename", info["note"])                    # .env.txt warning
            empty = load_env(Path(d) / "nope")
            self.assertIsNone(empty["path"])
        os.environ.pop("ZZ_TEST_A", None)
        os.environ.pop("ZZ_TEST_B", None)


class TestIngest(unittest.TestCase):
    def test_chunk_overlap_and_page_tracking(self):
        pages = [" ".join(f"w{i}" for i in range(0, 300)), " ".join(f"w{i}" for i in range(300, 500))]
        chunks = chunk_pages(pages, chunk_words=100, overlap=20)
        self.assertGreater(len(chunks), 4)
        # every word is covered, first chunk starts on page 1, last ends on page 2
        joined = " ".join(c[0] for c in chunks)
        for i in (0, 150, 299, 300, 499):
            self.assertIn(f"w{i}", joined.split())
        self.assertEqual(chunks[0][1], 1)
        self.assertEqual(chunks[-1][2], 2)
        # consecutive chunks overlap
        a, b = chunks[0][0].split(), chunks[1][0].split()
        self.assertTrue(set(a[-10:]) & set(b[:25]))

    def test_chunking_terminates_on_tiny_input(self):
        self.assertEqual(chunk_pages([""], 100, 20), [])
        self.assertEqual(len(chunk_pages(["just a few words"], 100, 20)), 1)

    def test_clean_pages_strips_headers_page_numbers_hyphenation(self):
        topics = "alpha beta gamma delta epsilon zeta eta theta".split()
        pages = [f"ACME REGULATION 2024\nBody text about {t} with regu-\nlation inside the {t} clause.\nPage {i} of 8" for i, t in enumerate(topics)]
        cleaned = clean_pages(pages)
        for p in cleaned:
            self.assertNotIn("ACME REGULATION", p)
            self.assertNotIn("Page", p)
            self.assertIn("regulation inside", p)

    def test_ligatures_normalised(self):
        self.assertIn("financial", clean_pages(["ﬁnancial risk"])[0])


class TestRetrieval(unittest.TestCase):
    def test_tokenize_splits_hyphens_and_drops_stopwords(self):
        toks = tokenize("The high-risk systems")
        self.assertIn("high", toks)
        self.assertNotIn("the", toks)

    def test_rrf_prefers_docs_found_by_both(self):
        fused = rrf_fuse([[1, 2, 3], [3, 4, 1]], [1.0, 1.0], k=60)
        ranked = sorted(fused, key=fused.get, reverse=True)
        self.assertEqual(set(ranked[:2]), {1, 3})
        self.assertEqual(set(fused), {1, 2, 3, 4})

    def test_minmax(self):
        self.assertEqual(minmax([]), {})
        self.assertEqual(minmax([(1, 5.0), (2, 5.0)]), {1: 1.0, 2: 1.0})
        m = minmax([(1, 0.0), (2, 10.0)])
        self.assertEqual((m[1], m[2]), (0.0, 1.0))


class TestCitations(unittest.TestCase):
    def test_extract_refs_formats(self):
        valid, invalid = extract_refs("A [1]. B [2][3]. C [1, 4]. D [9].", n_context=4)
        self.assertEqual(valid, [1, 2, 3, 4])
        self.assertEqual(invalid, [9])


class FakePipe:
    """Returns a fixed ranking so metric arithmetic can be checked by hand."""

    def __init__(self, chunks, ranking):
        self.chunks, self.ranking = chunks, ranking

    def retrieve(self, query, mode="x", k=5):
        return [Hit(self.chunks[i], 1.0) for i in self.ranking[query][:k]]


class TestMetrics(unittest.TestCase):
    def setUp(self):
        texts = [
            "providers shall keep logs for six months",   # 0 -> evidence A
            "unrelated text about something else",        # 1
            "deployers must monitor the system",           # 2 -> evidence B
            "more unrelated filler content here",          # 3
        ]
        self.chunks = [Chunk(i, "doc", "Doc", 1, 1, t) for i, t in enumerate(texts)]
        self.norms = {c.idx: norm(c.text) for c in self.chunks}

    def test_precision_recall_hand_computed(self):
        item = EvalItem("q1", "q1", ["shall keep logs", "must monitor the system"])
        pipe = FakePipe(self.chunks, {"q1": [1, 0, 3, 2]})
        r = evaluate_mode(pipe, [item], "x", [1, 2, 4], self.norms)
        self.assertAlmostEqual(r["precision"][1], 0.0)
        self.assertAlmostEqual(r["precision"][2], 0.5)     # one relevant in top-2
        self.assertAlmostEqual(r["precision"][4], 0.5)     # two relevant in top-4
        self.assertAlmostEqual(r["recall"][1], 0.0)
        self.assertAlmostEqual(r["recall"][2], 0.5)        # 1 of 2 evidence snippets
        self.assertAlmostEqual(r["recall"][4], 1.0)
        self.assertAlmostEqual(r["mrr"], 0.5)              # first relevant at rank 2

    def test_validate_flags_unmatched_evidence(self):
        good = EvalItem("g", "q", ["deployers must monitor"])
        bad = EvalItem("b", "q", ["this text is nowhere"])
        problems = validate_evalset([good, bad], self.chunks)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("b "))


if __name__ == "__main__":
    unittest.main()
