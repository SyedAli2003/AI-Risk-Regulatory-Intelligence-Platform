#!/usr/bin/env python
"""AI Risk & Regulatory Intelligence Platform: entry point for IDLE *and* the command line.

FROM IDLE (no command line needed)
    Open this file in IDLE and press F5 (Run > Run Module). A numbered menu appears in the Shell:
        1 install dependencies   2 download documents   3 build index   4 ask a question
        5 chat                   6 make eval set         7 evaluate      8 unit tests   s status
    Want to run one fixed command instead of the menu? Set IDLE_ARGS just below, e.g.
        IDLE_ARGS = ["ask", "What does the EU AI Act require for high-risk systems?"]

FROM A TERMINAL (unchanged)
    python main.py install | download | index | status | test
    python main.py ask "What does the EU AI Act require for high-risk systems?" [--mode bm25] [--show-context] [--json]
    python main.py chat
    python main.py make-evalset --n 60
    python main.py eval --ks 1 3 5 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                   # relative paths behave the same wherever IDLE was launched from
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# IDLE: optionally run ONE command when you press F5, instead of showing the menu.
# Leave as None for the menu. Terminal arguments always take priority over this.
IDLE_ARGS: list[str] | None = None
# IDLE_ARGS = ["ask", "What are the obligations of providers of high-risk AI systems?"]
# IDLE_ARGS = ["eval", "--ks", "1", "3", "5", "10"]
# ---------------------------------------------------------------------------

if "idlelib" in sys.modules:                     # progress bars (carriage returns) render as clutter in IDLE
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from regintel.config import MODES, Config  # noqa: E402  (light import: no heavy deps needed yet)

_PIPELINE = None   # cached so the menu doesn't reload models on every question


def get_pipeline(cfg: Config):
    global _PIPELINE
    if _PIPELINE is None:
        from regintel.pipeline import RAGPipeline

        print("[setup] loading indexes and models (first run downloads models; this can take a few minutes) ...")
        _PIPELINE = RAGPipeline(cfg)
    return _PIPELINE


# =========================================================================== commands
def cmd_install(args, cfg: Config) -> None:
    req = ROOT / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    print("[install]", " ".join(cmd))
    print("[install] this downloads several hundred MB (PyTorch); please wait ...\n")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash on Windows; 0 elsewhere
    )
    for line in proc.stdout:                     # stream pip's output into the IDLE shell
        print(line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"pip exited with code {rc}; see the log above")
    print("\n[install] done. Next: option 2 (download documents), then 3 (build index).")


def _env_report() -> list[str]:
    """Human-readable result of the .env loading step."""
    from regintel.config import ENV_INFO

    if ENV_INFO["path"]:
        lines = [f"loaded {ENV_INFO['path']}", f"settings found: {', '.join(ENV_INFO['keys']) or '(none: file is empty?)'}"]
    else:
        lines = [ENV_INFO["note"]]
    if ENV_INFO["path"] and ENV_INFO["note"]:
        lines.append(ENV_INFO["note"])
    return lines


def cmd_status(args, cfg: Config) -> None:
    import importlib.util

    ok = lambda b: "OK     " if b else "MISSING"  # noqa: E731
    docs = [p for p in cfg.raw_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt"}]
    index_ready = cfg.chunks_path.exists() and cfg.bm25_path.exists() and (cfg.index_dir / "faiss.index").exists()
    evalset = cfg.eval_dir / "eval_set.jsonl"

    needed = {
        "pypdf": "pypdf", "rank_bm25": "rank-bm25", "nltk": "nltk", "numpy": "numpy",
        "sentence_transformers": "sentence-transformers", "faiss": "faiss-cpu", "requests": "requests",
    }
    needed.update({"anthropic": "anthropic"} if cfg.llm_provider == "anthropic" else {"transformers": "transformers", "torch": "torch"})
    missing = [pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None]

    print("Setup status")
    print(f"  {ok(not missing)} Python packages" + (f"  (missing: {', '.join(missing)})" if missing else ""))
    print(f"  {ok(bool(docs))} documents in data/raw/        ({len(docs)} files)")
    print(f"  {ok(index_ready)} search indexes in data/index/")
    print(f"  {ok(evalset.exists())} evaluation set            (data/eval/eval_set.jsonl)")
    from regintel.config import ENV_INFO

    print(f"  {ok(bool(ENV_INFO['path']))} .env file")
    for line in _env_report():
        print(f"            {line}")
    print(f"  LLM_PROVIDER = {cfg.llm_provider}")
    if cfg.llm_provider == "anthropic":
        print(f"  {ok(bool(os.getenv('ANTHROPIC_API_KEY')))} ANTHROPIC_API_KEY  (model: {cfg.anthropic_model})")
    else:
        print(f"  using local model {cfg.hf_llm_model}")

    if missing:
        print("\nNext step: option 1 (install dependencies).")
    elif not docs:
        print("\nNext step: option 2 (download documents), or copy your own PDFs into data/raw/.")
    elif not index_ready:
        print("\nNext step: option 3 (build index).")
    else:
        print("\nReady: option 4 (ask a question).")


def cmd_test(args, cfg: Config) -> None:
    import unittest

    tests = str(ROOT / "tests")
    suite = unittest.defaultTestLoader.discover(tests, top_level_dir=tests)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("some unit tests failed; see above")


def cmd_download(args, cfg: Config) -> None:
    from regintel.download import download

    download(cfg.raw_dir, only=args.only, force=args.force)


def cmd_index(args, cfg: Config) -> None:
    global _PIPELINE
    from regintel.index import build_all
    from regintel.ingest import build_chunks

    print("[index] first run downloads the embedding model; building can take a few minutes ...")
    build_all(cfg, build_chunks(cfg))
    _PIPELINE = None                              # force a reload of the fresh indexes


def cmd_ask(args, cfg: Config) -> None:
    ans = get_pipeline(cfg).ask(args.question, k=args.k, mode=args.mode)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("\n" + ans.pretty(show_context=args.show_context))


def cmd_chat(args, cfg: Config) -> None:
    pipe = get_pipeline(cfg)
    print("Ask about the indexed documents. Press Enter on an empty line to go back.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q:
            break
        print("\n" + pipe.ask(q, k=args.k, mode=args.mode).pretty())


def cmd_make_evalset(args, cfg: Config) -> None:
    from regintel.evalset import make_evalset
    from regintel.llm import get_llm

    make_evalset(cfg.chunks_path, get_llm(cfg), cfg.eval_dir / args.out, n=args.n, seed=args.seed)


def cmd_eval(args, cfg: Config) -> None:
    from regintel.evaluate import run_evaluation

    path = Path(args.evalset) if args.evalset else cfg.eval_dir / "eval_set.jsonl"
    run_evaluation(get_pipeline(cfg), path, sorted(set(args.ks)), args.modes, cfg.eval_dir / "results.json")


# =========================================================================== argparse (terminal use)
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("install", help="pip-install requirements.txt using this Python")
    s.set_defaults(fn=cmd_install)

    s = sub.add_parser("status", help="check what is set up and what to do next")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("test", help="run the model-free unit tests")
    s.set_defaults(fn=cmd_test)

    s = sub.add_parser("download", help="download the public document corpus")
    s.add_argument("--only", nargs="*", help="substring filter on file names, e.g. --only nist")
    s.add_argument("--force", action="store_true", help="re-download existing files")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("index", help="chunk documents and build BM25 + FAISS indexes")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("ask", help="answer one question with citations")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=None, help="passages given to the LLM (default: config)")
    s.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    s.add_argument("--show-context", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("chat", help="interactive question loop")
    s.add_argument("-k", type=int, default=None)
    s.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("make-evalset", help="LLM-generate a starter evaluation set")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="eval_set.jsonl", help="file name inside data/eval/")
    s.set_defaults(fn=cmd_make_evalset)

    s = sub.add_parser("eval", help="Precision@K / Recall@K across retrieval modes")
    s.add_argument("--evalset", default=None, help="path to JSONL (default: data/eval/eval_set.jsonl)")
    s.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    s.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    s.set_defaults(fn=cmd_eval)
    return p


def run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args, Config())
    return 0


# =========================================================================== interactive menu (IDLE)
MENU = """
================ AI Risk & Regulatory Intelligence Platform ================
  1) Install dependencies (pip)        5) Chat (ask several questions)
  2) Download documents                6) Generate evaluation set (uses the LLM)
  3) Build search index                7) Run evaluation (Precision@K / Recall@K)
  4) Ask one question                  8) Run unit tests
  s) Setup status      h) Command-line help      q) Quit
============================================================================="""


def _prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{text}{suffix}: ").strip() or default


def _yes(text: str, default: bool = False) -> bool:
    ans = input(f"{text} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not ans else ans.startswith("y")


def _menu_download() -> list[str]:
    return ["download"] + (["--force"] if _yes("Re-download files that already exist?") else [])


def _menu_ask() -> list[str] | None:
    q = _prompt("Your question").lstrip("-").strip()
    if not q:
        print("No question entered.")
        return None
    mode = _prompt(f"Retrieval mode ({' / '.join(MODES)})", "hybrid_rerank")
    if mode not in MODES:
        print("Unknown mode; using hybrid_rerank.")
        mode = "hybrid_rerank"
    argv = ["ask", q, "--mode", mode]
    if _yes("Also show the retrieved passages?"):
        argv.append("--show-context")
    return argv


def _menu_make_evalset() -> list[str]:
    n = _prompt("How many questions to generate", "60")
    return ["make-evalset", "--n", n if n.isdigit() else "60"]


def _menu_eval() -> list[str]:
    ks = _prompt("K values", "1 3 5 10").replace(",", " ").split()
    ks = ks if ks and all(k.isdigit() for k in ks) else ["1", "3", "5", "10"]
    default_path = "data/eval/eval_set.jsonl"
    path = _prompt("Evaluation file", default_path)
    return ["eval", "--ks", *ks] + (["--evalset", path] if path != default_path else [])


def _hint(e: Exception) -> None:
    if isinstance(e, ModuleNotFoundError):
        print("Hint: a required package is missing. Choose option 1 (install dependencies), then retry.")
    elif "ANTHROPIC_API_KEY" in str(e):
        print(f"Hint: the LLM provider is currently '{Config().llm_provider}', which needs ANTHROPIC_API_KEY.")
        print("      What the program found when loading your settings:")
        for line in _env_report():
            print(f"        - {line}")
        print("      To use a LOCAL model instead, the .env file (next to main.py, named exactly .env) needs:")
        print("        LLM_PROVIDER=hf")
        print("        HF_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct")
        print("      After editing .env, run this file again with F5.")
    elif isinstance(e, FileNotFoundError):
        print("Hint: do option 2 (download documents) and option 3 (build index) first. Option s shows what is missing.")


def _run_safely(argv: list[str]) -> None:
    """Run one command; report errors in the shell instead of closing the session."""
    try:
        run(argv)
    except KeyboardInterrupt:
        print("\n[cancelled]")
    except SystemExit:
        print("[invalid arguments]")
    except Exception as e:  # noqa: BLE001
        expected = isinstance(e, (FileNotFoundError, ModuleNotFoundError)) or "ANTHROPIC_API_KEY" in str(e)
        if not expected:
            traceback.print_exc()          # full traceback only for unexpected errors
        print(f"\nError: {e}")
        _hint(e)


def menu() -> int:
    actions = {
        "1": lambda: ["install"],
        "2": _menu_download,
        "3": lambda: ["index"],
        "4": _menu_ask,
        "5": lambda: ["chat"],
        "6": _menu_make_evalset,
        "7": _menu_eval,
        "8": lambda: ["test"],
        "s": lambda: ["status"],
    }
    while True:
        print(MENU)
        try:
            choice = input("Choose an option: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in {"q", "quit", "exit", "0"}:
            print("Bye.")
            return 0
        if choice in {"h", "help", "?"}:
            build_parser().print_help()
            continue
        maker = actions.get(choice)
        if maker is None:
            print("Please choose one of the listed options.")
            continue
        try:
            argv = maker()
        except (EOFError, KeyboardInterrupt):
            print("\n[cancelled]")
            continue
        if argv:
            _run_safely(argv)


# =========================================================================== entry
def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:                       # terminal usage: python main.py <command> ...
        return run(argv)
    if IDLE_ARGS:                  # IDLE with a pre-set command
        _run_safely(list(IDLE_ARGS))
        return 0
    return menu()                  # IDLE / double-click: interactive menu


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
