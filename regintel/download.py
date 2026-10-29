"""Download the regulatory / risk-framework corpus into data/raw/.

All documents are public. Some publishers (notably EUR-Lex) block scripted downloads;
for any file that fails, open the URL in a browser, save the PDF into data/raw/ under the
listed file name, and re-run `python main.py index`.

Always check the publisher's site for newer / consolidated / amended versions before you
treat any answer as current: regulations get corrected and amended.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config

# Theme: AI governance + model/ICT/data risk. Mix of frameworks (NIST), law (EU), and supervisory guidance (Fed, BCBS).
DOCS: list[dict] = [
    {
        "file": "nist_ai_rmf_1_0.pdf",
        "title": "NIST AI Risk Management Framework 1.0 (NIST AI 100-1)",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
    },
    {
        "file": "nist_ai_600_1_genai_profile.pdf",
        "title": "NIST AI 600-1: Generative AI Profile",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
    },
    {
        "file": "nist_csf_2_0.pdf",
        "title": "NIST Cybersecurity Framework 2.0 (CSWP 29)",
        "url": "https://nvlpubs.nist.gov/nistpubs/CSWP/NIST.CSWP.29.pdf",
    },
    {
        "file": "eu_ai_act_2024_1689.pdf",
        "title": "EU AI Act, Regulation (EU) 2024/1689",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
    },
    {
        "file": "eu_gdpr_2016_679.pdf",
        "title": "GDPR, Regulation (EU) 2016/679",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
    },
    {
        "file": "eu_dora_2022_2554.pdf",
        "title": "DORA, Regulation (EU) 2022/2554 (digital operational resilience)",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554",
    },
    {
        "file": "fed_sr_11_7_model_risk.pdf",
        "title": "Federal Reserve SR 11-7: Supervisory Guidance on Model Risk Management",
        "url": "https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf",
    },
    {
        "file": "bcbs_239_risk_data_aggregation.pdf",
        "title": "BCBS 239: Principles for effective risk data aggregation and risk reporting",
        "url": "https://www.bis.org/publ/bcbs239.pdf",
    },
]


def download(raw_dir: Path | None = None, only: list[str] | None = None, force: bool = False) -> None:
    import requests

    cfg = Config()
    raw_dir = raw_dir or cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; regintel-research/0.1)"}

    meta_path = raw_dir / "sources.json"
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    failed: list[tuple[dict, str]] = []

    for d in DOCS:
        if only and not any(o in d["file"] for o in only):
            continue
        dest = raw_dir / d["file"]
        if dest.exists() and not force:
            print(f"[skip] {d['file']} already present")
            meta[d["file"]] = {"title": d["title"], "url": d["url"]}
            continue
        try:
            print(f"[get ] {d['file']}")
            r = requests.get(d["url"], headers=headers, timeout=90)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("response was not a PDF (the site probably requires a real browser)")
            dest.write_bytes(r.content)
            meta[d["file"]] = {"title": d["title"], "url": d["url"]}
            print(f"       saved {len(r.content) / 1e6:.1f} MB")
        except Exception as e:  # noqa: BLE001
            failed.append((d, str(e)))

    # sources.json powers human-readable titles + URLs in citations
    meta_path.write_text(json.dumps(meta, indent=2))

    if failed:
        print("\nCould not download these automatically:")
        for d, err in failed:
            print(f"  - {d['file']}: {err}\n    open {d['url']} in a browser, save as data/raw/{d['file']}")
        print("Then run: python main.py index")
    else:
        print("\nAll documents ready. Next: python main.py index")
