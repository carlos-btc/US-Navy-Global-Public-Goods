"""
01_index_readings.py
- Scans ./Readings for PDFs
- Extracts minimal metadata (best-effort) and writes Bibliography/readings_auto.bib
- Produces a CSV index for auditability

"""

from pathlib import Path
from pypdf import PdfReader
import re
import csv

ROOT = Path(__file__).resolve().parents[1]
READINGS = ROOT / "Readings"
OUT_BIB = ROOT / "Bibliography" / "readings_auto.bib"
OUT_CSV = ROOT / "Bibliography" / "readings_index.csv"

def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:60] if s else "unknown"

def guess_year(text: str) -> str | None:
    m = re.search(r"(19|20)\d{2}", text)
    return m.group(0) if m else None

def main():
    pdfs = sorted(READINGS.glob("*.pdf"))
    rows = []
    bib_entries = ["% Auto-generated from ./Readings\n"]
    for pdf in pdfs:
        reader = PdfReader(str(pdf))
        meta = reader.metadata or {}
        title = (meta.title or pdf.stem).strip()
        author = (meta.author or "").strip()
        # best-effort year guess from first page
        first = ""
        try:
            first = (reader.pages[0].extract_text() or "")[:2000]
        except Exception:
            pass
        year = guess_year(first) or ""
        key = f"{slugify(author.split(',')[0] or pdf.stem)}{year or ''}"
        key = key or slugify(pdf.stem)

        rows.append({
            "file": pdf.name,
            "key": key,
            "title": title,
            "author": author,
            "year": year
        })

        bib_entries.append(f"@misc{{{key},\n"
                           f"  title  = {{{title}}},\n"
                           f"  author = {{{author}}},\n"
                           f"  year   = {{{year}}},\n"
                           f"  note   = {{From ./Readings/{pdf.name}}}\n"
                           f"}}\n\n")

    OUT_BIB.write_text("".join(bib_entries), encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","key","title","author","year"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_BIB}")
    print(f"Wrote {OUT_CSV}")
    if not pdfs:
        print("No PDFs found in ./Readings. Add them and rerun.")

if __name__ == "__main__":
    main()
