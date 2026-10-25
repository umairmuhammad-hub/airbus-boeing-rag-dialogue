"""Step 1 - Turn the Airbus/Boeing PDFs into clean text files (data/txt/).

All three systems (RAG, GraphRAG, LightRAG) read exactly these text files,
so the comparison between them is fair.
"""
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
PDF_DIR, TXT_DIR = ROOT / "data" / "pdf", ROOT / "data" / "txt"
TXT_DIR.mkdir(parents=True, exist_ok=True)


def page_to_text(page):
    """Text of one page, with tables kept as Markdown tables (row/column structure intact)."""
    try:
        tables = page.find_tables().tables
    except Exception:
        tables = []
    boxes = [pymupdf.Rect(t.bbox) for t in tables]
    parts = []  # (y position, text)
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        r = pymupdf.Rect(x0, y0, x1, y1)
        if not any(r.intersects(b) for b in boxes) and text.strip():
            parts.append((y0, text.strip()))
    for t, b in zip(tables, boxes):
        md = t.to_markdown().replace("<br>", " ").strip()
        if md:
            parts.append((b.y0, md))
    return "\n".join(text for _, text in sorted(parts, key=lambda p: p[0]))


def clean(pages):
    """Remove headers/footers, page numbers and broken hyphenation."""
    # A line that repeats on many pages is a running header or footer.
    counts = Counter(l.strip() for p in pages for l in set(p.splitlines()) if l.strip())
    limit = max(3, len(pages) // 3)
    kept = []
    for p in pages:
        lines = [l for l in p.splitlines()
                 if l.strip()
                 and (l.lstrip().startswith("|") or counts[l.strip()] < limit)
                 and not re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", l.strip(), re.I)]
        kept.append("\n".join(lines))
    text = "\n\n".join(kept)
    text = unicodedata.normalize("NFKC", text)        # ligatures such as "ﬂ" -> "fl"
    text = re.sub(r"(fi|fl) (?=[a-z])", r"\1", text)  # "fi rst" -> "first" (broken ligatures)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join "air-\nplane"
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))["documents"]
    rows = []
    for doc in corpus:
        pdf = PDF_DIR / doc["file"]
        if not pdf.exists():
            print(f"MISSING: {pdf.name} - skipped")
            continue
        d = pymupdf.open(pdf)
        first, last = doc["pages"] or [1, len(d)]
        last = min(last, len(d))
        pages = [page_to_text(d[i]) for i in range(first - 1, last)]
        text = clean(pages)
        header = f"Document: {doc['title']}\nManufacturer: {doc['maker']}\nSource id: {doc['id']}\n\n"
        (TXT_DIR / f"{doc['id']}.txt").write_text(header + text, encoding="utf-8")
        rows.append({"id": doc["id"], "maker": doc["maker"], "title": doc["title"],
                     "pdf_pages": len(d), "pages_used": last - first + 1,
                     "words": len(text.split()), "characters": len(text)})
        print(f"{pdf.name}: pages {first}-{last} -> {len(text.split()):,} words")

    with open(ROOT / "metrics" / "corpus_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} documents, {sum(r['words'] for r in rows):,} words in total -> data/txt/")


if __name__ == "__main__":
    main()
