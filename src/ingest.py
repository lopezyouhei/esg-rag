from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2400  # char
CHUNK_OVERLAP = 300  # char


def iter_pdfs(raw_dir: Path) -> list[Path]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {raw_dir}/, add a report there first.")
    return pdfs


def extract_pages(pdf_path: Path):
    """Yields (page number, text) for non-empty pages. Pages are 1-indexed.

    Args:
        pdf_path (Path): Path to pdf
    """
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                yield i, text

    finally:
        doc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw", help="directory of source PDFs")
    ap.add_argument("--out", default="data/chunks.jsonl", help="output JSONL path")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    n_docs = n_pages = n_chunks = total_chars = 0
    with out_path.open("w", encoding="utf-8") as f:
        for pdf_path in iter_pdfs(raw_dir):
            n_docs += 1
            source = pdf_path.name
            stem = pdf_path.stem
            doc_chunks = 0
            for page_number, page_text in extract_pages(pdf_path):
                n_pages += 1
                for i, chunk in enumerate(splitter.split_text(page_text)):
                    record = {
                        "chunk_id": f"{stem}_p{page_number}_c{i}",
                        "source_file": source,
                        "page_number": page_number,
                        "text": chunk,
                        "char_count": len(chunk),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_chunks += 1
                    doc_chunks += 1
                    total_chars += len(chunk)

            print(f"    {source}: {doc_chunks} chunks")

    avg = round(total_chars / n_chunks) if n_chunks else 0
    print(
        f"\n{n_docs} PDF(s), {n_pages} pages -> {n_chunks} chunks "
        f"(avg {avg} chars/chunk)"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
