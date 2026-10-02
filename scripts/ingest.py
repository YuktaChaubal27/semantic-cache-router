"""Chunk and index every .md/.txt file under a folder, then bump corpus_version.

Bumping the version instantly invalidates every cached answer built on the
old corpus, because cache lookups filter on the current version.

    python scripts/ingest.py data/docs
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
from app import db
from app.rag import index_chunks


def chunk(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    paras, out, buf = [p.strip() for p in text.split("\n\n") if p.strip()], [], ""
    for p in paras:
        if len(buf) + len(p) > size and buf:
            out.append(buf)
            buf = buf[-overlap:]
        buf = f"{buf}\n\n{p}".strip()
    if buf:
        out.append(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()

    chunks = []
    for f in sorted(Path(args.folder).rglob("*")):
        if f.suffix.lower() in {".md", ".txt"}:
            for i, c in enumerate(chunk(f.read_text(encoding="utf-8"))):
                chunks.append({"id": f"{f}:{i}", "text": c, "source": f.name})
    index_chunks(chunks)
    v = db.bump_corpus_version()
    print(f"Indexed {len(chunks)} chunks. corpus_version is now {v}; old cache entries are invalid.")


if __name__ == "__main__":
    main()
