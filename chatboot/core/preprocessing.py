"""
preprocessing.py — Text & document preprocessing pipeline
==========================================================
Runs BEFORE indexing. Handles:
  • Loading JSON / CSV / TXT / MD documents from disk
  • Unicode normalization, lowercasing, whitespace cleanup
  • Optional stopword removal & spaCy lemmatization
  • Recursive chunking with overlap
  • Deduplication by content hash

ML/NLP families: NLP preprocessing.
"""

from __future__ import annotations
import json
import csv
import re
import hashlib
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable

try:
    import spacy

    _NLP = None

    def _get_nlp():
        global _NLP
        if _NLP is None:
            try:
                _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except OSError:
                # Model not installed — see README
                _NLP = spacy.blank("en")
        return _NLP

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

    def _get_nlp():
        return None


# Simple English stopword list (no NLTK dependency)
STOPWORDS = set("""
a an the and or but if then else when while is am are was were be been being have has had do does
did of in on at to from for with about as by into over under up down off out so than that this these
those it its he she they them his her their we us our you your i my me not no nor very can will
just should would could may might must shall
""".split())


@dataclass
class Chunk:
    """A single text chunk that gets indexed into the vector store."""

    chunk_id: str
    text: str
    raw_text: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "raw_text": self.raw_text,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    strip_punctuation: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = True
    dedupe: bool = True
    chunk_size: int = 256
    chunk_overlap: int = 32
    min_chunk_words: int = 8


def _flatten_record(obj, depth: int = 0) -> str:
    """Flatten any nested JSON-ish record into a single descriptive text blob."""
    if depth > 5:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if obj is None:
        return ""
    if isinstance(obj, list):
        return " ".join(_flatten_record(v, depth + 1) for v in obj)
    if isinstance(obj, dict):
        return " | ".join(
            f"{k}: {_flatten_record(v, depth + 1)}" for k, v in obj.items()
        )
    return ""


def load_documents(path: str | Path) -> list[dict]:
    """
    Load a single file or every file in a directory. Returns a list of
    dicts with shape {source, text, metadata}.
    """
    p = Path(path)
    if p.is_dir():
        results = []
        for f in p.iterdir():
            if f.suffix.lower() in (".json", ".csv", ".txt", ".md"):
                results.extend(load_documents(f))
        return results

    ext = p.suffix.lower()
    name = p.stem
    docs: list[dict] = []

    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]
        for i, rec in enumerate(records):
            docs.append(
                {
                    "source": p.name,
                    "text": _flatten_record(rec),
                    "metadata": {
                        "record_idx": i,
                        "raw": rec if isinstance(rec, dict) else {"value": rec},
                    },
                }
            )
    elif ext == ".csv":
        with p.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                docs.append(
                    {
                        "source": p.name,
                        "text": " | ".join(f"{k}: {v}" for k, v in row.items() if v),
                        "metadata": {"record_idx": i, "raw": row},
                    }
                )
    elif ext in (".txt", ".md"):
        text = p.read_text(encoding="utf-8")
        # split on blank lines into paragraphs
        for i, para in enumerate(re.split(r"\n{2,}", text)):
            para = para.strip()
            if len(para) > 20:
                docs.append(
                    {"source": p.name, "text": para, "metadata": {"record_idx": i}}
                )

    return docs


_PUNCT_RE = re.compile(r"[^\w\s\.\-/]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str, cfg: PreprocessConfig) -> str:
    """Apply the configured cleaning steps."""
    # Unicode NFKC normalize
    text = unicodedata.normalize("NFKC", text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str, cfg: PreprocessConfig) -> list[str]:
    """Tokenize + (optionally) lemmatize + remove stopwords."""
    if cfg.lemmatize and HAS_SPACY:
        nlp = _get_nlp()
        toks = [
            t.lemma_.lower() for t in nlp(text) if not t.is_space and t.lemma_.strip()
        ]
    else:
        toks = text.split()
    if cfg.remove_stopwords:
        toks = [t for t in toks if t.lower() not in STOPWORDS]
    return toks


def normalize_for_index(text: str, cfg: PreprocessConfig) -> str:
    """Cleaned + tokenized text reassembled as a single string for embedding."""
    cleaned = clean_text(text, cfg)
    toks = tokenize(cleaned, cfg)
    return " ".join(toks)


def chunk_words(
    text: str, chunk_size: int = 256, overlap: int = 32, min_words: int = 8
) -> list[str]:
    """
    Split text into overlapping word-token chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) >= min_words else []
    out = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        seg = words[i : i + chunk_size]
        if len(seg) >= min_words:
            out.append(" ".join(seg))
        i += step
    return out


def preprocess(
    docs: list[dict], cfg: PreprocessConfig | None = None, progress=None
) -> list[Chunk]:
    """
    Run the full preprocessing pipeline over a list of documents.

    Args:
        docs: list of {source, text, metadata}
        cfg:  PreprocessConfig, or default if None
        progress: optional callback (stage: str, pct: float, msg: str)

    Returns:
        list[Chunk] ready for embedding & indexing.
    """
    cfg = cfg or PreprocessConfig()
    out: list[Chunk] = []
    seen_hashes: set[str] = set()
    total = max(1, len(docs))

    for i, d in enumerate(docs):
        if progress:
            progress("chunk", i / total, f"{d['source']}")
        raw = d["text"]
        if not raw:
            continue

        # Step 1: clean raw text → indexable form
        normalized = normalize_for_index(raw, cfg)

        # Step 2: chunk into pieces
        # Chunking on the cleaned text for retrieval, raw text for display
        sub_chunks_clean = chunk_words(
            normalized, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_words
        )
        sub_chunks_raw = chunk_words(
            raw, cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_words
        )
        # Realign lengths (cleaning can change word counts; pad raw with last)
        n = len(sub_chunks_clean)
        while len(sub_chunks_raw) < n:
            sub_chunks_raw.append(sub_chunks_clean[len(sub_chunks_raw)])
        sub_chunks_raw = sub_chunks_raw[:n]

        for j, (clean, raw_seg) in enumerate(zip(sub_chunks_clean, sub_chunks_raw)):
            # Step 3: dedupe
            if cfg.dedupe:
                h = hashlib.md5(clean.encode()).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
            cid = f"{d['source']}::{d['metadata'].get('record_idx', 0)}::{j}"
            out.append(
                Chunk(
                    chunk_id=cid,
                    text=clean,
                    raw_text=raw_seg,
                    source=d["source"],
                    metadata=d.get("metadata", {}),
                )
            )
    if progress:
        progress("done", 1.0, f"{len(out)} chunks")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path-to-file-or-dir>")
        sys.exit(0)
    docs = load_documents(sys.argv[1])
    print(f"Loaded {len(docs)} documents")
    cfg = PreprocessConfig()
    chunks = preprocess(
        docs, cfg, progress=lambda s, p, m: print(f"[{s}] {p:5.1%} {m}")
    )
    print(f"Produced {len(chunks)} chunks")
    if chunks:
        print("\nFirst chunk:")
        print(f"  id:   {chunks[0].chunk_id}")
        print(f"  text: {chunks[0].text[:120]}…")
