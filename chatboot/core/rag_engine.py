"""
rag_engine.py — Retrieval-Augmented Generation engine
======================================================
Capabilities:
  • Dense embeddings via sentence-transformers (DL)
  • FAISS HNSW vector index (ML)
  • BM25 sparse retrieval (ML)
  • Hybrid retrieval with Reciprocal Rank Fusion (ML)
  • Cross-encoder reranker (DL)
  • Persistent index per dataset

ML/DL families: DL embedding + reranker, ML index + fusion.
"""

from __future__ import annotations
import json
import pickle
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .preprocessing import Chunk

logger = logging.getLogger(__name__)
FALLBACK_EMBED_DIM = 384
FALLBACK_EMBED_MODEL = f"hashing-fallback-{FALLBACK_EMBED_DIM}"

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder

    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class RAGConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    use_bm25: bool = True
    top_k_dense: int = 8
    top_k_bm25: int = 8
    top_k_final: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    hnsw_M: int = 32
    hnsw_ef_construction: int = 200


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float  # combined relevance score
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "score": float(self.score),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rerank_score": self.rerank_score,
        }


class RAGEngine:
    """One engine instance can hold multiple named indexes (one per dataset)."""

    def __init__(
        self, cfg: RAGConfig | None = None, persist_dir: str | Path = "indexes"
    ):
        self.cfg = cfg or RAGConfig()
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._encoder = None  # lazy-loaded SentenceTransformer
        self._encoder_model_name = None
        self._reranker = None  # lazy-loaded CrossEncoder
        self._reranker_model_name = None
        self.indexes: dict[str, dict] = {}  # name → {chunks, faiss, bm25}

    @property
    def encoder(self):
        if not HAS_ST:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install: pip install sentence-transformers"
            )
        if self._encoder is None or self._encoder_model_name != self.cfg.embed_model:
            logger.info("Loading embedding model: %s", self.cfg.embed_model)
            self._encoder = SentenceTransformer(self.cfg.embed_model)
            self._encoder_model_name = self.cfg.embed_model
        return self._encoder

    @staticmethod
    def _hashing_embeddings(texts: list[str]) -> np.ndarray:
        """
        Lightweight local embedding fallback for environments where
        sentence-transformers has not been installed yet.
        """
        embs = np.zeros((len(texts), FALLBACK_EMBED_DIM), dtype="float32")
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % FALLBACK_EMBED_DIM
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                embs[row, bucket] += sign
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def _encode_texts(
        self, texts: list[str], model_name: str | None = None
    ) -> np.ndarray:
        model_name = model_name or self.cfg.embed_model
        if not HAS_ST or model_name == FALLBACK_EMBED_MODEL:
            return self._hashing_embeddings(texts)
        old_model = self.cfg.embed_model
        try:
            self.cfg.embed_model = model_name
            return self.encoder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        finally:
            self.cfg.embed_model = old_model

    @property
    def reranker(self):
        if not (HAS_ST and self.cfg.use_reranker):
            return None
        if (
            self._reranker is None
            or self._reranker_model_name != self.cfg.reranker_model
        ):
            logger.info("Loading reranker: %s", self.cfg.reranker_model)
            self._reranker = CrossEncoder(self.cfg.reranker_model)
            self._reranker_model_name = self.cfg.reranker_model
        return self._reranker

    @property
    def embed_dim(self) -> int:
        if not HAS_ST:
            return FALLBACK_EMBED_DIM
        return self.encoder.get_sentence_embedding_dimension()

    def build_index(self, name: str, chunks: list[Chunk], progress=None) -> dict:
        """Embed + index a list of chunks under the given name. Persists to disk."""
        if not chunks:
            raise ValueError("No chunks to index")
        if progress:
            progress("embed", 0.0, f"Embedding {len(chunks)} chunks…")

        texts = [c.text for c in chunks]
        # Embed in batches with progress
        batch = 32
        embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch):
            seg = self._encode_texts(texts[i : i + batch])
            embs.append(seg)
            if progress:
                progress(
                    "embed",
                    min(0.95, (i + batch) / len(texts)),
                    f"{min(i + batch, len(texts))}/{len(texts)}",
                )
        embeddings = np.vstack(embs).astype("float32")

        # Build FAISS index (HNSW for fast approximate search)
        if progress:
            progress("index", 0.95, "Building FAISS index")
        if HAS_FAISS:
            index = faiss.IndexHNSWFlat(embeddings.shape[1], self.cfg.hnsw_M)
            index.hnsw.efConstruction = self.cfg.hnsw_ef_construction
            index.add(embeddings)
        else:
            # Fallback: pure-numpy cosine similarity index
            index = {"embeddings": embeddings, "type": "numpy_fallback"}

        # Build BM25 sparse index
        if progress:
            progress("bm25", 0.97, "Building BM25 index")
        bm25 = None
        if HAS_BM25 and self.cfg.use_bm25:
            tokenized_corpus = [t.split() for t in texts]
            bm25 = BM25Okapi(tokenized_corpus)

        idx = {
            "chunks": chunks,
            "faiss": index,
            "embeddings": embeddings,
            "bm25": bm25,
            "embed_model": self.cfg.embed_model if HAS_ST else FALLBACK_EMBED_MODEL,
        }
        self.indexes[name] = idx
        self._persist(name, idx)
        if progress:
            progress("done", 1.0, f"{len(chunks)} chunks indexed")
        return {
            "name": name,
            "chunks": len(chunks),
            "dim": int(embeddings.shape[1]),
            "embed_model": idx["embed_model"],
            "has_bm25": bm25 is not None,
        }

    def _persist(self, name: str, idx: dict):
        d = self.persist_dir / name
        d.mkdir(exist_ok=True)
        # Chunks
        with (d / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in idx["chunks"]], f, ensure_ascii=False)
        # Embeddings
        np.save(d / "embeddings.npy", idx["embeddings"])
        # FAISS
        if HAS_FAISS and not isinstance(idx["faiss"], dict):
            faiss.write_index(idx["faiss"], str(d / "faiss.index"))
        # BM25
        if idx["bm25"] is not None:
            with (d / "bm25.pkl").open("wb") as f:
                pickle.dump(idx["bm25"], f)
        # Meta
        meta = {
            "embed_model": idx["embed_model"],
            "n_chunks": len(idx["chunks"]),
            "dim": int(idx["embeddings"].shape[1]),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    def load_index(self, name: str) -> Optional[dict]:
        """Load a previously-built index from disk."""
        d = self.persist_dir / name
        if not d.exists() or not (d / "chunks.json").exists():
            return None
        # Chunks
        raw_chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in raw_chunks]
        # Embeddings
        embeddings = np.load(d / "embeddings.npy")
        # FAISS
        faiss_idx = None
        if HAS_FAISS and (d / "faiss.index").exists():
            faiss_idx = faiss.read_index(str(d / "faiss.index"))
        else:
            faiss_idx = {"embeddings": embeddings, "type": "numpy_fallback"}
        # BM25
        bm25 = None
        if (d / "bm25.pkl").exists():
            with (d / "bm25.pkl").open("rb") as f:
                bm25 = pickle.load(f)
        meta = json.loads((d / "meta.json").read_text())
        idx = {
            "chunks": chunks,
            "faiss": faiss_idx,
            "embeddings": embeddings,
            "bm25": bm25,
            "embed_model": meta["embed_model"],
        }
        self.indexes[name] = idx
        return idx

    def list_indexes(self) -> list[dict]:
        out = []
        for d in self.persist_dir.iterdir():
            if d.is_dir() and (d / "meta.json").exists():
                meta = json.loads((d / "meta.json").read_text())
                out.append({"name": d.name, **meta})
        return out

    def delete_index(self, name: str):
        d = self.persist_dir / name
        if d.exists():
            import shutil

            shutil.rmtree(d)
        self.indexes.pop(name, None)

    def _dense_search(
        self, query_emb: np.ndarray, idx: dict, k: int
    ) -> list[tuple[int, float]]:
        """Returns list of (chunk_idx, similarity)."""
        if HAS_FAISS and not isinstance(idx["faiss"], dict):
            D, I = idx["faiss"].search(query_emb.reshape(1, -1).astype("float32"), k)
            # HNSW returns L2 distances — convert to cosine sim for normalized vectors
            # ||a - b||^2 = 2(1 - cos(a,b)) for unit vectors
            sims = 1 - D[0] / 2
            return [(int(i), float(s)) for i, s in zip(I[0], sims) if i != -1]
        # numpy fallback
        embs = idx["embeddings"]
        sims = embs @ query_emb
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def _bm25_search(
        self, query_tokens: list[str], idx: dict, k: int
    ) -> list[tuple[int, float]]:
        if idx["bm25"] is None:
            return []
        scores = idx["bm25"].get_scores(query_tokens)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def _rrf_fuse(
        rankings: list[list[tuple[int, float]]], k: int = 60
    ) -> dict[int, float]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, (idx, _) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def retrieve(
        self, query: str, name: str, top_k: int | None = None
    ) -> tuple[list[RetrievalResult], dict]:
        """
        Run the full retrieval pipeline: dense + sparse → RRF fusion → rerank.

        Returns (results, timings) — timings is a dict of per-stage ms.
        """
        import time

        idx = self.indexes.get(name) or self.load_index(name)
        if idx is None:
            raise KeyError(f"Index '{name}' not found")
        top_k = top_k or self.cfg.top_k_final
        timings = {}

        # 1. Dense
        t = time.perf_counter()
        q_emb = self._encode_texts([query], idx.get("embed_model"))[0].astype("float32")
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense_hits = self._dense_search(q_emb, idx, self.cfg.top_k_dense)
        timings["dense_ms"] = (time.perf_counter() - t) * 1000

        # 2. BM25
        t = time.perf_counter()
        sparse_hits = self._bm25_search(query.lower().split(), idx, self.cfg.top_k_bm25)
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        # 3. RRF
        t = time.perf_counter()
        fused = self._rrf_fuse([dense_hits, sparse_hits], self.cfg.rrf_k)
        ranked = sorted(fused.items(), key=lambda x: -x[1])
        timings["fusion_ms"] = (time.perf_counter() - t) * 1000

        # Build candidate results
        dense_ranks = {i: r for r, (i, _) in enumerate(dense_hits)}
        sparse_ranks = {i: r for r, (i, _) in enumerate(sparse_hits)}
        candidates = []
        for cidx, score in ranked[: max(top_k * 2, 10)]:
            chunk = idx["chunks"][cidx]
            candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    dense_rank=dense_ranks.get(cidx),
                    sparse_rank=sparse_ranks.get(cidx),
                )
            )

        # 4. Rerank (optional)
        if self.reranker is not None and candidates:
            t = time.perf_counter()
            pairs = [(query, c.chunk.text) for c in candidates]
            scores = self.reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c.rerank_score = float(s)
            candidates.sort(key=lambda c: -c.rerank_score)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        timings["total_ms"] = sum(timings.values())
        return candidates[:top_k], timings


if __name__ == "__main__":
    import sys
    from .preprocessing import load_documents, preprocess, PreprocessConfig

    if len(sys.argv) < 3:
        print("Usage: python rag_engine.py <data-dir> <query>")
        sys.exit(0)
    docs = load_documents(sys.argv[1])
    chunks = preprocess(docs, PreprocessConfig())
    eng = RAGEngine()
    eng.build_index(
        "demo", chunks, progress=lambda s, p, m: print(f"[{s}] {p:5.1%} {m}")
    )
    results, timings = eng.retrieve(sys.argv[2], "demo")
    print(f"\nTimings (ms): {timings}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.3f}] {r.chunk.source}")
        print(f"   {r.chunk.raw_text[:160]}…\n")
