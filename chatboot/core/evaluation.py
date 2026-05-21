"""
evaluation.py — Evaluation metrics for the RAG pipeline
========================================================
Computes:
  • Retrieval:    Precision@k, Recall@k, MRR, nDCG@k
  • Generation:   ROUGE-L, BLEU-4
  • Semantic:     BERTScore (or sentence-transformer cosine if unavailable)
  • Faithfulness: LLM-as-judge OR lexical-overlap proxy
  • Latency:      total / per-stage milliseconds

Gold set shape (list of dicts):
  {
    "query": "What is COVID-19?",
    "expected_chunk_ids": ["med.json::1::0", ...],   # optional
    "expected_answer":    "COVID-19 is a respiratory…",
    "dataset": "med",
  }
"""

from __future__ import annotations
import math
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import Optional, Callable

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from rouge_score import rouge_scorer

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score_fn

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not retrieved or not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return sum(
        (1.0 if r in relevant else 0.0) / math.log2(i + 2)
        for i, r in enumerate(retrieved[:k])
    )


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\b\w+\b", text.lower())


def bleu_n(prediction: str, reference: str, n: int = 4) -> float:
    """Lightweight BLEU implementation (no NLTK). Includes brevity penalty."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0

    weights = [1 / n] * n
    precisions = []
    for k in range(1, n + 1):
        pred_grams = Counter(tuple(pred[i : i + k]) for i in range(len(pred) - k + 1))
        ref_grams = Counter(tuple(ref[i : i + k]) for i in range(len(ref) - k + 1))
        overlap = sum((pred_grams & ref_grams).values())
        total = max(1, sum(pred_grams.values()))
        precisions.append(overlap / total)
    # Smooth zeros
    precisions = [p if p > 0 else 1e-9 for p in precisions]
    log_p = sum(w * math.log(p) for w, p in zip(weights, precisions))
    bp = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / max(1, len(pred)))
    return bp * math.exp(log_p)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score. Uses rouge_score if available, else LCS impl."""
    if HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, prediction)["rougeL"].fmeasure
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    # LCS via DP
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (
                (dp[i][j] + 1) if pred[i] == ref[j] else max(dp[i][j + 1], dp[i + 1][j])
            )
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


_SEM_MODEL = None


def _get_sem_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    global _SEM_MODEL
    if _SEM_MODEL is None and HAS_ST:
        _SEM_MODEL = SentenceTransformer(name)
    return _SEM_MODEL


def semantic_similarity(prediction: str, reference: str) -> float:
    """Cosine similarity of sentence embeddings."""
    m = _get_sem_model()
    if m is None:
        return 0.0
    embs = m.encode([prediction, reference], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def bertscore_f1(predictions: list[str], references: list[str]) -> float:
    """Average BERTScore F1 (requires bert-score)."""
    if not HAS_BERTSCORE:
        return 0.0
    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    return float(f1.mean())


def lexical_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Fraction of answer tokens that appear in the retrieved contexts.
    A simple proxy for "is the answer grounded in retrieved evidence?".
    """
    ans = set(_tokens(answer))
    if not ans:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx |= set(_tokens(c))
    return len(ans & ctx) / len(ans)


def semantic_faithfulness(answer: str, contexts: list[str]) -> float:
    """Max semantic similarity between answer and any retrieved context."""
    if not contexts:
        return 0.0
    m = _get_sem_model()
    if m is None:
        return lexical_faithfulness(answer, contexts)
    embs = m.encode([answer] + contexts, normalize_embeddings=True)
    ans_emb = embs[0]
    ctx_embs = embs[1:]
    return float(max(np.dot(ans_emb, c) for c in ctx_embs))


@dataclass
class QueryResult:
    query: str
    answer: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    expected_answer: str
    contexts: list[str]
    latency_ms: float = 0.0
    # Per-query metrics
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    rouge_l: float = 0.0
    bleu_4: float = 0.0
    sem_sim: float = 0.0
    faithfulness: float = 0.0


@dataclass
class EvalReport:
    run_id: str
    n_queries: int
    per_query: list[QueryResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "n_queries": self.n_queries,
            "aggregate": self.aggregate,
            "per_query": [asdict(q) for q in self.per_query],
        }

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def run_evaluation(
    gold_set: list[dict],
    answer_fn: Callable[[str, str], tuple[str, list[str], list[str], float]],
    run_id: str = "eval",
    k: int = 5,
    progress: Callable | None = None,
) -> EvalReport:
    """
    Run the full eval suite.

    Args:
        gold_set:    list of {query, expected_chunk_ids, expected_answer, dataset}
        answer_fn:   given (query, dataset) returns
                       (answer_text, retrieved_chunk_ids, retrieved_contexts, latency_ms)
        run_id:      label for this eval run
        k:           top-k for retrieval metrics
    """
    per_query: list[QueryResult] = []
    n = len(gold_set)
    for i, item in enumerate(gold_set):
        if progress:
            progress(i / max(1, n), item["query"])
        ans, retr_ids, contexts, lat_ms = answer_fn(
            item["query"], item.get("dataset", "")
        )
        exp_ids = item.get("expected_chunk_ids", []) or []
        exp_ans = item.get("expected_answer", "") or ""

        q = QueryResult(
            query=item["query"],
            answer=ans,
            retrieved_ids=retr_ids,
            expected_ids=exp_ids,
            expected_answer=exp_ans,
            contexts=contexts,
            latency_ms=lat_ms,
        )
        # Retrieval
        if exp_ids:
            q.precision = precision_at_k(retr_ids, exp_ids, k)
            q.recall = recall_at_k(retr_ids, exp_ids, k)
            q.mrr = reciprocal_rank(retr_ids, exp_ids)
            q.ndcg = ndcg_at_k(retr_ids, exp_ids, k)
        # Generation
        if exp_ans:
            q.rouge_l = rouge_l(ans, exp_ans)
            q.bleu_4 = bleu_n(ans, exp_ans, 4)
            q.sem_sim = semantic_similarity(ans, exp_ans)
        q.faithfulness = semantic_faithfulness(ans, contexts)
        per_query.append(q)

    # Aggregate
    if per_query:

        def avg(field):
            return sum(getattr(q, field) for q in per_query) / len(per_query)

        agg = {
            f"precision@{k}": avg("precision"),
            f"recall@{k}": avg("recall"),
            "mrr": avg("mrr"),
            f"ndcg@{k}": avg("ndcg"),
            "rouge_l": avg("rouge_l"),
            "bleu_4": avg("bleu_4"),
            "semantic_sim": avg("sem_sim"),
            "faithfulness": avg("faithfulness"),
            "avg_latency_ms": avg("latency_ms"),
        }
        # Optional BERTScore (batched, slow)
        if HAS_BERTSCORE:
            preds = [q.answer for q in per_query if q.expected_answer]
            refs = [q.expected_answer for q in per_query if q.expected_answer]
            if preds:
                agg["bertscore_f1"] = bertscore_f1(preds, refs)
    else:
        agg = {}

    if progress:
        progress(1.0, "done")
    return EvalReport(
        run_id=run_id, n_queries=len(gold_set), per_query=per_query, aggregate=agg
    )


if __name__ == "__main__":
    print("ROUGE-L:", rouge_l("the quick brown fox", "the quick brown fox jumps over"))
    print("BLEU-4 :", bleu_n("the quick brown fox", "the quick brown fox jumps over"))
    print("Recall@3:", recall_at_k(["a", "b", "c", "d"], ["c", "x"], 3))
    print("MRR    :", reciprocal_rank(["a", "b", "c"], ["c"]))
    print("nDCG@3 :", ndcg_at_k(["a", "b", "c"], ["c", "a"], 3))
