"""ChatBoot core engines."""
from .preprocessing import (
    Chunk, PreprocessConfig, load_documents, preprocess,
    clean_text, tokenize, chunk_words,
)
from .rag_engine import RAGEngine, RAGConfig, RetrievalResult
from .nlp_engine import NLPEngine, NLPAnalysis
from .cv_engine import CVEngine, CVConfig, OCRResult
from .evaluation import (
    run_evaluation, EvalReport, QueryResult,
    precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k,
    rouge_l, bleu_n, semantic_similarity, semantic_faithfulness,
)
from .llm import generate_answer, LLMConfig

__all__ = [
    "Chunk", "PreprocessConfig", "load_documents", "preprocess",
    "clean_text", "tokenize", "chunk_words",
    "RAGEngine", "RAGConfig", "RetrievalResult",
    "NLPEngine", "NLPAnalysis",
    "CVEngine", "CVConfig", "OCRResult",
    "run_evaluation", "EvalReport", "QueryResult",
    "precision_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k",
    "rouge_l", "bleu_n", "semantic_similarity", "semantic_faithfulness",
    "generate_answer", "LLMConfig",
]
