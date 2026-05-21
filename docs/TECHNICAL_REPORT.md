# ChatBoot RAG - Technical Report

## Executive Summary

**ChatBoot RAG** is a comprehensive Retrieval-Augmented Generation chatbot application developed as a university graduation project. The system enables users to chat with indexed datasets using a sophisticated hybrid retrieval pipeline combining dense vector search (transformer embeddings), sparse keyword search (BM25), and cross-encoder reranking. The application is built with Python and Streamlit, providing both a simple User Mode for end users and a comprehensive Developer Mode for dataset management, configuration, and evaluation.

This technical report provides an in-depth analysis of the system's architecture, implementation details, algorithms, data structures, and configuration options. The report is organized to serve both high-level understanding needs and low-level implementation reference requirements.

---

## Table of Contents

- [1. System Architecture Overview](#1-system-architecture-overview)
  - [1.1 High-Level Architecture](#11-high-level-architecture)
  - [1.2 Component Interaction Diagram](#12-component-interaction-diagram)
  - [1.3 Data Flow Analysis](#13-data-flow-analysis)
- [2. Core Engine Modules](#2-core-engine-modules)
  - [2.1 Preprocessing Module](#21-preprocessing-module)
  - [2.2 RAG Engine Module](#22-rag-engine-module)
  - [2.3 LLM Module](#23-llm-module)
  - [2.4 NLP Engine Module](#24-nlp-engine-module)
  - [2.5 Computer Vision Engine](#25-computer-vision-engine)
  - [2.6 Evaluation Module](#26-evaluation-module)
- [3. Algorithms and Techniques](#3-algorithms-and-techniques)
  - [3.1 Hybrid Retrieval Pipeline](#31-hybrid-retrieval-pipeline)
  - [3.2 BM25 Algorithm](#32-bm25-algorithm)
  - [3.3 FAISS HNSW Indexing](#33-faiss-hnsw-indexing)
  - [3.4 Reciprocal Rank Fusion](#34-reciprocal-rank-fusion)
  - [3.5 Cross-Encoder Reranking](#35-cross-encoder-reranking)
  - [3.6 Intent Classification](#36-intent-classification)
  - [3.7 Sentiment Analysis](#37-sentiment-analysis)
  - [3.8 Named Entity Recognition](#38-named-entity-recognition)
  - [3.9 OCR Pipeline](#39-ocr-pipeline)
- [4. Data Structures and Storage](#4-data-structures-and-storage)
  - [4.1 Core Data Classes](#41-core-data-classes)
  - [4.2 Index File Format](#42-index-file-format)
  - [4.3 Session State Management](#43-session-state-management)
- [5. API Integrations](#5-api-integrations)
  - [5.1 Anthropic Claude](#51-anthropic-claude)
  - [5.2 OpenAI GPT](#52-openai-gpt)
  - [5.3 Ollama Local](#53-ollama-local)
  - [5.4 Extractive Fallback](#54-extractive-fallback)
- [6. User Interface Architecture](#6-user-interface-architecture)
  - [6.1 Streamlit Application Structure](#61-streamlit-application-structure)
  - [6.2 Mode System](#62-mode-system)
  - [6.3 Developer Mode Tabs](#63-developer-mode-tabs)
  - [6.4 Caching Strategy](#64-caching-strategy)
- [7. Configuration Reference](#7-configuration-reference)
  - [7.1 PreprocessConfig](#71-preprocessconfig)
  - [7.2 RAGConfig](#72-ragconfig)
  - [7.3 LLMConfig](#73-llmconfig)
  - [7.4 CVConfig](#74-cvconfig)
- [8. Dependencies and Requirements](#8-dependencies-and-requirements)
  - [8.1 Python Packages](#81-python-packages)
  - [8.2 System Dependencies](#82-system-dependencies)
- [9. Performance Considerations](#9-performance-considerations)
  - [9.1 Fallback Chains](#91-fallback-chains)
  - [9.2 Lazy Loading](#92-lazy-loading)
  - [9.3 Memory Management](#93-memory-management)
- [10. Conclusion](#10-conclusion)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

ChatBoot RAG follows a layered, modular architecture that separates concerns across distinct components:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                               │
│                          streamlit_app.py (UI)                            │
│                    User Mode + Developer Mode (7 tabs)                    │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │   RAG Engine      │ │   NLP Engine      │ │   CV Engine       │
        │ (Retrieval)      │ │  (Analysis)       │ │    (OCR)          │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │                   │                   │
        ┌───────────┼───────────────────┼───────────────────┼────────────┐
        │           │                   │                   │            │
        ▼           ▼                   ▼                   ▼            │
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐       │
│  Preprocessing    │ │       LLM         │ │   Evaluation      │       │
│   (Data Prep)     │ │  (Generation)    │ │    (Metrics)      │       │
└───────────────────┘ └───────────────────┘ └───────────────────┘       │
        │                   │                   │                         │
        └───────────────────┴───────────────────┴─────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                    EXTERNAL SERVICES                           │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
        │  │Anthropic │  │ OpenAI   │  │  Ollama  │  │Tesseract │        │
        │  │ (Claude) │  │  (GPT)   │  │ (Local)  │  │   (OCR)  │        │
        │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
        └──────────────────────────────────────────────────────────────────┘
```

The architecture consists of three primary layers:

**Presentation Layer**: The Streamlit web application provides both User Mode (simple chat interface) and Developer Mode (comprehensive management console with 7 tabs).

**Processing Layer**: Six core modules handle specific aspects of the RAG pipeline:
- Preprocessing handles document loading, text cleaning, and chunking
- RAG Engine manages embeddings, vector indexing, and retrieval
- LLM Module generates answers using various providers
- NLP Engine provides query analysis (intent, sentiment, entities)
- CV Engine handles OCR for images and scanned PDFs
- Evaluation Module computes metrics for pipeline assessment

**Service Layer**: External services including LLM providers (Anthropic, OpenAI, Ollama) and OCR engine (Tesseract).

### 1.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              streamlit_app.py                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ init_state() → Session State Initialization                         │   │
│  │   - mode, active_index, chat_history                                │   │
│  │   - preproc_cfg, rag_cfg, llm_cfg                                   │   │
│  │   - logs, last_eval, show_sources                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ get_rag_engine()│    │get_nlp_engine()│    │ get_cv_engine() │
│ @st.cache_res.  │    │ @st.cache_res. │    │ @st.cache_res.  │
│                 │    │                 │    │                 │
│ - RAGEngine     │    │ - NLPEngine     │    │ - CVEngine      │
│   - build_index│    │   - train_intent│    │   - ocr_image   │
│   - retrieve   │    │   - analyze     │    │   - ocr_pdf     │
│   - load_index │    │   - sentiment   │    │   - preprocess  │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         │        ┌──────────────┼──────────────┐       │
         │        ▼              ▼              ▼       │
         │  ┌──────────────────────────────────────────┐ │
         │  │           generate_answer()               │ │
         │  │  - _call_anthropic() → Claude            │ │
         │  │  - _call_openai()    → GPT               │ │
         │  │  - _call_ollama()    → Local Llama       │ │
         │  │  - _extractive()     → Fallback          │ │
         │  └──────────────────────────────────────────┘ │
         │                      │                         │
         │                      ▼                         │
         │  ┌──────────────────────────────────────────┐ │
         │  │        run_evaluation()                   │ │
         │  │  - precision_at_k, recall_at_k            │ │
         │  │  - rouge_l, bleu_n                        │ │
         │  │  - semantic_similarity                    │ │
         │  └──────────────────────────────────────────┘ │
         │                      │                         │
         └──────────────────────┴─────────────────────────┘
```

### 1.3 Data Flow Analysis

The system operates in two distinct phases: Indexing (offline) and Querying (online).

#### Indexing Phase (Offline Processing)

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│   Upload    │───▶│  load_docs   │───▶│  preprocess   │───▶│  chunking   │
│   (JSON/    │    │              │    │  (cleaning,  │    │  (word      │
│   CSV/TXT/ │    │              │    │   normalize) │    │   window)   │
│   MD/PDF)   │    │              │    │              │    │             │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬──────┘
                                                                      │
                     ┌───────────────────────────────────────────────┘
                     ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│  FAISS      │◀───│   embed      │◀───│   build_index │◀───│   chunks    │
│  HNSW       │    │  (MiniLM)   │    │               │    │   list      │
│  index      │    │              │    │              │    │             │
└─────────────┘    └──────────────┘    └───────────────┘    └──────┬──────┘
     │                                                           │
     │                ┌───────────────────────────────────────────┘
     ▼                ▼
┌─────────────┐    ┌──────────────┐
│   BM25      │◀───│   build      │
│   index     │    │   (Okapi)   │
└─────────────┘    └──────────────┘
```

1. **Document Upload**: Files in JSON, CSV, TXT, MD format are uploaded through the Developer Mode interface. Images and PDFs trigger the OCR pipeline first.

2. **Document Loading**: The `load_documents()` function parses files and extracts source, text, and metadata. JSON files are recursively flattened to a maximum depth of 5 levels.

3. **Preprocessing Pipeline**: Each document passes through:
   - Unicode NFKC normalization
   - Lowercase conversion (if enabled)
   - Punctuation stripping (if enabled)
   - Stopword removal (if enabled)
   - spaCy lemmatization (if enabled)

4. **Chunking**: Cleaned text is split into overlapping word-based chunks using a sliding window algorithm:
   - Default chunk size: 256 words
   - Default overlap: 32 words
   - Minimum chunk size: 8 words

5. **Deduplication**: Identical chunks (by MD5 hash) are removed.

6. **Index Building**:
   - Dense embeddings generated using sentence-transformers (MiniLM-L6-v2 by default)
   - FAISS HNSW index created (M=32, efConstruction=200)
   - BM25 index built using Okapi formula

#### Querying Phase (Online Processing)

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│  User      │───▶│   NLP        │───▶│   Dense       │───▶│   BM25      │
│  Query     │    │   Analysis   │    │   Retrieval   │    │   Retrieval │
│            │    │  (intent,    │    │  (FAISS kNN)  │    │  (Okapi)    │
│            │    │   sentiment,  │    │               │    │             │
│            │    │   entities)  │    │               │    │             │
└─────────────┘    └──────────────┘    └───────┬───────┘    └──────┬──────┘
                                                │                  │
                              ┌──────────────────┴──────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Reciprocal Rank     │
                    │     Fusion (RRF)    │
                    │   k=60              │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
           ┌─────────────┐        ┌─────────────┐
           │   Cross-   │        │   Skip      │
           │  Encoder   │        │  (if no    │
           │  Rerank    │        │  reranker)  │
           └──────┬──────┘        └──────┬──────┘
                  │                      │
                  └──────────┬───────────┘
                             ▼
                    ┌─────────────────────┐
                    │  LLM Generation     │
                    │  (Claude/GPT/       │
                    │   Ollama/Extractive)│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Answer with       │
                    │   Source Citations  │
                    └─────────────────────┘
```

1. **Query Input**: User enters a query in the chat interface.

2. **NLP Analysis**: The query passes through:
   - Intent classification (TF-IDF + Logistic Regression)
   - Sentiment analysis (VADER)
   - Named entity recognition (spaCy)
   - Keyword extraction (TF-IDF)

3. **Dense Retrieval**: Query embedding generated using the same model as indexing. FAISS HNSW performs approximate k-nearest-neighbor search.

4. **Sparse Retrieval**: Query tokenized and BM25 scores computed against the corpus.

5. **Reciprocal Rank Fusion**: Results from both retrievers combined using the RRF formula with k=60:
   ```
   RRF_Score(d) = Σ (1 / (k + rank_i(d))) for all retrievers i
   ```

6. **Cross-Encoder Reranking** (if enabled): Top candidates re-scored using a cross-encoder model that jointly encodes query-document pairs.

7. **LLM Generation**: Final context passed to the LLM (or extractive fallback) to generate the answer with citations.

8. **Response Delivery**: Answer returned with expandable source citations showing retrieved chunks and similarity scores.

---

## 2. Core Engine Modules

### 2.1 Preprocessing Module

**File**: `core/preprocessing.py`

**Purpose**: Handles all document preprocessing before indexing, including loading, cleaning, tokenizing, and chunking.

#### Key Classes

**Chunk Dataclass**
```python
@dataclass
class Chunk:
    chunk_id: str           # Unique identifier: "source::record_idx::chunk_idx"
    text: str               # Cleaned text used for retrieval (normalized)
    raw_text: str          # Original unclean text (for display/citations)
    source: str            # Original file name
    metadata: dict = field(default_factory=dict)
```

**PreprocessConfig Dataclass**
```python
@dataclass
class PreprocessConfig:
    lowercase: bool = True           # Convert to lowercase + NFKC normalize
    strip_punctuation: bool = True   # Remove punctuation characters
    remove_stopwords: bool = False  # Remove English stopwords
    lemmatize: bool = True          # Use spaCy for lemmatization
    dedupe: bool = True             # Deduplicate chunks by hash
    chunk_size: int = 256           # Words per chunk
    chunk_overlap: int = 32         # Overlap between chunks
    min_chunk_words: int = 8        # Minimum words per chunk
```

#### Key Functions

| Function | Description |
|----------|-------------|
| `load_documents(path)` | Load JSON, CSV, TXT, MD files; returns list of `{source, text, metadata}` |
| `clean_text(text, cfg)` | Unicode NFKC normalization, lowercase, punctuation stripping |
| `tokenize(text, cfg)` | Tokenize with optional lemmatization and stopword removal |
| `chunk_words(text, chunk_size, overlap, min_words)` | Sliding window chunking algorithm |
| `preprocess(docs, cfg, progress)` | Full preprocessing pipeline producing Chunk objects |

#### Implementation Details

The chunking algorithm uses a sliding window approach:
```python
def chunk_words(text: str, chunk_size: int = 256, overlap: int = 32, min_words: int = 8) -> list[str]:
    tokens = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i:i + chunk_size])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
    return chunks
```

Deduplication uses MD5 hashing of cleaned text content, ensuring unique chunks in the index.

### 2.2 RAG Engine Module

**File**: `core/rag_engine.py`

**Purpose**: Core retrieval pipeline with dense embeddings, FAISS indexing, BM25 sparse retrieval, hybrid fusion, and cross-encoder reranking.

#### Key Classes

**RAGConfig Dataclass**
```python
@dataclass
class RAGConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True
    use_bm25: bool = True
    top_k_dense: int = 8
    top_k_bm25: int = 8
    top_k_final: int = 5
    rrf_k: int = 60
    hnsw_M: int = 32
    hnsw_ef_construction: int = 200
```

**RetrievalResult Dataclass**
```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None
```

#### RAGEngine Methods

| Method | Description |
|--------|-------------|
| `build_index(name, chunks, progress)` | Embed chunks, build FAISS + BM25 indexes, persist to disk |
| `load_index(name)` | Load previously-built index from disk |
| `list_indexes()` | List all persisted indexes with metadata |
| `delete_index(name)` | Remove index from disk and memory |
| `retrieve(query, name, top_k)` | Run full retrieval pipeline, return results + timing |

#### Supported Embedding Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| all-MiniLM-L6-v2 | 384 | Default, fast, good quality |
| all-mpnet-base-v2 | 768 | Higher quality, slower |
| bge-large-en-v1.5 | 1024 | Best quality, heaviest |
| e5-large-v2 | 1024 | Good for retrieval tasks |

### 2.3 LLM Module

**File**: `core/llm.py`

**Purpose**: Multi-provider LLM integration for answer generation with automatic fallback.

#### Key Classes

**LLMConfig Dataclass**
```python
@dataclass
class LLMConfig:
    provider: str = "auto"                    # auto | anthropic | openai | ollama | extractive
    anthropic_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
```

#### Key Function

```python
def generate_answer(
    query: str,
    contexts: list[str],
    cfg: LLMConfig | None = None,
    history: list[dict] | None = None
) -> tuple[str, str]:
    """Generate answer to query, grounded in contexts.
    
    Returns:
        tuple: (answer_text, provider_used)
    """
```

#### Provider Priority (auto mode)
1. **Anthropic (Claude)** - requires `ANTHROPIC_API_KEY` environment variable
2. **OpenAI (GPT)** - requires `OPENAI_API_KEY` environment variable
3. **Ollama (Local)** - requires running Ollama server at configured host
4. **Extractive Fallback** - always works, concatenates top contexts with cite markers

#### System Prompt

```
You are a precise, helpful assistant. Use ONLY the provided context to answer. 
Cite sources inline using bracket notation like [1], [2] matching the context 
numbers. If the context does not contain the answer, say so plainly. 
Be concise and well-structured (use short paragraphs or bullet lists).
```

### 2.4 NLP Engine Module

**File**: `core/nlp_engine.py`

**Purpose**: Query analysis including intent classification, sentiment analysis, named entity recognition, and keyword extraction.

#### Key Classes

**NLPAnalysis Dataclass**
```python
@dataclass
class NLPAnalysis:
    intent: str = "unknown"
    intent_confidence: float = 0.0
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    entities: list[dict] = field(default_factory=list)  # [{"label", "text"}, ...]
    keywords: list[str] = field(default_factory=list)
```

#### Default Intent Categories

| Intent | Trigger Keywords |
|--------|------------------|
| definition | what is, define, explain, describe |
| compare | compare, vs, difference between |
| summarize | summarize, key points, tldr |
| calculate | calculate, compute, how much |

#### NLPEngine Methods

| Method | Description |
|--------|-------------|
| `train_intent(X, y)` | Train TF-IDF + Logistic Regression classifier |
| `predict_intent(text)` | Predict intent with confidence score |
| `analyze_sentiment(text)` | VADER-based sentiment analysis |
| `extract_entities(text)` | spaCy NER (or fallback regex) |
| `extract_keywords(text, top_k)` | TF-IDF-based keyword extraction |
| `analyze(text)` | Full NLP analysis pipeline |

### 2.5 Computer Vision Engine

**File**: `core/cv_engine.py`

**Purpose**: OCR for images and scanned PDFs using Tesseract and OpenCV preprocessing.

#### Key Classes

**OCRResult Dataclass**
```python
@dataclass
class OCRResult:
    text: str
    confidence: float       # Average word confidence (0-1)
    page: int = 0
    word_count: int = 0
```

**CVConfig Dataclass**
```python
@dataclass
class CVConfig:
    deskew: bool = True               # Automatic image deskewing
    denoise: bool = True              # Image denoising
    adaptive_threshold: bool = True   # Adaptive thresholding
    min_confidence: float = 30.0     # Minimum word confidence
    tesseract_lang: str = "eng"
    dpi: int = 300                   # PDF rasterization DPI
```

#### CVEngine Methods

| Method | Description |
|--------|-------------|
| `preprocess_image(img)` | Grayscale, denoise, threshold, deskew |
| `ocr_image(path)` | OCR single image file |
| `ocr_pdf(path)` | OCR each page of PDF |
| `file_to_text(path)` | Auto-route based on extension |

#### Supported Formats
PNG, JPG, JPEG, TIF, TIFF, BMP, PDF

### 2.6 Evaluation Module

**File**: `core/evaluation.py`

**Purpose**: Comprehensive metrics for RAG pipeline evaluation.

#### Key Classes

**QueryResult Dataclass**
```python
@dataclass
class QueryResult:
    query: str
    answer: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    expected_answer: str
    contexts: list[str]
    latency_ms: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    rouge_l: float = 0.0
    bleu_4: float = 0.0
    sem_sim: float = 0.0
    faithfulness: float = 0.0
```

**EvalReport Dataclass**
```python
@dataclass
class EvalReport:
    run_id: str
    n_queries: int
    per_query: list[QueryResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
```

#### Metric Functions

**Retrieval Metrics:**
- `precision_at_k(retrieved, relevant, k)`
- `recall_at_k(retrieved, relevant, k)`
- `reciprocal_rank(retrieved, relevant)` - MRR
- `ndcg_at_k(retrieved, relevant, k)`

**Generation Metrics:**
- `rouge_l(prediction, reference)` - ROUGE-L F1
- `bleu_n(prediction, reference, n)` - BLEU-4

**Semantic Metrics:**
- `semantic_similarity(prediction, reference)` - Cosine similarity of embeddings
- `bertscore_f1(predictions, references)` - BERTScore F1 (optional)

**Faithfulness Metrics:**
- `lexical_faithfulness(answer, contexts)` - Token overlap
- `semantic_faithfulness(answer, contexts)` - Max semantic similarity

---

## 3. Algorithms and Techniques

### 3.1 Hybrid Retrieval Pipeline

The hybrid retrieval system combines dense (embedding-based) and sparse (keyword-based) retrieval methods to leverage the strengths of both approaches:

- **Dense Retrieval**: Captures semantic similarity and handles synonymy
- **Sparse Retrieval**: Excels at exact keyword matching and rare terms
- **Combination**: Provides more robust retrieval than either method alone

The pipeline proceeds as follows:
1. Generate query embedding using sentence-transformer
2. Perform dense kNN search on FAISS HNSW index
3. Perform sparse BM25 search on tokenized corpus
4. Combine results using Reciprocal Rank Fusion
5. Optionally rerank top candidates with cross-encoder

### 3.2 BM25 Algorithm

**Implementation**: Uses `rank_bm25.BM25Okapi` from the rank-bm25 library

**Scoring Formula**:
```
BM25(D, Q) = Σ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
```

Where:
- `f(q_i, D)` = term frequency of q_i in document D
- `|D|` = document length
- `avgdl` = average document length in collection
- `k1` = term frequency saturation parameter (default: 1.5)
- `b` = document length normalization parameter (default: 0.75)
- `IDF(q_i)` = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5))

**Configuration** (via rank-bm25 defaults):
- k1 = 1.5
- b = 0.75

### 3.3 FAISS HNSW Indexing

**Index Type**: Hierarchical Navigable Small World (HNSW)

**Configuration**:
```python
hnsw_M: int = 32                    # Number of bi-directional links per node
hnsw_ef_construction: int = 200     # Search width during index construction
```

**Creation**:
```python
index = faiss.IndexHNSWFlat(embeddings.shape[1], self.cfg.hnsw_M)
index.hnsw.efConstruction = self.cfg.hnsw_ef_construction
index.add(embeddings)
```

**Search**:
```python
D, I = index.search(query_emb.reshape(1, -1).astype("float32"), k)
# Convert L2 distance to cosine similarity for normalized vectors
# ||a - b||^2 = 2(1 - cos(a,b)) for unit vectors
sims = 1 - D[0] / 2
```

**HNSW Characteristics**:
- Approximate nearest neighbor search
- logarithmic search complexity
- Memory-efficient compared to exact methods
- Quality/speed trade-off controlled by M and efConstruction parameters

### 3.4 Reciprocal Rank Fusion

**Algorithm**: Combines multiple ranked retrieval results into a single ranking

**Formula**:
```
RRF_Score(d) = Σ (1 / (k + rank_retriever(d))) for all retrievers
```

**Default k value**: 60 (standard in information retrieval literature)

**Implementation**:
```python
@staticmethod
def _rrf_fuse(rankings: list[list[tuple[int, float]]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused
```

**Properties**:
- Parameter-free (except k)
- Handles rank ties gracefully
- No relevance score normalization required
- Benefits from combining diverse retrieval methods

### 3.5 Cross-Encoder Reranking

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Process**:
1. Take top candidates from RRF fusion
2. Create query-document pairs
3. Score each pair using cross-encoder
4. Re-rank by cross-encoder scores

**Why Cross-Encoders Are Better**:
- Joint encoding of query and document (vs. separate encoding in bi-encoders)
- Captures fine-grained interaction between query terms and document
- More accurate at cost of slower inference

**Trade-off**:
- Cross-encoders: accurate but slow (apply to top-k only)
- Bi-encoders: fast but less accurate (apply to entire corpus)

### 3.6 Intent Classification

**Pipeline**: TF-IDF + Logistic Regression

**Configuration**:
```python
Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
```

**Training Data Generation**:
- Seed phrases for each intent category
- Expanded with topic fillers (medical, technical terms)
- Creates balanced training set

**Prediction**:
```python
proba = self._intent_clf.predict_proba([text])[0]
idx = int(proba.argmax())
return self._intent_labels[idx], float(proba[idx])
```

### 3.7 Sentiment Analysis

**Implementation**: VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Classification Thresholds**:
- Compound >= 0.05: Positive
- Compound <= -0.05: Negative
- Between -0.05 and 0.05: Neutral

**Fallback**: Simple lexicon-based sentiment if VADER unavailable

### 3.8 Named Entity Recognition

**Primary**: spaCy `en_core_web_sm` model

**Supported Entity Types**:
- PERSON: People
- ORG: Organizations
- GPE: Countries/Cities
- DATE, TIME, MONEY, PERCENT, etc.

**Fallback**: Capitalized n-gram extraction with MISC label

### 3.9 OCR Pipeline

**Image Preprocessing**:
1. Grayscale conversion (if color)
2. FastNlMeansDenoising (h=10)
3. Adaptive Gaussian thresholding (block size=31, C=10)
4. Deskewing via minAreaRect angle calculation

**Tesseract Integration**:
- Uses `image_to_data()` for per-word confidence
- Filters words below minimum confidence threshold (default: 30)
- Extracts text with bounding box information

**PDF Processing**:
- Uses pdf2image to convert pages to PIL images
- Processes each page individually
- Concatenates results with page markers

---

## 4. Data Structures and Storage

### 4.1 Core Data Classes

#### Chunk
```python
@dataclass
class Chunk:
    chunk_id: str           # Format: "source::record_idx::chunk_idx"
    text: str               # Normalized (lowercased, cleaned, lemmatized)
    raw_text: str           # Original text for display
    source: str             # Original filename
    metadata: dict          # Additional data (record_idx, raw record)
```

#### RetrievalResult
```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float            # Combined RRF score
    dense_rank: Optional[int]    # Rank in dense results
    sparse_rank: Optional[int]   # Rank in BM25 results
    rerank_score: Optional[float]  # Cross-encoder score
```

#### NLPAnalysis
```python
@dataclass
class NLPAnalysis:
    intent: str = "unknown"
    intent_confidence: float = 0.0
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    entities: list[dict] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
```

### 4.2 Index File Format

**Directory Structure**:
```
indexes/
└── <dataset_name>/
    ├── chunks.json          # Chunk data (JSON)
    ├── embeddings.npy       # Dense embeddings (NumPy binary)
    ├── faiss.index          # FAISS index (binary)
    ├── bm25.pkl             # BM25 model (pickle)
    └── meta.json            # Index metadata (JSON)
```

**chunks.json Format**:
```json
[
  {
    "chunk_id": "source::0::0",
    "text": "cleaned normalized text",
    "raw_text": "original raw text",
    "source": "filename.json",
    "metadata": {"record_idx": 0}
  }
]
```

**embeddings.npy Format**:
- Shape: (num_chunks, embedding_dim) - typically (N, 384)
- dtype: float32
- Storage: Row-major order, normalized to unit length

**meta.json Format**:
```json
{
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "n_chunks": 1234,
  "dim": 384
}
```

### 4.3 Session State Management

**Session State Variables** (streamlit_app.py):

| Variable | Type | Purpose |
|----------|------|---------|
| `mode` | string | Current mode: "User" or "Developer" |
| `active_index` | string | Currently selected dataset |
| `chat_history` | dict | Per-index chat histories |
| `chat_started` | dict | Per-index chat started flags |
| `preproc_cfg` | PreprocessConfig | Preprocessing settings |
| `rag_cfg` | RAGConfig | RAG engine settings |
| `llm_cfg` | LLMConfig | LLM generation settings |
| `logs` | list | In-memory log entries |
| `last_eval` | dict | Last evaluation report |
| `show_sources` | bool | Toggle source display |

---

## 5. API Integrations

### 5.1 Anthropic Claude

**Environment Variable**: `ANTHROPIC_API_KEY`

**Default Model**: `claude-haiku-4-5`

**Implementation**:
```python
def _call_anthropic(prompt: str, system: str, cfg: LLMConfig) -> str:
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text
```

### 5.2 OpenAI GPT

**Environment Variable**: `OPENAI_API_KEY`

**Default Model**: `gpt-4o-mini`

**Implementation**:
```python
def _call_openai(prompt: str, system: str, cfg: LLMConfig) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=cfg.openai_model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content
```

### 5.3 Ollama Local

**Default Host**: `http://localhost:11434`

**Default Model**: `llama3.1:8b`

**Implementation**:
```python
def _call_ollama(prompt: str, system: str, cfg: LLMConfig) -> str:
    body = json.dumps({
        "model": cfg.ollama_model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{cfg.ollama_host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data.get("response", "")
```

### 5.4 Extractive Fallback

**When Used**: No API keys available or all providers fail

**Implementation**:
```python
def _extractive(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "I couldn't find relevant information in the indexed documents."
    parts = []
    for i, ctx in enumerate(contexts[:3], 1):
        snippet = ctx if len(ctx) <= 300 else ctx[:300] + "..."
        parts.append(f"{snippet} [{i}]")
    return f"Based on the retrieved context:\n\n{'\\n\\n'.join(parts)}"
```

---

## 6. User Interface Architecture

### 6.1 Streamlit Application Structure

**Page Configuration**:
```python
st.set_page_config(
    page_title="ChatBoot RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

**Custom Dark Theme**:
```css
:root {
  --accent: #4a6cf7;
  --bg-panel: #10121a;
  --border: #1f2435;
  --text-main: #e8eaf2;
  --text-sub: #5a6280;
}
```

### 6.2 Mode System

**User Mode**: Simple chat interface
- Dataset selector in sidebar
- Chat with indexed data
- Toggle sources visibility
- Clear chat functionality
- Suggestion buttons

**Developer Mode**: Full management console
- 7 detailed tabs for different functions

### 6.3 Developer Mode Tabs

| Tab | Functionality |
|-----|---------------|
| 💬 Test Chat | Debug chat with retrieval trace, timing, NLP analysis |
| 📁 Datasets | Upload files, index local datasets, manage indexes |
| ⚙️ Preprocessing | Configure text cleaning and chunking parameters |
| 🧠 Models | Select embedding/reranker models, LLM provider, intent classifier |
| 🔗 Pipeline | Visual architecture diagram and algorithm documentation |
| 📊 Evaluation | Run evaluation with gold set, view metrics |
| 📜 Logs | Real-time log viewer with level filtering |

### 6.4 Caching Strategy

**@st.cache_resource Usage**:

1. **get_rag_engine()**: Singleton RAG engine instance
   ```python
   @st.cache_resource(show_spinner=False)
   def get_rag_engine() -> RAGEngine:
       return RAGEngine(persist_dir=INDEX_DIR)
   ```

2. **get_nlp_engine()**: NLP engine with lazy intent training
   ```python
   @st.cache_resource(show_spinner=False)
   def get_nlp_engine() -> NLPEngine:
       eng = NLPEngine(model_dir=MODELS_DIR)
       if eng._intent_clf is None:
           with st.spinner("Training intent classifier (first run only)…"):
               eng.train_intent()
       return eng
   ```

3. **get_cv_engine()**: Simple CV engine instance
   ```python
   @st.cache_resource(show_spinner=False)
   def get_cv_engine() -> CVEngine:
       return CVEngine()
   ```

---

## 7. Configuration Reference

### 7.1 PreprocessConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| lowercase | bool | True | Lowercase + NFKC normalize |
| strip_punctuation | bool | True | Remove punctuation |
| remove_stopwords | bool | False | Remove English stopwords |
| lemmatize | bool | True | Use spaCy lemmatization |
| dedupe | bool | True | Deduplicate by hash |
| chunk_size | int | 256 | Words per chunk |
| chunk_overlap | int | 32 | Overlap between chunks |
| min_chunk_words | int | 8 | Minimum chunk size |

### 7.2 RAGConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| embed_model | str | all-MiniLM-L6-v2 | Embedding model |
| reranker_model | str | ms-marco-MiniLM-L-6-v2 | Reranker model |
| use_reranker | bool | True | Enable reranking |
| use_bm25 | bool | True | Enable BM25 |
| top_k_dense | int | 8 | Dense retrieval top-k |
| top_k_bm25 | int | 8 | BM25 retrieval top-k |
| top_k_final | int | 5 | Final results |
| rrf_k | int | 60 | RRF constant |
| hnsw_M | int | 32 | HNSW M parameter |
| hnsw_ef_construction | int | 200 | HNSW efConstruction |

### 7.3 LLMConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| provider | str | auto | LLM provider |
| anthropic_model | str | claude-haiku-4-5 | Claude model |
| openai_model | str | gpt-4o-mini | GPT model |
| ollama_model | str | llama3.1:8b | Ollama model |
| ollama_host | str | http://localhost:11434 | Ollama endpoint |
| temperature | float | 0.2 | Generation temperature |
| max_tokens | int | 1024 | Max tokens |

### 7.4 CVConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| deskew | bool | True | Enable deskewing |
| denoise | bool | True | Enable denoising |
| adaptive_threshold | bool | True | Enable thresholding |
| min_confidence | float | 30.0 | Minimum OCR confidence |
| tesseract_lang | str | eng | Tesseract language |
| dpi | int | 300 | PDF rasterization DPI |

---

## 8. Dependencies and Requirements

### 8.1 Python Packages

| Category | Package | Version |
|----------|---------|---------|
| Web | streamlit | >=1.30, <2.0 |
| Data | pandas | >=2.0 |
| Compute | numpy | >=1.24 |
| ML | scikit-learn | >=1.3 |
| DL | sentence-transformers | >=2.5 |
| Vector Search | faiss-cpu | >=1.7.4 |
| Sparse Search | rank-bm25 | >=0.2.2 |
| NLP | spacy | >=3.7 |
| NLP | vaderSentiment | >=3.3.2 |
| CV | opencv-python | >=4.9 |
| CV | pytesseract | >=0.3.10 |
| CV | Pillow | >=10.0 |
| CV | pdf2image | >=1.16 |
| Metrics | rouge-score | >=0.1.2 |
| LLM | anthropic | >=0.39 |
| LLM | openai | >=1.40 |

### 8.2 System Dependencies

**For Full OCR Functionality**:

- **Tesseract OCR**: `apt install tesseract-ocr` (Linux) or Windows installer
- **Poppler**: `apt install poppler-utils` (Linux) or Windows installer

**Optional**:
- **spaCy Model**: `python -m spacy download en_core_web_sm`

---

## 9. Performance Considerations

### 9.1 Fallback Chains

The system implements graceful degradation at multiple levels:

| Component | Primary | Fallback |
|-----------|---------|----------|
| Embeddings | sentence-transformers | Hashing (384-dim BLAKE2b) |
| spaCy | en_core_web_sm | Blank model or regex tokenization |
| FAISS | IndexHNSWFlat | Numpy cosine similarity |
| BM25 | rank-bm25 | Disabled if unavailable |
| LLM | Anthropic/OpenAI/Ollama | Extractive fallback |

### 9.2 Lazy Loading

All heavy models are lazy-loaded:
- Sentence-transformer encoder only loaded on first retrieval
- Cross-encoder reranker only loaded when enabled and needed
- spaCy model loaded on first NLP analysis
- Intent classifier trained/loaded on first Developer Mode access

### 9.3 Memory Management

- Session state scoped to user session
- Indexes loaded on demand (one at a time)
- Chunk data in memory limited to active index
- Rolling log buffer (max 500 entries)

---

## 10. Conclusion

ChatBoot RAG demonstrates a complete, production-ready implementation of a modern Retrieval-Augmented Generation system. The architecture successfully combines multiple retrieval paradigms (dense, sparse, reranking) with flexible LLM integration and comprehensive evaluation capabilities.

**Key Strengths**:

1. **Modular Design**: Clean separation between preprocessing, retrieval, generation, and evaluation
2. **Graceful Degradation**: Multiple fallback levels ensure system works without external dependencies
3. **Hybrid Retrieval**: Combines semantic and keyword search for robust results
4. **Flexible LLM Integration**: Supports multiple providers with automatic fallback
5. **Comprehensive NLP**: Full query analysis pipeline with intent, sentiment, entities, keywords
6. **OCR Support**: Handles images and scanned PDFs through Tesseract
7. **Extensive Evaluation**: 15+ metrics covering retrieval, generation, and faithfulness

**Technical Achievements**:

- BM25 algorithm for sparse retrieval
- FAISS HNSW indexing for efficient approximate nearest neighbor search
- Reciprocal Rank Fusion for combining retrieval methods
- Cross-encoder reranking for improved result quality
- TF-IDF + Logistic Regression for intent classification
- VADER for sentiment analysis
- spaCy for named entity recognition

The system provides both a simple User Mode for end-user interaction and a comprehensive Developer Mode for dataset management, configuration, and evaluation, making it suitable for both production use and educational purposes in understanding RAG systems.

---

*Technical Report generated for ChatBoot RAG - University Graduation Project*