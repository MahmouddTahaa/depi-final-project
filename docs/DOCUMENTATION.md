# ChatBoot RAG - Comprehensive Documentation

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Quick Start Guide](#2-quick-start-guide)
- [3. User Guide](#3-user-guide)
  - [3.1 User Mode](#31-user-mode)
  - [3.2 Developer Mode](#32-developer-mode)
- [4. Developer Guide](#4-developer-guide)
  - [4.1 Architecture Overview](#41-architecture-overview)
  - [4.2 Core Modules](#42-core-modules)
  - [4.3 Configuration Reference](#43-configuration-reference)
  - [4.4 Extending the Application](#44-extending-the-application)
- [5. API Reference](#5-api-reference)
  - [5.1 Preprocessing Module](#51-preprocessing-module)
  - [5.2 RAG Engine Module](#52-rag-engine-module)
  - [5.3 LLM Module](#53-llm-module)
  - [5.4 NLP Engine Module](#54-nlp-engine-module)
  - [5.5 CV Engine Module](#55-cv-engine-module)
  - [5.6 Evaluation Module](#56-evaluation-module)
- [6. Datasets](#6-datasets)
- [7. Troubleshooting](#7-troubleshooting)
- [8. Appendix](#8-appendix)

---

# 1. Project Overview

**ChatBoot RAG** is a Retrieval-Augmented Generation (RAG) chatbot application built with Python and Streamlit. It enables users to chat with indexed datasets using a sophisticated hybrid retrieval pipeline combining dense (transformer embeddings) and sparse (BM25) retrieval methods.

## 1.1 Key Features

- **Dual Interface Modes**: Simple User Mode for chatting, Developer Mode for dataset management and configuration
- **Hybrid Retrieval**: Combines dense vector search (FAISS) with sparse BM25 retrieval using Reciprocal Rank Fusion
- **Cross-encoder Reranking**: Improves retrieval quality with transformer-based reranking
- **Multi-format Support**: Handles JSON, CSV, TXT, MD files, plus images and PDFs via OCR
- **Flexible LLM Integration**: Works with Anthropic Claude, OpenAI GPT, Ollama (local), or extractive fallback
- **Comprehensive NLP**: Intent classification, sentiment analysis, named entity recognition, keyword extraction
- **Extensive Evaluation**: 15+ metrics including Recall@k, MRR, nDCG, ROUGE, BLEU, semantic similarity, and faithfulness

## 1.2 Tech Stack

| Category | Technology |
|----------|------------|
| Web Framework | Streamlit |
| Deep Learning | sentence-transformers (MiniLM, MPNet, BGE) |
| Machine Learning | scikit-learn, FAISS, BM25 |
| NLP | spaCy, VADER |
| Computer Vision | OpenCV, pytesseract, pdf2image |
| LLM Providers | Anthropic, OpenAI, Ollama |
| Evaluation | ROUGE, BLEU, BERTScore |

## 1.3 Project Structure

```
chatboot_py/
├── streamlit_app.py         # Main Streamlit UI application
├── requirements.txt         # Python dependencies
├── core/                    # Core engine modules
│   ├── __init__.py         # Module exports
│   ├── preprocessing.py    # Document loading, cleaning, chunking
│   ├── rag_engine.py      # RAG pipeline (embeddings, FAISS, BM25, RRF)
│   ├── llm.py             # LLM providers (Anthropic, OpenAI, Ollama)
│   ├── nlp_engine.py      # Query analysis (intent, sentiment, entities)
│   ├── cv_engine.py       # OCR for images/PDFs
│   └── evaluation.py      # Metrics (Recall, ROUGE, BLEU, etc.)
├── datasets/               # Pre-built topic datasets (JSON)
├── data/                   # Uploaded files (runtime)
├── indexes/                # Generated RAG indexes (runtime)
├── models/                 # Saved ML models (runtime)
├── eval_runs/              # Evaluation reports (runtime)
├── run.sh / run.bat        # App launchers
├── start_app.bat          # Quick launcher
└── setup_once.bat         # One-time dependency installer
```

---

# 2. Quick Start Guide

## 2.1 Prerequisites

- Python 3.10+
- 8GB RAM (16GB recommended for full functionality)
- For OCR: Tesseract OCR and Poppler installed on system

## 2.2 Installation

### Windows (One-time Setup)

1. Run `setup_once.bat` to create virtual environment and install dependencies
2. Download spaCy model: `python -m spacy download en_core_web_sm`

### Windows (Running the App)

Run `start_app.bat` or `run.bat`

### Linux/macOS

```bash
chmod +x run.sh
./run.sh
```

### Manual Launch

```bash
# Activate virtual environment if using one
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Run the app
python -m streamlit run streamlit_app.py
```

## 2.3 Access the Application

Open your browser and navigate to: `http://localhost:8501`

---

# 3. User Guide

## 3.1 User Mode

User Mode provides a simple chat interface for interacting with your indexed datasets.

### Starting a Chat Session

1. **Select Mode**: Choose "User" from the sidebar radio button
2. **Select Dataset**: Choose an indexed dataset from the dropdown menu
3. **Start Chat**: Click "Start chat" button

### Asking Questions

Once a chat session is started, type your questions in the chat input field. The system will:
1. Analyze your query (intent, sentiment, entities)
2. Retrieve relevant context chunks
3. Generate an answer with source citations

### Available Actions

| Action | Description |
|--------|-------------|
| Suggestion Buttons | Quick queries like "Give me an overview", "What are the key topics?" |
| Source Expander | Expand to view retrieved evidence chunks with similarity scores |
| Clear Chat | Reset the conversation for the selected dataset |
| Show Sources Toggle | Toggle visibility of source citations in responses |

### Understanding Responses

Answers are displayed with:
- Main response text
- Expandable "Sources" section showing retrieved chunks
- Similarity scores for each retrieved chunk
- Source file references

## 3.2 Developer Mode

Developer Mode provides comprehensive tools for dataset management, configuration, and evaluation.

### Tab 1: Test Chat

Same as User Mode but with additional debugging panels:
- **Retrieval Trace**: Shows retrieved chunks with scores
- **Timing Info**: Latency breakdown for each pipeline stage
- **NLP Analysis**: Intent, sentiment, entities, keywords
- **Faithfulness Score**: How well the answer is grounded in retrieved context

### Tab 2: Datasets

**Upload Files**:
- Supported formats: JSON, CSV, TXT, MD, PNG, JPG, PDF
- Images and PDFs are processed through OCR automatically

**Index Local Datasets**:
- Select from 11 pre-built topic datasets
- Creates searchable index from bundled content

**Manage Indexes**:
- **Use**: Switch to an index for chatting
- **Preview**: View sample chunks from the index
- **Re-index**: Rebuild index with current preprocessing settings
- **Delete**: Remove an index and its files

### Tab 3: Preprocessing

Configure the document preprocessing pipeline:

| Setting | Description |
|---------|-------------|
| Lowercase | Convert text to lowercase with NFKC normalization |
| Strip Punctuation | Remove punctuation characters |
| Remove Stopwords | Filter out common English stopwords |
| Lemmatize | Use spaCy for word lemmatization |
| Deduplicate | Remove duplicate chunks by hash |
| Chunk Size | Number of words per chunk (default: 256) |
| Chunk Overlap | Overlap between consecutive chunks (default: 32) |
| Minimum Chunk Words | Minimum words required to create a chunk (default: 8) |

### Tab 4: Models

Configure retrieval and generation models:

**Embedding Settings**:
- Model selection: all-MiniLM-L6-v2, all-mpnet-base-v2, bge-large-en-v1.5, e5-large-v2
- Dimension: 384, 768, or 1024 based on model

**Reranker Settings**:
- Enable/disable cross-encoder reranking
- Model: ms-marco-MiniLM-L-6-v2

**Hybrid Retrieval**:
- Toggle BM25 sparse retrieval
- Adjust top-k values for dense, BM25, and final results

**LLM Settings**:
- Provider: auto, anthropic, openai, ollama, extractive
- Temperature: 0.0 to 1.5
- Max tokens: 128 to 4096

**Intent Classifier**:
- View training metrics
- Retrain classifier with custom data

### Tab 5: Pipeline

Visual architecture diagram showing the complete RAG pipeline:

**Indexing Flow**:
```
Raw docs → OCR → Clean → Chunk → Embed → FAISS HNSW + BM25
```

**Query Flow**:
```
Query → Intent Classify → Dense kNN ⊕ BM25 → Rerank → LLM → Answer
```

Includes algorithm table detailing all components.

### Tab 6: Evaluation

Run comprehensive evaluation on your RAG pipeline:

**Gold Set**: Upload JSON file with query-answer-chunk mappings, or auto-sample from active index

**Configuration**:
- k values for retrieval metrics
- Sample size

**Metrics Reported**:
- **Retrieval**: Precision@k, Recall@k, MRR, nDCG@k
- **Generation**: ROUGE-L, BLEU-4
- **Semantic**: Similarity, BERTScore
- **Faithfulness**: Lexical and semantic

**Output**: Aggregate scores + per-query breakdown

### Tab 7: Logs

View application logs with filtering:
- Log level: INFO, WARN, ERROR
- Timestamp, module, and message for each entry

---

# 4. Developer Guide

## 4.1 Architecture Overview

ChatBoot RAG follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        streamlit_app.py                          │
│                    (UI Layer - Streamlit)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ RAG Engine  │ │ NLP Engine  │ │  CV Engine  │              │
│  │ (Retrieval) │ │  (Analysis) │ │   (OCR)    │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │Preprocessing│ │    LLM      │ │  Evaluation │              │
│  │   (Data)    │ │(Generation) │ │   (Metrics) │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                     External Services                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │Anthropic │ │ OpenAI   │ │  Ollama  │ │ Tesseract│        │
│  │ (Claude) │ │  (GPT)   │ │ (Local)  │ │   (OCR)  │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Indexing (Offline)**:
1. Load documents (JSON, CSV, TXT, MD)
2. OCR images/PDFs if needed
3. Clean and normalize text
4. Chunk into smaller pieces
5. Generate embeddings
6. Build FAISS index and BM25 model

**Query (Online)**:
1. Parse user query
2. Analyze query (intent, sentiment, entities)
3. Generate query embedding
4. Dense retrieval (FAISS kNN)
5. Sparse retrieval (BM25)
6. Combine with Reciprocal Rank Fusion
7. Rerank with cross-encoder
8. Generate answer with LLM
9. Return answer with citations

## 4.2 Core Modules

### Preprocessing Module (`core/preprocessing.py`)

Handles all document preprocessing before indexing.

**Key Classes**:

```python
@dataclass
class Chunk:
    chunk_id: str           # Unique identifier
    text: str              # Cleaned text for retrieval
    raw_text: str          # Original text
    source: str            # Source file name
    metadata: dict         # Additional metadata

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
```

**Key Functions**:
- `load_documents(path)`: Load files from various formats
- `clean_text(text, cfg)`: Apply text cleaning
- `tokenize(text, cfg)`: Tokenize with optional lemmatization
- `chunk_words(text, chunk_size, overlap, min_words)`: Create overlapping chunks
- `preprocess(docs, cfg, progress)`: Full preprocessing pipeline

### RAG Engine Module (`core/rag_engine.py`)

Core retrieval pipeline with hybrid dense+sparse search.

**Key Classes**:

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

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None
```

**Key Methods**:
- `build_index(name, chunks, progress)`: Create new index
- `load_index(name)`: Load existing index
- `list_indexes()`: List all indexes
- `delete_index(name)`: Remove index
- `retrieve(query, name, top_k)`: Run retrieval pipeline

**Supported Embedding Models**:
| Model | Dimensions | Description |
|-------|------------|-------------|
| all-MiniLM-L6-v2 | 384 | Default, fast |
| all-mpnet-base-v2 | 768 | Higher quality |
| bge-large-en-v1.5 | 1024 | Best quality |
| e5-large-v2 | 1024 | Good for retrieval |

### LLM Module (`core/llm.py`)

Answer generation with multiple provider support.

**Key Classes**:

```python
@dataclass
class LLMConfig:
    provider: str = "auto"
    anthropic_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
```

**Key Function**:
```python
def generate_answer(
    query: str,
    contexts: list[str],
    cfg: LLMConfig,
    history: list[dict]
) -> tuple[str, str]:
    """Generate answer to query, grounded in contexts.
    
    Returns:
        tuple: (answer_text, provider_used)
    """
```

**Provider Priority (auto mode)**:
1. Anthropic (Claude) - requires `ANTHROPIC_API_KEY`
2. OpenAI (GPT) - requires `OPENAI_API_KEY`
3. Ollama (local) - requires running Ollama server
4. Extractive fallback - always works (concatenates top chunks)

**System Prompt**:
```
You are a precise, helpful assistant. Use ONLY the provided context to answer. 
Cite sources inline using bracket notation like [1], [2] matching the context 
numbers. If the context does not contain the answer, say so plainly. 
Be concise and well-structured (use short paragraphs or bullet lists).
```

### NLP Engine Module (`core/nlp_engine.py`)

Query analysis and understanding.

**Key Classes**:

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

**Intent Classes**:
| Intent | Trigger Words |
|--------|---------------|
| definition | what is, define, explain, describe |
| compare | compare, vs, difference between |
| summarize | summarize, key points, tldr |
| calculate | calculate, compute, how much |

**Key Methods**:
- `train_intent(X, y)`: Train intent classifier
- `predict_intent(text)`: Predict query intent
- `analyze_sentiment(text)`: Analyze sentiment (VADER)
- `extract_entities(text)`: Named entity recognition (spaCy)
- `extract_keywords(text, top_k)`: Keyword extraction (TF-IDF)
- `analyze(text)`: Full NLP analysis

### CV Engine Module (`core/cv_engine.py`)

OCR for images and scanned documents.

**Key Classes**:

```python
@dataclass
class OCRResult:
    text: str
    confidence: float
    page: int = 0
    word_count: int = 0

@dataclass
class CVConfig:
    deskew: bool = True
    denoise: bool = True
    adaptive_threshold: bool = True
    min_confidence: float = 30.0
    tesseract_lang: str = "eng"
    dpi: int = 300
```

**Supported Formats**: PNG, JPG, JPEG, TIF, TIFF, BMP, PDF

**Key Methods**:
- `preprocess_image(img)`: Apply image preprocessing
- `ocr_image(path)`: OCR single image
- `ocr_pdf(path)`: OCR PDF (page by page)
- `file_to_text(path)`: Auto-route based on extension

### Evaluation Module (`core/evaluation.py`)

Comprehensive metrics for RAG pipeline evaluation.

**Retrieval Metrics**:
- `precision_at_k(retrieved, relevant, k)`: Precision at k
- `recall_at_k(retrieved, relevant, k)`: Recall at k
- `reciprocal_rank(retrieved, relevant)`: Mean Reciprocal Rank
- `ndcg_at_k(retrieved, relevant, k)`: Normalized DCG

**Generation Metrics**:
- `rouge_l(prediction, reference)`: ROUGE-L F1 score
- `bleu_n(prediction, reference, n)`: BLEU score

**Semantic Metrics**:
- `semantic_similarity(prediction, reference)`: Cosine similarity
- `bertscore_f1(predictions, references)`: BERTScore F1

**Faithfulness Metrics**:
- `lexical_faithfulness(answer, contexts)`: Token overlap
- `semantic_faithfulness(answer, contexts)`: Max semantic similarity

**Key Classes**:
```python
@dataclass
class QueryResult:
    query: str
    answer: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    expected_answer: str
    contexts: list[str]
    latency_ms: float
    precision: float
    recall: float
    mrr: float
    ndcg: float
    rouge_l: float
    bleu_4: float
    sem_sim: float
    faithfulness: float

@dataclass
class EvalReport:
    run_id: str
    n_queries: int
    per_query: list[QueryResult]
    aggregate: dict
```

## 4.3 Configuration Reference

### Environment Variables

| Variable | Provider | Required For |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic | Claude API access |
| `OPENAI_API_KEY` | OpenAI | GPT API access |
| `OLLAMA_HOST` | Ollama | Local LLM (default: http://localhost:11434) |

### Configuration Classes

**PreprocessConfig**:
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

**RAGConfig**:
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

**LLMConfig**:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| provider | str | auto | LLM provider |
| anthropic_model | str | claude-haiku-4-5 | Claude model |
| openai_model | str | gpt-4o-mini | GPT model |
| ollama_model | str | llama3.1:8b | Ollama model |
| ollama_host | str | http://localhost:11434 | Ollama endpoint |
| temperature | float | 0.2 | Generation temperature |
| max_tokens | int | 1024 | Max tokens |

## 4.4 Extending the Application

### Adding New LLM Providers

To add a new LLM provider:

1. Update `LLMConfig` to include new provider option
2. Modify `generate_answer()` function to handle new provider
3. Add API key validation for the new provider

Example:
```python
def generate_answer(query, contexts, cfg, history):
    if cfg.provider == "new_provider":
        # Add implementation
        return new_provider_call(query, contexts), "new_provider"
```

### Adding New Data Formats

To support additional file formats:

1. Update `load_documents()` in `preprocessing.py`
2. Add format detection and parsing logic
3. Handle metadata extraction as needed

Example:
```python
def load_documents(path):
    if path.suffix == '.newformat':
        return load_new_format(path)
    # Existing loaders...
```

### Custom Embedding Models

To use a custom embedding model:

1. Ensure model is available in sentence-transformers or compatible
2. Update `RAGConfig.embed_model` with model name
3. Rebuild the index with new model

```python
cfg = RAGConfig(embed_model="your/custom:model")
rag_engine.build_index("my_index", chunks, config=cfg)
```

### Custom Intent Classes

To add custom intents:

1. Prepare training data with intent labels
2. Call `nlp_engine.train_intent(X, y)` with labeled data
3. Model will be saved to `models/` directory

```python
# Prepare training data
X = ["how does X work", "explain Y", ...]
y = ["explain", "explain", ...]

# Train
report = nlp_engine.train_intent(X, y)
print(f"Accuracy: {report['accuracy']}")
```

---

# 5. API Reference

## 5.1 Preprocessing Module

### Classes

#### `Chunk`

```python
@dataclass
class Chunk:
    chunk_id: str           # Unique identifier (format: "source::record_idx::chunk_idx")
    text: str              # Cleaned text used for retrieval
    raw_text: str          # Original text (for display/citations)
    source: str            # Original file name
    metadata: dict         # Additional metadata
```

#### `PreprocessConfig`

```python
@dataclass
class PreprocessConfig:
    lowercase: bool = True           # Convert to lowercase + NFKC normalize
    strip_punctuation: bool = True # Remove punctuation
    remove_stopwords: bool = False # Remove English stopwords
    lemmatize: bool = True          # Use spaCy for lemmatization
    dedupe: bool = True             # Deduplicate chunks by hash
    chunk_size: int = 256           # Words per chunk
    chunk_overlap: int = 32         # Overlap between chunks
    min_chunk_words: int = 8        # Minimum words per chunk
```

### Functions

#### `load_documents(path: str | Path) -> list[dict]`

Load a single file or directory of files.

**Parameters**:
- `path`: File or directory path

**Returns**: List of dicts with `{source, text, metadata}`

**Supported Formats**: JSON, CSV, TXT, MD

#### `clean_text(text: str, cfg: PreprocessConfig) -> str`

Apply Unicode NFKC normalization, lowercase, and punctuation stripping.

#### `tokenize(text: str, cfg: PreprocessConfig) -> list[str]`

Tokenize and optionally lemmatize text, optionally remove stopwords.

#### `normalize_for_index(text: str, cfg: PreprocessConfig) -> str`

Cleaned + tokenized text reassembled as a single string for embedding.

#### `chunk_words(text: str, chunk_size: int, overlap: int, min_words: int) -> list[str]`

Split text into overlapping word-token chunks.

**Parameters**:
- `text`: Input text
- `chunk_size`: Words per chunk (default: 256)
- `overlap`: Overlap in words (default: 32)
- `min_words`: Minimum words to create a chunk (default: 8)

**Returns**: List of chunk strings

#### `preprocess(docs: list[dict], cfg: PreprocessConfig, progress=None) -> list[Chunk]`

Run the full preprocessing pipeline.

**Parameters**:
- `docs`: List of documents from load_documents
- `cfg`: Preprocessing configuration
- `progress`: Optional callback function(stage, pct, msg)

**Returns**: List of Chunk objects

## 5.2 RAG Engine Module

### Classes

#### `RAGConfig`

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

#### `RetrievalResult`

```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None
```

### RAGEngine Methods

#### `build_index(name: str, chunks: list[Chunk], progress=None) -> dict`

Embed and index chunks under a given name.

**Parameters**:
- `name`: Index name
- `chunks`: List of Chunk objects
- `progress`: Optional callback function

**Returns**: Dict with `name`, `chunks`, `dim`, `embed_model`, `has_bm25`

#### `load_index(name: str) -> Optional[dict]`

Load a previously-built index from disk.

**Parameters**:
- `name`: Index name

**Returns**: Index dict or None if not found

#### `list_indexes() -> list[dict]`

List all existing indexes with metadata.

**Returns**: List of index metadata dicts

#### `delete_index(name: str)`

Delete an index and its files.

**Parameters**:
- `name`: Index name

#### `retrieve(query: str, name: str, top_k: int) -> tuple[list[RetrievalResult], dict]`

Run full retrieval pipeline.

**Parameters**:
- `query`: Search query
- `name`: Index name
- `top_k`: Number of results

**Returns**: (results, timings) where timings includes embed_ms, dense_ms, bm25_ms, fusion_ms, rerank_ms, total_ms

## 5.3 LLM Module

### Classes

#### `LLMConfig`

```python
@dataclass
class LLMConfig:
    provider: str = "auto"
    anthropic_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024
```

### Functions

#### `generate_answer(query: str, contexts: list[str], cfg: LLMConfig, history: list[dict]) -> tuple[str, str]`

Generate answer to query, grounded in contexts.

**Parameters**:
- `query`: User question
- `contexts`: Retrieved context strings
- `cfg`: LLM configuration
- `history`: Chat history for context

**Returns**: (answer_text, provider_used)

## 5.4 NLP Engine Module

### Classes

#### `NLPAnalysis`

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

### NLPEngine Methods

#### `train_intent(X, y) -> dict`

Train intent classifier using TF-IDF + Logistic Regression.

**Parameters**:
- X: List of query strings
- y: List of intent labels

**Returns**: Report with accuracy, macro_f1, labels, confusion_matrix

#### `predict_intent(text: str) -> tuple[str, float]`

Predict intent and confidence.

**Parameters**:
- text: Query string

**Returns**: (intent_label, confidence)

#### `analyze_sentiment(text: str) -> tuple[str, float]`

Analyze sentiment using VADER.

**Parameters**:
- text: Input text

**Returns**: (sentiment, score)

#### `extract_entities(text: str) -> list[dict]`

Extract named entities using spaCy.

**Parameters**:
- text: Input text

**Returns**: List of {text, label}

#### `extract_keywords(text: str, top_k: int) -> list[str]`

Extract top keywords using TF-IDF.

**Parameters**:
- text: Input text
- top_k: Number of keywords

**Returns**: List of keyword strings

#### `analyze(text: str) -> NLPAnalysis`

Run full NLP analysis.

**Parameters**:
- text: Input text

**Returns**: NLPAnalysis object with all fields

## 5.5 CV Engine Module

### Classes

#### `OCRResult`

```python
@dataclass
class OCRResult:
    text: str
    confidence: float
    page: int = 0
    word_count: int = 0
```

#### `CVConfig`

```python
@dataclass
class CVConfig:
    deskew: bool = True
    denoise: bool = True
    adaptive_threshold: bool = True
    min_confidence: float = 30.0
    tesseract_lang: str = "eng"
    dpi: int = 300
```

### CVEngine Methods

#### `preprocess_image(img: np.ndarray) -> np.ndarray`

Apply preprocessing (grayscale, denoise, threshold, deskew).

**Parameters**:
- img: Input image as numpy array

**Returns**: Preprocessed image

#### `ocr_image(image_path) -> OCRResult`

OCR a single image file.

**Parameters**:
- image_path: Path to image file

**Returns**: OCRResult object

#### `ocr_pdf(pdf_path) -> list[OCRResult]`

OCR each page of a scanned PDF.

**Parameters**:
- pdf_path: Path to PDF file

**Returns**: List of OCRResult objects (one per page)

#### `file_to_text(path) -> str`

Auto-route to appropriate loader based on extension.

**Parameters**:
- path: Path to file

**Returns**: Extracted text

## 5.6 Evaluation Module

### Functions

#### Retrieval Metrics

```python
precision_at_k(retrieved: list, relevant: list, k: int) -> float
recall_at_k(retrieved: list, relevant: list, k: int) -> float
reciprocal_rank(retrieved: list, relevant: list) -> float
ndcg_at_k(retrieved: list, relevant: list, k: int) -> float
```

#### Generation Metrics

```python
rouge_l(prediction: str, reference: str) -> float
bleu_n(prediction: str, reference: str, n: int = 4) -> float
```

#### Semantic Metrics

```python
semantic_similarity(text1: str, text2: str) -> float
bertscore_f1(predictions: list, references: list) -> list[float]
```

#### Faithfulness Metrics

```python
lexical_faithfulness(answer: str, contexts: list[str]) -> float
semantic_faithfulness(answer: str, contexts: list[str]) -> float
```

### Classes

#### `QueryResult`

```python
@dataclass
class QueryResult:
    query: str
    answer: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    expected_answer: str
    contexts: list[str]
    latency_ms: float
    precision: float
    recall: float
    mrr: float
    ndcg: float
    rouge_l: float
    bleu_4: float
    sem_sim: float
    faithfulness: float
```

#### `EvalReport`

```python
@dataclass
class EvalReport:
    run_id: str
    n_queries: int
    per_query: list[QueryResult]
    aggregate: dict
```

#### `run_evaluation(gold_set, answer_fn, run_id, k, progress) -> EvalReport`

Run full evaluation suite.

**Parameters**:
- gold_set: List of {query, expected_chunk_ids, expected_answer, dataset}
- answer_fn: Function(query, dataset) -> (answer, retrieved_ids, contexts, latency_ms)
- run_id: Evaluation run identifier
- k: k value for retrieval metrics
- progress: Optional callback function

**Returns**: EvalReport object

---

# 6. Datasets

## 6.1 Pre-built Datasets

The project includes 11 pre-built datasets in the `datasets/` folder:

| Dataset | Domain | Description |
|---------|--------|-------------|
| AI_Machine_Learning_2024.json | AI/ML | Artificial intelligence and machine learning topics |
| Climate_Environment_2024.json | Environment | Climate change and environmental science |
| Cybersecurity_Privacy_2024.json | Security | Cybersecurity and privacy topics |
| Drug_Medication_Reference_2024.json | Healthcare | Drug and medication reference |
| Financial_Literacy_2024.json | Finance | Financial literacy and concepts |
| Legal_FAQ_2024.json | Law | Legal frequently asked questions |
| Medical_Disease_Dataset_2024.json | Healthcare | Medical diseases and conditions |
| Mental_Health_Psychology_2024.json | Health | Mental health and psychology |
| Nutrition_Diet_Science_2024.json | Health | Nutrition and diet science |
| Sports_Medicine_Fitness_2024.json | Health | Sports medicine and fitness |
| World_History_2024.json | History | World history events |

## 6.2 Dataset Structure

### JSON Format (Recommended)

```json
{
  "disease": "Example Disease",
  "symptoms": ["symptom1", "symptom2"],
  "causes": "Cause description",
  "risk_factors": ["factor1", "factor2"],
  "treatments": ["treatment1", "treatment2"],
  "prevention": "Prevention method",
  "when_to_see_doctor": "Warning signs",
  "icd10": "ICD-10 code",
  "category": "Disease category"
}
```

### CSV Format

```
column1,column2,column3
value1,value2,value3
```

### TXT/MD Format

```
First paragraph of content.

Second paragraph of content.

Third paragraph of content.
```

## 6.3 Creating Custom Datasets

1. Prepare your data in one of the supported formats
2. Upload via Developer Mode > Datasets tab
3. Configure preprocessing settings
4. Index the dataset
5. Begin querying

---

# 7. Troubleshooting

## 7.1 Common Issues

### No Embeddings Generated

**Symptom**: Index build fails or returns no chunks

**Solutions**:
- Ensure text has at least `min_chunk_words` (default: 8) words per chunk
- Check that preprocessing settings aren't too aggressive
- Verify the input file contains valid text

### LLM API Errors

**Symptom**: "API key not found" or connection errors

**Solutions**:
- Set environment variable for the provider
- Use extractive fallback mode (provider: "extractive")
- Check API key is valid and has sufficient credits

### OCR Not Working

**Symptom**: Images/PDFs not extracting text

**Solutions**:
- Ensure Tesseract is installed: `tesseract --version`
- Ensure Poppler is installed (for PDFs)
- Check image quality - low resolution images may fail
- Try adjusting CVConfig settings (lower min_confidence)

### Slow Performance

**Symptom**: Queries take too long

**Solutions**:
- Use smaller embedding model (all-MiniLM-L6-v2)
- Reduce top_k values
- Disable reranker if not needed
- Use extractive fallback for instant responses
- Ensure sufficient RAM (16GB recommended)

### spaCy Not Available

**Symptom**: NLP features fail with "spaCy not available"

**Solutions**:
- Run: `python -m spacy download en_core_web_sm`
- Falls back to regex tokenization (reduced functionality)

### Index Not Found

**Symptom**: "Index not found" error

**Solutions**:
- Ensure index was created successfully
- Check index name matches exactly (case-sensitive)
- Verify index files exist in `indexes/` directory

## 7.2 Performance Tuning

| Component | Setting | Impact |
|-----------|---------|--------|
| Embedding | all-MiniLM-L6-v2 | Faster, lower memory |
| Embedding | bge-large-en-v1.5 | Slower, higher quality |
| Reranker | Disabled | Faster retrieval |
| BM25 | Disabled | Skip sparse retrieval |
| Top-k | Lower values | Faster, fewer results |
| LLM | Extractive | Instant responses |

---

# 8. Appendix

## 8.1 Algorithm Summary

| Stage | Family | Algorithm | Library |
|-------|--------|-----------|---------|
| OCR | CV | Tesseract + OpenCV | pytesseract, opencv-python |
| Text cleaning | NLP | regex + NFKC normalize | stdlib, spaCy |
| Lemmatization | NLP | spaCy en_core_web_sm | spaCy |
| Chunking | NLP | Word-window with overlap | stdlib |
| Embedding | DL | MiniLM-L6-v2 (Transformer) | sentence-transformers |
| Sparse retrieval | ML | BM25 (Okapi) | rank-bm25 |
| Dense index | ML | FAISS HNSW (M=32, ef=200) | faiss-cpu |
| Hybrid fusion | ML | RRF (k=60) | custom |
| Reranker | DL | ms-marco-MiniLM-L-6-v2 | sentence-transformers |
| Intent classifier | ML | TF-IDF + Logistic Regression | scikit-learn |
| Generation | DL | Claude / GPT / Llama / Extractive | anthropic, openai, ollama |
| Sentiment | NLP | VADER (rule-based) | vaderSentiment |
| Eval: Retrieval | - | Recall@k, MRR, nDCG@k | custom |
| Eval: Generation | - | ROUGE-L, BLEU-4 | rouge-score |
| Eval: Semantic | - | BERTScore + sentence-cosine | bert-score |
| Eval: Faithfulness | - | Semantic overlap answer↔contexts | custom |

## 8.2 Fallback Behavior

| Component | Fallback |
|-----------|----------|
| Sentence-transformers | Hashing fallback (384-dim) |
| spaCy model | Blank spaCy model or regex tokenization |
| LLM providers | Extractive fallback (concatenates top 3 contexts) |
| FAISS | Numpy cosine similarity fallback |
| BM25 | Disabled if not available |

## 8.3 Index Persistence

Indexes are saved with the following structure:
```
indexes/<index_name>/
├── chunks.json        # Chunk data
├── embeddings.npy     # NumPy embeddings
├── faiss.index        # FAISS index (if available)
├── bm25.pkl           # BM25 model (if available)
└── meta.json          # Metadata (embed_model, n_chunks, dim)
```

## 8.4 Requirements

```
streamlit>=1.30,<2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
sentence-transformers>=2.5
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
spacy>=3.7
vaderSentiment>=3.3.2
opencv-python>=4.9
pytesseract>=0.3.10
Pillow>=10.0
pdf2image>=1.16
rouge-score>=0.1.2
anthropic>=0.39
openai>=1.40
```

## 8.5 System Dependencies

For full OCR functionality:
- **Tesseract OCR**: `apt install tesseract-ocr` (Linux) or installer (Windows)
- **Poppler**: `apt install poppler-utils` (Linux) or installer (Windows)

---

