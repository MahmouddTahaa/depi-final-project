# ChatBoot RAG

ChatBoot is a Streamlit chatbot application that lets users chat with indexed
datasets using a Retrieval-Augmented Generation (RAG) pipeline.

The app has two main modes:

- User mode: simple chat interface for asking questions about the selected dataset.
- Developer mode: dataset upload, preprocessing, model settings, pipeline view,
  evaluation, and logs.

## Project Structure

```text
chatboot_py/
  streamlit_app.py          Main Streamlit GUI
  requirements.txt          Python dependencies
  setup_once.bat            One-time dependency setup for Windows
  start_app.bat             Start the app without reinstalling packages
  run.bat                   Alternative Windows launcher
  run.sh                    Linux/macOS launcher
  data/                     Default data files
  datasets/                 Local datasets available from Developer mode
  indexes/                  Generated RAG indexes
  models/                   Saved ML models
  eval_runs/                Saved evaluation reports
  core/
    preprocessing.py        Load, clean, tokenize, chunk, and deduplicate data
    rag_engine.py           Embeddings, vector search, BM25, RRF, reranking
    llm.py                  LLM providers and extractive fallback
    nlp_engine.py           Intent, sentiment, entities, keywords
    cv_engine.py            OCR for images and PDFs
    evaluation.py           Retrieval and answer-quality metrics
```

## How To Run

### Recommended daily start

After setup is complete, run:

```bat
start_app.bat
```

Or run manually:

```bat
python -m streamlit run streamlit_app.py
```

The app opens at:

```text
http://localhost:8501
```

### One-time setup only

Use this only if dependencies are missing:

```bat
setup_once.bat
```

Do not install requirements every time. Reinstalling repeatedly wastes disk
space. The setup script uses `--no-cache-dir` to reduce storage usage.

## How To Use The App

### Developer Mode

Use Developer mode first to prepare a dataset.

1. Open the sidebar.
2. Choose `Developer`.
3. Go to the `Datasets` tab.
4. Either upload a dataset file or choose a file from the local `datasets/` folder.
5. Click `Preprocess & index` or `Index selected local dataset`.
6. Wait until the dataset is indexed.

Supported uploaded file types:

- JSON
- CSV
- TXT
- MD
- PNG/JPG/JPEG images
- PDF files

For images and PDFs, the app applies OCR first, then preprocessing and indexing.

### User Mode

After a dataset is indexed:

1. Choose `User` mode in the sidebar.
2. Select the active dataset.
3. Click `Start chat`.
4. Ask questions in the chat box.
5. Open the sources expander to see retrieved evidence chunks.

## Preprocessing Pipeline

When a dataset is uploaded or selected, the app runs:

```text
Load file
-> clean text
-> tokenize / lemmatize
-> split into chunks
-> deduplicate chunks
-> create embeddings
-> build retrieval index
```

Preprocessing settings are available in:

```text
Developer -> Preprocessing
```

You can control:

- Lowercase and Unicode normalization
- Punctuation stripping
- Stopword removal
- Lemmatization
- Deduplication
- Chunk size
- Chunk overlap
- Minimum words per chunk

If preprocessing creates zero chunks, the app retries with a lower minimum word
count so small records can still be indexed.

## RAG Pipeline

The RAG flow is:

```text
User question
-> query embedding
-> dense retrieval
-> optional BM25 keyword retrieval
-> Reciprocal Rank Fusion
-> optional reranking
-> answer generation from retrieved context
```

The implementation is in:

```text
core/rag_engine.py
core/llm.py
```

The app supports:

- Sentence-transformer embeddings when installed
- FAISS vector index when installed
- BM25 keyword search when installed
- Cross-encoder reranker when enabled
- Hashing fallback embeddings if sentence-transformers is missing
- Extractive fallback answers if no LLM provider is configured

## LLM Providers

The answer generator can use:

- Anthropic
- OpenAI
- Ollama
- Extractive fallback

If no API key is configured, the app still works using extractive fallback. The
answer quality is simpler, but the chat remains usable.

Optional environment variables:

```bat
set ANTHROPIC_API_KEY=your_key_here
set OPENAI_API_KEY=your_key_here
```

## Evaluation

Evaluation is available in:

```text
Developer -> Evaluation
```

It can calculate:

- Precision@k
- Recall@k
- MRR
- nDCG@k
- ROUGE-L
- BLEU-4
- Semantic similarity
- Faithfulness
- Average latency

You can upload a gold-set JSON file or auto-sample from the active index.

## Computer Vision And OCR

For image and scanned PDF uploads, the app uses:

- OpenCV preprocessing
- Tesseract OCR
- pdf2image for scanned PDFs

Extra system tools may be needed:

- Tesseract OCR
- Poppler for scanned PDFs

Text datasets do not need these tools.

## Troubleshooting

### `streamlit` is not recognized

Run:

```bat
python -m streamlit run streamlit_app.py
```

### `No module named streamlit`

Run setup once:

```bat
setup_once.bat
```

### SpaCy model error

The app can still run without the full spaCy model, but for best NLP support run:

```bat
python -m spacy download en_core_web_sm
```

### Dataset upload creates no chunks

Use `Developer -> Preprocessing` and lower `Min words per chunk`, or use the
current app version, which retries automatically with a smaller value.

### C drive storage is low

Do not reinstall dependencies repeatedly. You can clear pip cache with:

```bat
python -m pip cache purge
```

## Main Files

- `streamlit_app.py`: GUI and app routing
- `core/preprocessing.py`: dataset loading and preprocessing
- `core/rag_engine.py`: retrieval and indexing
- `core/llm.py`: answer generation
- `core/nlp_engine.py`: query analysis
- `core/cv_engine.py`: OCR
- `core/evaluation.py`: metrics

## Notes

- Indexes are saved under `indexes/`.
- Uploaded files are copied into `data/`.
- Local ready-to-index files can be placed in `datasets/`.
- Use `start_app.bat` for normal running.
- Use `setup_once.bat` only when dependencies are missing.
