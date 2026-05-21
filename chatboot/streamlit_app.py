"""
streamlit_app.py — ChatBoot RAG with Streamlit GUI
====================================================
Two views in one app:
  • USER MODE      — clean chat: ask questions, get cited answers
  • DEVELOPER MODE — dataset upload, preprocessing config, model config,
                     pipeline view, evaluation metrics, logs

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations
import os
import sys
import json
import time
import io
import html
import re
from dataclasses import replace
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Make `core` importable
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    Chunk,
    PreprocessConfig,
    load_documents,
    preprocess,
    RAGEngine,
    RAGConfig,
    NLPEngine,
    CVEngine,
    CVConfig,
    generate_answer,
    LLMConfig,
    run_evaluation,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    rouge_l,
    bleu_n,
    semantic_faithfulness,
)

st.set_page_config(
    page_title="ChatBoot RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark theme CSS (matches the HTML mockup)
st.markdown(
    """
<style>
  :root {
    --accent: #4a6cf7;
    --bg-panel: #10121a;
    --border: #1f2435;
    --text-main: #e8eaf2;
    --text-sub: #5a6280;
  }
  .stApp { background: #0d0f17; color: var(--text-main); }
  .stSidebar { background: var(--bg-panel) !important; border-right: 1px solid var(--border); }
  .stMarkdown, .stText { color: var(--text-main); }
  .stChatMessage { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; }
  .stTextInput input, .stTextArea textarea { background: #1a1e2e !important; color: var(--text-main) !important; border-color: var(--border) !important; }
  .stButton button { background: var(--accent); color: white; border: none; }
  .stButton button:hover { background: #3a5cf0; }

  .cb-card { background: #10121a; border: 1px solid #1f2435; border-radius: 12px;
             padding: 12px 16px; margin-bottom: 10px; }
  .cb-pill { display: inline-block; padding: 2px 8px; border-radius: 5px;
             background: #1a1e2e; color: #b0b8d0; font-size: 11px;
             font-family: 'JetBrains Mono', monospace; margin: 2px; }
  .cb-source { padding: 8px 12px; background: #0f1120; border-left: 3px solid #4a6cf7;
               border-radius: 0 6px 6px 0; margin: 6px 0; font-size: 12px; color: #b0b8d0; }
  .cb-source b { color: #e8eaf2; }
  .cb-metric-big { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .cb-metric-label { font-size: 11px; color: #5a6280; text-transform: uppercase; letter-spacing: 1px; }
</style>
""",
    unsafe_allow_html=True,
)


ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "indexes"
DATA_DIR = ROOT / "data"
DATASETS_DIR = ROOT / "datasets"
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "eval_runs"

for p in [INDEX_DIR, DATA_DIR, DATASETS_DIR, MODELS_DIR, EVAL_DIR]:
    p.mkdir(exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> RAGEngine:
    return RAGEngine(persist_dir=INDEX_DIR)


@st.cache_resource(show_spinner=False)
def get_nlp_engine() -> NLPEngine:
    eng = NLPEngine(model_dir=MODELS_DIR)
    if eng._intent_clf is None:
        with st.spinner("Training intent classifier (first run only)…"):
            eng.train_intent()
    return eng


@st.cache_resource(show_spinner=False)
def get_cv_engine() -> CVEngine:
    return CVEngine()


def init_state():
    ss = st.session_state
    ss.setdefault("mode", "User")
    ss.setdefault("active_index", None)
    ss.setdefault("chat_history", {})  # {index_name: [{role, content, sources, nlp}]}
    ss.setdefault("chat_started", {})  # {index_name: bool}
    ss.setdefault("preproc_cfg", PreprocessConfig())
    ss.setdefault("rag_cfg", RAGConfig())
    ss.setdefault("llm_cfg", LLMConfig())
    ss.setdefault("logs", [])
    ss.setdefault("last_eval", None)
    ss.setdefault("show_sources", True)


def log(level: str, module: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.logs.append(
        {"ts": ts, "level": level, "module": module, "msg": msg}
    )
    # Cap log length
    if len(st.session_state.logs) > 500:
        st.session_state.logs = st.session_state.logs[-500:]


init_state()


def list_indexes_full() -> list[dict]:
    """List existing indexes with metadata."""
    return get_rag_engine().list_indexes()


def list_local_dataset_files() -> list[Path]:
    """Return bundled/local datasets that can be indexed without upload."""
    allowed = {".json", ".csv", ".txt", ".md"}
    files: dict[str, Path] = {}
    for folder in (DATASETS_DIR, DATA_DIR):
        if folder.exists():
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in allowed:
                    files.setdefault(path.name, path)
    return sorted(files.values(), key=lambda p: p.name.lower())


def safe_index_name(name: str) -> str:
    """Return a filesystem-safe index name while preserving readability."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._-") or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_uploaded_file(uploaded) -> Path:
    """Persist an uploaded file using only its basename."""
    save_path = DATA_DIR / Path(uploaded.name).name
    save_path.write_bytes(uploaded.getvalue())
    return save_path


def source_card(source: dict, limit: int = 300) -> str:
    text = source["text"]
    snippet = text[:limit] + ("..." if len(text) > limit else "")
    return (
        f"<div class='cb-source'><b>[{source['num']}]</b> "
        f"<span class='cb-pill'>sim {source['score']:.2f}</span> "
        f"<span class='cb-pill'>{html.escape(str(source['source']))}</span><br/>"
        f"{html.escape(snippet)}</div>"
    )


def index_dataset(
    name: str,
    file_path: Path | None = None,
    raw_text: str | None = None,
    file_name: str | None = None,
):
    """Run the full preprocessing + indexing pipeline on a dataset."""
    rag = get_rag_engine()
    cfg = st.session_state.preproc_cfg
    name = safe_index_name(name)

    prog = st.progress(0.0, text="Loading documents…")
    log("INFO", "preproc", f"Starting indexing for '{name}'")

    # Load
    if file_path:
        docs = load_documents(file_path)
    elif raw_text and file_name:
        # Synthetic doc from uploaded text
        save_path = DATA_DIR / Path(file_name).name
        save_path.write_text(raw_text, encoding="utf-8")
        docs = load_documents(save_path)
    else:
        st.error("No data provided")
        return None

    if not docs:
        st.error("No documents could be parsed from the upload.")
        return None

    log("INFO", "preproc", f"Loaded {len(docs)} documents")

    # Preprocess
    def pp_progress(stage, pct, msg):
        prog.progress(min(0.5, pct * 0.5), text=f"Preprocessing · {msg}")

    chunks = preprocess(docs, cfg, progress=pp_progress)
    if not chunks and cfg.min_chunk_words > 1:
        relaxed_cfg = replace(cfg, min_chunk_words=1)
        chunks = preprocess(docs, relaxed_cfg, progress=pp_progress)
        if chunks:
            log(
                "WARN",
                "preproc",
                "No chunks met the configured minimum word count; retried with min_chunk_words=1",
            )
    log(
        "INFO",
        "preproc",
        f"Produced {len(chunks)} chunks "
        f"(chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})",
    )
    if not chunks:
        prog.empty()
        st.error(
            "Preprocessing finished, but no indexable text chunks were created. Try a larger text file or lower the minimum words per chunk."
        )
        return None

    # Index
    def idx_progress(stage, pct, msg):
        prog.progress(0.5 + pct * 0.5, text=f"Indexing · {stage} · {msg}")

    try:
        rag.cfg = st.session_state.rag_cfg
        info = rag.build_index(name, chunks, progress=idx_progress)
    except Exception as exc:
        prog.empty()
        log("ERROR", "rag", f"Indexing failed for '{name}': {exc}")
        st.error(f"Indexing failed: {exc}")
        return None

    log("INFO", "rag", f"Index built: {info}")
    prog.empty()
    st.success(f"✓ Indexed **{name}** — {info['chunks']} chunks · {info['dim']}-d")
    return info


def answer_query(query: str, index_name: str) -> dict:
    """Run the full RAG pipeline for one query."""
    t0 = time.perf_counter()
    rag = get_rag_engine()
    rag.cfg = st.session_state.rag_cfg
    nlp = get_nlp_engine()

    log("INFO", "rag", f"Query: {query!r}")
    results, timings = rag.retrieve(query, index_name)
    log(
        "INFO", "rag", f"Retrieved {len(results)} chunks in {timings['total_ms']:.0f}ms"
    )

    contexts = [r.chunk.raw_text for r in results]
    chunk_ids = [r.chunk.chunk_id for r in results]

    # Build chat history for context
    hist = st.session_state.chat_history.get(index_name, [])
    history_msgs = [{"role": m["role"], "content": m["content"]} for m in hist[-4:]]

    t_gen = time.perf_counter()
    answer, provider = generate_answer(
        query, contexts, st.session_state.llm_cfg, history_msgs
    )
    gen_ms = (time.perf_counter() - t_gen) * 1000
    log("INFO", "llm", f"Generated via {provider} in {gen_ms:.0f}ms")

    nlp_data = nlp.analyze(query)
    faith = semantic_faithfulness(answer, contexts)

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "answer": answer,
        "provider": provider,
        "results": results,
        "contexts": contexts,
        "chunk_ids": chunk_ids,
        "timings": {**timings, "gen_ms": gen_ms, "total_ms": total_ms},
        "nlp": nlp_data,
        "faithfulness": faith,
    }


with st.sidebar:
    st.markdown("### 🤖 ChatBoot")
    st.caption("RAG · ML · DL · CV")

    st.session_state.mode = st.radio(
        "Mode",
        ["User", "Developer"],
        index=0 if st.session_state.mode == "User" else 1,
        horizontal=True,
    )

    st.markdown("---")
    st.markdown("#### 📚 Active dataset")
    indexes = list_indexes_full()
    if indexes:
        names = [i["name"] for i in indexes]
        default_idx = (
            names.index(st.session_state.active_index)
            if st.session_state.active_index in names
            else 0
        )
        chosen = st.selectbox(
            "Choose", names, index=default_idx, label_visibility="collapsed"
        )
        st.session_state.active_index = chosen
        info = next(i for i in indexes if i["name"] == chosen)
        st.caption(f"{info['n_chunks']} chunks · {info['dim']}-d")
    else:
        st.info("No indexes yet — upload a dataset from Developer mode.")

    st.markdown("---")
    if st.session_state.mode == "User":
        st.markdown("#### Settings")
        st.session_state.show_sources = st.toggle(
            "Show sources", value=st.session_state.show_sources
        )
        if st.button("🗑 Clear chat", use_container_width=True):
            if st.session_state.active_index:
                st.session_state.chat_history[st.session_state.active_index] = []
                st.session_state.chat_started[st.session_state.active_index] = False
                st.rerun()
    else:
        st.caption(f"Backend status: ✅ ready")
        st.caption(f"LLM provider order: {st.session_state.llm_cfg.provider}")

    # Show API key hints
    with st.expander("🔑 LLM provider keys"):
        st.caption("Set environment variables before launch:")
        st.code("ANTHROPIC_API_KEY=...\nOPENAI_API_KEY=...", language="bash")
        st.caption("Or rely on Ollama / extractive fallback.")
        st.write(f"Anthropic: {'✅' if os.getenv('ANTHROPIC_API_KEY') else '—'}")
        st.write(f"OpenAI:    {'✅' if os.getenv('OPENAI_API_KEY')    else '—'}")


# ---------
# USER MODE
# ---------


def user_mode():
    if not st.session_state.active_index:
        st.title("Welcome to ChatBoot")
        st.write(
            "No knowledge base is loaded yet. Switch to **Developer** mode in the sidebar "
            "to upload a dataset."
        )
        return

    name = st.session_state.active_index
    st.title(f"Ask the assistant")
    st.caption(f"Grounded in **{name}** · answers include source citations")

    # Initialize chat history for this index
    if name not in st.session_state.chat_history:
        st.session_state.chat_history[name] = []
    history = st.session_state.chat_history[name]

    if not history and not st.session_state.chat_started.get(name, False):
        st.markdown("##### Ready to begin")
        st.write(
            "Start a new chat with the selected dataset, then ask questions grounded in its indexed sources."
        )
        if st.button("Start chat", type="primary", use_container_width=False):
            st.session_state.chat_started[name] = True
            st.rerun()
        return

    # Render past messages
    for m in history:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if (
                m["role"] == "assistant"
                and m.get("sources")
                and st.session_state.show_sources
            ):
                with st.expander(f"📚 {len(m['sources'])} sources"):
                    for s in m["sources"]:
                        st.markdown(source_card(s), unsafe_allow_html=True)

    # Welcome suggestions if no history
    if not history:
        st.markdown("##### Try asking:")
        cols = st.columns(2)
        suggestions = [
            "Give me an overview of this dataset",
            "What are the key topics covered?",
            "List the main entries",
            "Summarize the most important points",
        ]
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, use_container_width=True, key=f"sugg_{i}"):
                st.session_state._pending_q = s
                st.rerun()

    # Composer
    query = st.chat_input("Ask anything about the indexed documents…")
    if not query and st.session_state.get("_pending_q"):
        query = st.session_state.pop("_pending_q")

    if query:
        history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating…"):
                out = answer_query(query, name)
            st.write(out["answer"])
            sources = [
                {
                    "num": i + 1,
                    "source": r.chunk.source,
                    "score": r.score,
                    "text": r.chunk.raw_text,
                    "chunk_id": r.chunk.chunk_id,
                }
                for i, r in enumerate(out["results"])
            ]
            if st.session_state.show_sources and sources:
                with st.expander(f"📚 {len(sources)} sources"):
                    for s in sources:
                        st.markdown(source_card(s), unsafe_allow_html=True)
        history.append(
            {
                "role": "assistant",
                "content": out["answer"],
                "sources": sources,
                "nlp": out["nlp"].to_dict(),
                "timings": out["timings"],
                "faithfulness": out["faithfulness"],
                "provider": out["provider"],
            }
        )


# --------------
# DEVELOPER MODE
# --------------


def developer_mode():
    st.title("Developer Console")

    tabs = st.tabs(
        [
            "💬 Test Chat",
            "📁 Datasets",
            "⚙️ Preprocessing",
            "🧠 Models",
            "🔗 Pipeline",
            "📊 Evaluation",
            "📜 Logs",
        ]
    )

    with tabs[0]:
        dev_test_chat()
    with tabs[1]:
        dev_datasets()
    with tabs[2]:
        dev_preprocessing()
    with tabs[3]:
        dev_models()
    with tabs[4]:
        dev_pipeline_view()
    with tabs[5]:
        dev_evaluation()
    with tabs[6]:
        dev_logs()


def dev_test_chat():
    """Same chat UI as user mode but with retrieval trace & NLP shown inline."""
    if not st.session_state.active_index:
        st.warning("Upload + index a dataset first (Datasets tab).")
        return

    name = st.session_state.active_index
    st.caption(f"Active: **{name}** · all timings and intermediate steps shown.")

    col_chat, col_trace = st.columns([2, 1])

    with col_chat:
        if name not in st.session_state.chat_history:
            st.session_state.chat_history[name] = []
        history = st.session_state.chat_history[name]

        # Render last 6 messages
        for m in history[-6:]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        query = st.chat_input("Test query…", key="dev_chat_input")
        if query:
            history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Running RAG…"):
                    out = answer_query(query, name)
                st.write(out["answer"])
            history.append(
                {
                    "role": "assistant",
                    "content": out["answer"],
                    "sources": [
                        {
                            "num": i + 1,
                            "source": r.chunk.source,
                            "score": r.score,
                            "text": r.chunk.raw_text,
                            "chunk_id": r.chunk.chunk_id,
                        }
                        for i, r in enumerate(out["results"])
                    ],
                    "nlp": out["nlp"].to_dict(),
                    "timings": out["timings"],
                    "faithfulness": out["faithfulness"],
                    "provider": out["provider"],
                }
            )
            st.rerun()

    with col_trace:
        last = next((m for m in reversed(history) if m["role"] == "assistant"), None)
        if not last:
            st.info("Ask a question to see retrieval traces.")
            return

        st.markdown("#### Retrieved chunks")
        for s in last.get("sources", [])[:5]:
            st.markdown(source_card(s, limit=200), unsafe_allow_html=True)

        st.markdown("#### Latency")
        t = last.get("timings", {})
        st.code(
            json.dumps({k: round(v, 1) for k, v in t.items()}, indent=2),
            language="json",
        )

        st.markdown("#### NLP")
        nlp = last.get("nlp", {})
        st.markdown(
            f"**Intent:** `{nlp.get('intent')}` ({nlp.get('intent_confidence', 0):.2f})"
        )
        st.markdown(
            f"**Sentiment:** {nlp.get('sentiment')} ({nlp.get('sentiment_score', 0):.2f})"
        )
        if nlp.get("entities"):
            st.markdown(
                "**Entities:** "
                + " ".join(
                    f"<span class='cb-pill'><b>{html.escape(str(e['label']))}</b> {html.escape(str(e['text']))}</span>"
                    for e in nlp["entities"]
                ),
                unsafe_allow_html=True,
            )
        if nlp.get("keywords"):
            st.markdown(
                "**Keywords:** "
                + " ".join(
                    f"<span class='cb-pill'>{html.escape(str(k))}</span>"
                    for k in nlp["keywords"]
                ),
                unsafe_allow_html=True,
            )

        st.markdown(f"#### Faithfulness: `{last.get('faithfulness', 0):.3f}`")
        st.caption(f"Generated via **{last.get('provider', '?')}**")


def dev_datasets():
    st.subheader("📁 Datasets")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload a JSON, CSV, TXT, MD, or image/PDF (CV-OCR applied automatically)",
            type=["json", "csv", "txt", "md", "png", "jpg", "jpeg", "pdf"],
            accept_multiple_files=False,
        )
        custom_name = st.text_input(
            "Index name", value="", placeholder="auto-from-filename"
        )

    if uploaded and st.button("📥 Preprocess & index"):
        suffix = Path(uploaded.name).suffix.lower()
        index_name = safe_index_name(custom_name.strip() or Path(uploaded.name).stem)
        indexed = False

        if suffix in (".png", ".jpg", ".jpeg", ".pdf"):
            # CV path: OCR first, then treat as TXT
            st.info("Running OCR (CV pipeline)…")
            save = save_uploaded_file(uploaded)
            cv = get_cv_engine()
            try:
                text = cv.file_to_text(save)
                log("INFO", "cv", f"OCR'd {uploaded.name}: {len(text)} chars")
                indexed = (
                    index_dataset(
                        index_name, raw_text=text, file_name=index_name + ".txt"
                    )
                    is not None
                )
            except Exception as e:
                log("ERROR", "cv", f"OCR failed for '{uploaded.name}': {e}")
                st.error(f"OCR failed: {e}")
        else:
            try:
                save = save_uploaded_file(uploaded)
                indexed = index_dataset(index_name, file_path=save) is not None
            except Exception as e:
                log("ERROR", "datasets", f"Upload failed for '{uploaded.name}': {e}")
                st.error(f"Upload failed: {e}")
        if indexed:
            st.session_state.active_index = index_name
            st.rerun()

    local_files = list_local_dataset_files()
    if local_files:
        st.markdown("---")
        st.markdown("#### Index from local datasets folder")
        selected = st.selectbox(
            "Choose a dataset file",
            local_files,
            format_func=lambda p: p.name,
        )
        local_name = st.text_input(
            "Local index name",
            value=safe_index_name(selected.stem),
            key="local_index_name",
        )
        if st.button("Index selected local dataset", use_container_width=True):
            index_name = safe_index_name(local_name or selected.stem)
            if index_dataset(index_name, file_path=selected) is not None:
                st.session_state.active_index = index_name
                st.rerun()

    st.markdown("---")
    st.markdown("#### Indexed datasets")
    indexes = list_indexes_full()
    if not indexes:
        st.info("Nothing indexed yet.")
        return

    for info in indexes:
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
            c1.markdown(f"**{info['name']}**")
            c1.caption(
                f"{info['n_chunks']} chunks · {info['dim']}-d · {info['embed_model']}"
            )
            if c2.button("Use", key=f"use_{info['name']}"):
                st.session_state.active_index = info["name"]
                st.rerun()
            if c3.button("Preview", key=f"prev_{info['name']}"):
                rag = get_rag_engine()
                idx = rag.load_index(info["name"])
                st.code(
                    "\n\n".join(
                        f"[{i+1}] {c.raw_text[:200]}…"
                        for i, c in enumerate(idx["chunks"][:3])
                    )
                )
            if c4.button("Re-index", key=f"reidx_{info['name']}"):
                st.info("Re-upload the source file to re-index.")
            if c5.button("Delete", key=f"del_{info['name']}"):
                get_rag_engine().delete_index(info["name"])
                if st.session_state.active_index == info["name"]:
                    st.session_state.active_index = None
                st.rerun()
            st.markdown("---")


def dev_preprocessing():
    st.subheader("⚙️ Preprocessing pipeline")
    st.caption("These settings run BEFORE every (re-)index.")

    cfg = st.session_state.preproc_cfg

    # Pipeline visualization
    st.markdown("#### Pipeline stages")
    stages = [
        "Load",
        "OCR (if image)",
        "Clean",
        "Tokenize / Lemmatize",
        "Chunk",
        "Embed",
        "Index (FAISS + BM25)",
    ]
    cols = st.columns(len(stages))
    for i, (col, s) in enumerate(zip(cols, stages)):
        col.markdown(
            f"<div class='cb-card' style='text-align:center'>"
            f"<div style='color:#4a6cf7;font-size:11px;font-weight:600'>{i+1}</div>"
            f"<div style='font-size:12px;font-weight:600'>{s}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Text cleaning (NLP)")
        cfg.lowercase = st.toggle("Lowercase & NFKC normalize", value=cfg.lowercase)
        cfg.strip_punctuation = st.toggle(
            "Strip punctuation", value=cfg.strip_punctuation
        )
        cfg.remove_stopwords = st.toggle("Remove stopwords", value=cfg.remove_stopwords)
        cfg.lemmatize = st.toggle("Tokenize & lemmatize (spaCy)", value=cfg.lemmatize)
        cfg.dedupe = st.toggle("Deduplicate chunks (hash)", value=cfg.dedupe)

    with col2:
        st.markdown("#### Chunking")
        cfg.chunk_size = st.slider(
            "Chunk size (tokens)", 64, 1024, cfg.chunk_size, step=32
        )
        cfg.chunk_overlap = st.slider(
            "Overlap (tokens)", 0, 128, cfg.chunk_overlap, step=8
        )
        cfg.min_chunk_words = st.slider(
            "Min words per chunk", 1, 50, cfg.min_chunk_words
        )

    st.markdown("---")
    st.markdown("#### Current configuration")
    st.json(
        {
            "lowercase": cfg.lowercase,
            "strip_punctuation": cfg.strip_punctuation,
            "remove_stopwords": cfg.remove_stopwords,
            "lemmatize": cfg.lemmatize,
            "dedupe": cfg.dedupe,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "min_chunk_words": cfg.min_chunk_words,
        }
    )


def dev_models():
    st.subheader("🧠 Models")
    rcfg = st.session_state.rag_cfg
    lcfg = st.session_state.llm_cfg

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Embedding (DL)")
        rcfg.embed_model = st.selectbox(
            "Sentence-transformer",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-large-en-v1.5",
                "intfloat/e5-large-v2",
            ],
            index=0,
        )
        st.markdown("#### Reranker (DL)")
        rcfg.use_reranker = st.toggle(
            "Use cross-encoder reranker", value=rcfg.use_reranker
        )
        rcfg.reranker_model = st.text_input(
            "Reranker model", value=rcfg.reranker_model, disabled=not rcfg.use_reranker
        )

    with col2:
        st.markdown("#### Retrieval (ML)")
        rcfg.use_bm25 = st.toggle("Hybrid: dense + BM25 (RRF)", value=rcfg.use_bm25)
        rcfg.top_k_dense = st.slider("Top-k dense", 1, 20, rcfg.top_k_dense)
        rcfg.top_k_bm25 = st.slider("Top-k BM25", 1, 20, rcfg.top_k_bm25)
        rcfg.top_k_final = st.slider(
            "Top-k final (after rerank)", 1, 10, rcfg.top_k_final
        )

    st.markdown("---")
    st.markdown("#### Generation (LLM)")
    col3, col4 = st.columns(2)
    with col3:
        lcfg.provider = st.selectbox(
            "Provider",
            ["auto", "anthropic", "openai", "ollama", "extractive"],
            index=["auto", "anthropic", "openai", "ollama", "extractive"].index(
                lcfg.provider
            ),
        )
    with col4:
        lcfg.temperature = st.slider("Temperature", 0.0, 1.5, lcfg.temperature, 0.05)
        lcfg.max_tokens = st.slider("Max tokens", 128, 4096, lcfg.max_tokens, 64)

    st.markdown("---")
    st.markdown("#### Intent classifier (ML)")
    nlp = get_nlp_engine()
    if nlp._intent_train_report:
        r = nlp._intent_train_report
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{r['accuracy']:.1%}")
        c2.metric("Macro F1", f"{r['macro_f1']:.1%}")
        c3.metric("Train", r["n_train"])
        c4.metric("Test", r["n_test"])
        st.caption(f"Classes: {', '.join(r['labels'])}")
    if st.button("Retrain on seed data"):
        with st.spinner("Training…"):
            nlp.train_intent()
        st.rerun()


def dev_pipeline_view():
    st.subheader("🔗 Pipeline architecture")
    st.markdown("""
    ```
    INDEXING (offline):
      Raw docs → [CV: OCR if image] → [NLP: Clean / Tokenize / Lemmatize]
        → Chunk → [DL: Embed (MiniLM)] → FAISS HNSW + BM25 index

    QUERY (online):
      User query → [DL: Intent classify (LR/DistilBERT)]
        → [Dense kNN (FAISS) ⊕ BM25 (Reciprocal Rank Fusion)]
          → [DL: Cross-encoder rerank]
            → [LLM: Generate (Anthropic / OpenAI / Ollama / Extractive)]
              → Cited answer

    EVALUATION:
      Recall@k, MRR, nDCG@k, ROUGE-L, BLEU-4, BERTScore, Faithfulness
    ```
    """)

    st.markdown("#### Algorithms in play")
    rows = [
        (
            "OCR (CV)",
            "CV",
            "Tesseract + OpenCV deskew/binarize",
            "pytesseract · opencv-python",
        ),
        ("Image embedding (CV)", "CV", "CLIP ViT-B/32 (optional)", "open_clip"),
        ("Text cleaning (NLP)", "NLP", "regex + NFKC normalize", "stdlib · spaCy"),
        ("Lemmatization (NLP)", "NLP", "spaCy en_core_web_sm", "spacy"),
        ("Chunking (NLP)", "NLP", "Word-window with overlap", "stdlib"),
        (
            "Embedding (DL)",
            "DL",
            "MiniLM-L6-v2 (Transformer encoder)",
            "sentence-transformers",
        ),
        ("Sparse retrieval (ML)", "ML", "BM25 (Okapi)", "rank-bm25"),
        ("Dense index (ML)", "ML", "FAISS HNSW (M=32, ef=200)", "faiss-cpu"),
        ("Hybrid fusion (ML)", "ML", "Reciprocal Rank Fusion (k=60)", "custom"),
        ("Reranker (DL)", "DL", "ms-marco-MiniLM-L-6-v2", "sentence-transformers"),
        (
            "Intent classifier (ML)",
            "ML",
            "TF-IDF + Logistic Regression",
            "scikit-learn",
        ),
        (
            "Generation (DL)",
            "DL",
            "Claude / GPT / Llama / Extractive",
            "anthropic · openai · ollama",
        ),
        ("Sentiment (NLP)", "NLP", "VADER (rule-based)", "vaderSentiment"),
        ("Eval: Retrieval", "—", "Recall@k, MRR, nDCG@k", "custom"),
        ("Eval: Generation", "—", "ROUGE-L, BLEU-4", "rouge-score · custom"),
        ("Eval: Semantic", "—", "BERTScore + sentence-cosine", "bert-score"),
        ("Eval: Faithfulness", "—", "Semantic overlap answer↔contexts", "custom"),
    ]
    import pandas as pd

    df = pd.DataFrame(rows, columns=["Stage", "Family", "Algorithm", "Library"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def dev_evaluation():
    st.subheader("📊 Evaluation metrics")
    st.caption("Provide a gold question set as JSON, or use the auto-generated one.")

    # Generate or upload gold set
    col1, col2 = st.columns([2, 1])
    with col1:
        gold_upload = st.file_uploader(
            "Upload gold-set JSON (optional)",
            type=["json"],
            key="gold_upload",
        )
    with col2:
        n_auto = st.number_input("Or auto-sample N from active index", 1, 50, 8)

    if not st.session_state.active_index:
        st.warning("Set an active index first.")
        return

    if st.button("▶ Run evaluation"):
        # Build gold set
        if gold_upload:
            gold = json.loads(gold_upload.read())
        else:
            # Auto: sample chunks as ground-truth "expected_chunk_ids" for synthetic queries
            rag = get_rag_engine()
            idx = rag.load_index(st.session_state.active_index)
            chunks = idx["chunks"][:n_auto]
            gold = []
            for c in chunks:
                # Use the first 8 words as a synthetic query
                words = c.raw_text.split()[:8]
                q = " ".join(words).rstrip(".,;:") + "?"
                gold.append(
                    {
                        "query": q,
                        "expected_chunk_ids": [c.chunk_id],
                        "expected_answer": c.raw_text[:200],
                        "dataset": st.session_state.active_index,
                    }
                )

        with st.spinner(f"Running eval on {len(gold)} queries…"):

            def answer_fn(q, ds):
                out = answer_query(q, ds or st.session_state.active_index)
                return (
                    out["answer"],
                    out["chunk_ids"],
                    out["contexts"],
                    out["timings"]["total_ms"],
                )

            run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report = run_evaluation(
                gold, answer_fn, run_id=run_id, k=st.session_state.rag_cfg.top_k_final
            )
        report.save(EVAL_DIR / f"{run_id}.json")
        st.session_state.last_eval = report.to_dict()
        st.success(f"Done — saved {run_id}.json")

    report = st.session_state.last_eval
    if not report:
        st.info("No evaluation has been run yet. Click the button above.")
        return

    st.markdown(f"#### Last run: `{report['run_id']}` · {report['n_queries']} queries")

    # Metric cards
    agg = report["aggregate"]
    cols = st.columns(5)
    eval_k = st.session_state.rag_cfg.top_k_final
    metric_specs = [
        (f"Recall@{eval_k}", agg.get(f"recall@{eval_k}", 0), ""),
        ("MRR", agg.get("mrr", 0), ""),
        (f"nDCG@{eval_k}", agg.get(f"ndcg@{eval_k}", 0), ""),
        ("ROUGE-L", agg.get("rouge_l", 0), ""),
        ("Faithfulness", agg.get("faithfulness", 0), ""),
    ]
    for col, (label, val, unit) in zip(cols, metric_specs):
        col.markdown(
            f"<div class='cb-card'>"
            f"<div class='cb-metric-label'>{label}</div>"
            f"<div class='cb-metric-big'>{val:.3f}{unit}</div></div>",
            unsafe_allow_html=True,
        )

    cols2 = st.columns(4)
    more = [
        ("BLEU-4", agg.get("bleu_4", 0)),
        ("BERTScore F1", agg.get("bertscore_f1", 0)),
        ("Semantic sim", agg.get("semantic_sim", 0)),
        ("Avg latency", agg.get("avg_latency_ms", 0) / 1000),
    ]
    for col, (label, val) in zip(cols2, more):
        unit = "s" if "latency" in label.lower() else ""
        col.markdown(
            f"<div class='cb-card'><div class='cb-metric-label'>{label}</div>"
            f"<div class='cb-metric-big'>{val:.3f}{unit}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Per-query breakdown")
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "#": i + 1,
                "query": q["query"][:60],
                "recall": q["recall"],
                "rouge_l": q["rouge_l"],
                "faithfulness": q["faithfulness"],
                "latency_ms": q["latency_ms"],
            }
            for i, q in enumerate(report["per_query"])
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def dev_logs():
    st.subheader("📜 Logs")
    if not st.session_state.logs:
        st.info("No log entries yet — interact with the app to generate logs.")
        return
    levels = st.multiselect(
        "Level filter",
        ["INFO", "DEBUG", "WARN", "ERROR"],
        default=["INFO", "WARN", "ERROR"],
    )
    rows = [l for l in st.session_state.logs if l["level"] in levels]
    for l in rows[-100:][::-1]:
        color = {
            "INFO": "#4a6cf7",
            "DEBUG": "#5a6280",
            "WARN": "#f59e0b",
            "ERROR": "#ef4444",
        }.get(l["level"], "#5a6280")
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace; font-size:12px; padding:4px 0; border-bottom:1px solid #1f2435;'>"
            f"<span style='color:#5a6280;'>{html.escape(str(l['ts']))}</span> "
            f"<span style='color:{color}; font-weight:600; margin:0 8px;'>{html.escape(str(l['level']))}</span> "
            f"<span style='color:#4a6cf7;'>[{html.escape(str(l['module']))}]</span> "
            f"<span style='color:#e8eaf2;'>{html.escape(str(l['msg']))}</span></div>",
            unsafe_allow_html=True,
        )


# **Router**
if st.session_state.mode == "User":
    user_mode()
else:
    developer_mode()
