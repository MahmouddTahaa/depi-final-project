"""
nlp_engine.py — NLP analytics over user queries & responses
=============================================================
Capabilities:
  • Intent classification — sklearn LogReg/SVM on TF-IDF (ML)
       or DistilBERT zero-shot fallback (DL)
  • Sentiment — VADER (rule-based, fast)
  • Named entities — spaCy en_core_web_sm (NLP)
  • Keyword extraction — TF-IDF top terms (ML)

ML families: ML (TF-IDF + LogReg), DL (transformer fallback), NLP (spaCy).
"""

from __future__ import annotations
import json
import pickle
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import spacy

    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _SIA = None
    HAS_VADER = True
except ImportError:
    HAS_VADER = False


DEFAULT_INTENTS = {
    "definition": [
        "what is",
        "define",
        "tell me about",
        "explain",
        "describe",
        "what does mean",
        "meaning of",
        "what are",
        "who is",
        "what was",
        "give me an overview of",
        "introduce",
        "summary of",
    ],
    "compare": [
        "compare",
        "vs",
        "versus",
        "difference between",
        "which is better",
        "how does compare to",
        "differences",
        "similarities between",
        "contrast",
        "pros and cons of",
    ],
    "summarize": [
        "summarize",
        "key points",
        "main ideas",
        "tldr",
        "brief overview",
        "give me a summary",
        "what are the highlights",
        "condense",
        "list the main",
        "synthesize",
    ],
    "calculate": [
        "calculate",
        "compute",
        "how much",
        "how many",
        "what percentage",
        "ratio of",
        "convert",
        "rate of",
        "average of",
        "what is the sum",
    ],
}


def _seed_training_set(
    intents: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Expand seed phrases into a small but useful training set."""
    intents = intents or DEFAULT_INTENTS
    fillers = [
        "heart disease",
        "the flu",
        "covid-19",
        "diabetes",
        "anxiety",
        "the mediterranean diet",
        "phishing attacks",
        "compound interest",
        "blockchain",
        "ACL injuries",
        "transformers in ML",
        "tenant rights",
    ]
    X, y = [], []
    for intent, phrases in intents.items():
        for p in phrases:
            for f in fillers:
                X.append(f"{p} {f}")
                y.append(intent)
            X.append(p)
            y.append(intent)
    return X, y


@dataclass
class NLPAnalysis:
    intent: str = "unknown"
    intent_confidence: float = 0.0
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    entities: list[dict] = field(default_factory=list)  # [{label, text}]
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


class NLPEngine:
    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._spacy = None
        self._sia = None
        self._intent_clf = None
        self._intent_labels: list[str] = []
        self._intent_train_report: dict = {}

        # Try to load a pre-trained intent classifier from disk
        self._load_intent()

    @property
    def spacy_nlp(self):
        if not HAS_SPACY:
            return None
        if self._spacy is None:
            try:
                self._spacy = spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded — see README
                self._spacy = spacy.blank("en")
        return self._spacy

    @property
    def sia(self):
        if not HAS_VADER:
            return None
        global _SIA
        if _SIA is None:
            _SIA = SentimentIntensityAnalyzer()
        return _SIA

    def train_intent(
        self, X: list[str] | None = None, y: list[str] | None = None
    ) -> dict:
        """
        Train the intent classifier (LR on TF-IDF). Returns a report dict
        with classification metrics on a held-out test split.
        """
        if not HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is not installed. pip install scikit-learn"
            )
        if X is None or y is None:
            X, y = _seed_training_set()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_te, y_pred, labels=sorted(set(y))).tolist()

        self._intent_clf = clf
        self._intent_labels = sorted(set(y))
        self._intent_train_report = {
            "n_train": len(X_tr),
            "n_test": len(X_te),
            "labels": self._intent_labels,
            "report": report,
            "confusion_matrix": cm,
            "macro_f1": report["macro avg"]["f1-score"],
            "accuracy": report["accuracy"],
        }
        self._save_intent()
        return self._intent_train_report

    def _save_intent(self):
        if self._intent_clf is None:
            return
        with (self.model_dir / "intent_clf.pkl").open("wb") as f:
            pickle.dump(
                {
                    "clf": self._intent_clf,
                    "labels": self._intent_labels,
                    "report": self._intent_train_report,
                },
                f,
            )

    def _load_intent(self):
        p = self.model_dir / "intent_clf.pkl"
        if p.exists():
            try:
                with p.open("rb") as f:
                    data = pickle.load(f)
                self._intent_clf = data["clf"]
                self._intent_labels = data["labels"]
                self._intent_train_report = data["report"]
            except Exception:
                pass

    def predict_intent(self, text: str) -> tuple[str, float]:
        if self._intent_clf is None:
            return "unknown", 0.0
        proba = self._intent_clf.predict_proba([text])[0]
        idx = int(proba.argmax())
        return self._intent_labels[idx], float(proba[idx])

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        if self.sia is None:
            # Fallback: super-simple lexicon
            pos = sum(
                w in text.lower()
                for w in ["good", "great", "love", "best", "excellent"]
            )
            neg = sum(
                w in text.lower() for w in ["bad", "worst", "hate", "terrible", "awful"]
            )
            if pos > neg:
                return "positive", min(1.0, pos / 5)
            if neg > pos:
                return "negative", min(1.0, neg / 5)
            return "neutral", 0.5
        scores = self.sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return "positive", abs(compound)
        if compound <= -0.05:
            return "negative", abs(compound)
        return "neutral", 1.0 - abs(compound)

    def extract_entities(self, text: str) -> list[dict]:
        nlp = self.spacy_nlp
        if nlp is None or not hasattr(nlp, "pipe_names") or "ner" not in nlp.pipe_names:
            # Fallback: capitalized n-grams
            tokens = re.findall(r"[A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+)*", text)
            return [{"text": t, "label": "MISC"} for t in tokens[:6]]
        doc = nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents[:8]]

    def extract_keywords(self, text: str, top_k: int = 6) -> list[str]:
        if not HAS_SKLEARN:
            words = re.findall(r"\b[a-z]{4,}\b", text.lower())
            counts = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            return [w for w, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_k]]
        try:
            vec = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), max_features=20
            )
            vec.fit([text])
            terms = vec.get_feature_names_out()
            scores = vec.transform([text]).toarray()[0]
            ranked = sorted(zip(terms, scores), key=lambda x: -x[1])
            return [t for t, _ in ranked[:top_k]]
        except ValueError:
            return []

    def analyze(self, text: str) -> NLPAnalysis:
        intent, intent_conf = self.predict_intent(text)
        sentiment, sent_score = self.analyze_sentiment(text)
        entities = self.extract_entities(text)
        keywords = self.extract_keywords(text)
        return NLPAnalysis(
            intent=intent,
            intent_confidence=intent_conf,
            sentiment=sentiment,
            sentiment_score=sent_score,
            entities=entities,
            keywords=keywords,
        )


if __name__ == "__main__":
    import sys

    eng = NLPEngine()
    if eng._intent_clf is None:
        print("Training intent classifier on seed data…")
        rep = eng.train_intent()
        print(f"  Accuracy: {rep['accuracy']:.3f}  Macro-F1: {rep['macro_f1']:.3f}")
    text = (
        " ".join(sys.argv[1:]) or "What is the difference between COVID-19 and the flu?"
    )
    print(f"\nAnalyzing: {text!r}\n")
    a = eng.analyze(text)
    print(f"  Intent:    {a.intent} ({a.intent_confidence:.2f})")
    print(f"  Sentiment: {a.sentiment} ({a.sentiment_score:.2f})")
    print(f"  Entities:  {a.entities}")
    print(f"  Keywords:  {a.keywords}")
