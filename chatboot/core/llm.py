"""
llm.py — LLM wrapper for answer generation
=============================================
Tries providers in this order:
  1. Anthropic API   (if ANTHROPIC_API_KEY)
  2. OpenAI API      (if OPENAI_API_KEY)
  3. Local Ollama    (if OLLAMA_HOST reachable)
  4. Extractive fallback — concatenates retrieved chunks (NO LLM needed)

The fallback means the chatbot ALWAYS works, even with zero API keys.
"""

from __future__ import annotations
import os
import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "ollama"
    anthropic_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "phi4-mini"
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 1024


SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. Use ONLY the provided context to answer. "
    "Cite sources inline using bracket notation like [1], [2] matching the context "
    "numbers. If the context does not contain the answer, say so plainly. "
    "Be concise and well-structured (use short paragraphs or bullet lists)."
)


def _format_context(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


def _call_anthropic(prompt: str, system: str, cfg: LLMConfig) -> str:
    try:
        import anthropic
    except ImportError:
        return ""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as e:
        logger.warning("Anthropic call failed: %s", e)
        return ""


def _call_openai(prompt: str, system: str, cfg: LLMConfig) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        return ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
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
    except Exception as e:
        logger.warning("OpenAI call failed: %s", e)
        return ""


def _call_ollama(prompt: str, system: str, cfg: LLMConfig) -> str:
    body = json.dumps(
        {
            "model": cfg.ollama_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
        }
    ).encode()
    req = urllib.request.Request(
        f"{cfg.ollama_host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        logger.warning("Ollama call failed: %s", e)
        return ""


def _extractive(query: str, contexts: list[str]) -> str:
    """
    No-LLM fallback. Concatenates the top contexts with cite markers.
    Not a real generation — but always produces useful output.
    """
    if not contexts:
        return "I couldn't find relevant information in the indexed documents."
    parts = []
    for i, ctx in enumerate(contexts[:3], 1):
        snippet = ctx if len(ctx) <= 300 else ctx[:300] + "…"
        parts.append(f"{snippet} [{i}]")
    body = "\n\n".join(parts)
    return f"Based on the retrieved context:\n\n{body}"


def generate_answer(
    query: str,
    contexts: list[str],
    cfg: LLMConfig | None = None,
    history: list[dict] | None = None,
) -> tuple[str, str]:
    """
    Generate an answer to the query, grounded in the contexts.

    Returns: (answer_text, provider_used)
    """
    cfg = cfg or LLMConfig()
    history = history or []

    # Build the prompt
    ctx_str = _format_context(contexts) if contexts else "(no context available)"
    hist_str = ""
    if history:
        hist_str = "\n\nPrevious turns:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
        )
    prompt = f"CONTEXT:\n{ctx_str}\n{hist_str}\n\nQUESTION: {query}\n\nANSWER:"

    # Try providers in order
    providers = []
    if cfg.provider == "auto":
        providers = ["anthropic", "openai", "ollama", "extractive"]
    else:
        providers = [cfg.provider]

    for p in providers:
        if p == "anthropic":
            ans = _call_anthropic(prompt, SYSTEM_PROMPT, cfg)
            if ans:
                return ans, "anthropic"
        elif p == "openai":
            ans = _call_openai(prompt, SYSTEM_PROMPT, cfg)
            if ans:
                return ans, "openai"
        elif p == "ollama":
            ans = _call_ollama(prompt, SYSTEM_PROMPT, cfg)
            if ans:
                return ans, "ollama"
        elif p == "extractive":
            return _extractive(query, contexts), "extractive"

    return _extractive(query, contexts), "extractive"


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is COVID-19?"
    ctx = [
        "COVID-19 is a respiratory viral disease caused by the SARS-CoV-2 virus.",
        "Symptoms include fever, dry cough, fatigue, and loss of taste or smell.",
        "Risk factors include obesity, diabetes, and cardiovascular disease.",
    ]
    ans, provider = generate_answer(q, ctx)
    print(f"[{provider}] {ans}")
