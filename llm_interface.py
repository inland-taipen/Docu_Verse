"""
llm_interface.py
Unified LLM interface using Groq Cloud.
"""

import os
import time
import logging
from typing import Optional, Generator

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Singleton Groq client — instantiated once, reused for every request
# ---------------------------------------------------------------------------
_groq_client = None


def _get_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        logger.info("Groq client initialised (model: %s)", DEFAULT_MODEL)
    return _groq_client


# System prompt lives here — separate from user/context prompt for better adherence
SYSTEM_PROMPT = (
    "You are a strict PDF assistant. Follow these rules:\n"
    "1. Answer ONLY using the CONTEXT provided. Never use external knowledge.\n"
    "2. Respond in the same language as the QUESTION.\n"
    "3. Use ### headings, **bold** key terms, and bullet points for structure.\n"
    "4. Include at least one short exact quote from the CONTEXT in ASCII double quotes. "
    "Never translate the quote — keep it in the original language of the source.\n"
    '5. If the question cannot be answered from the context, reply exactly: '
    '"I cannot answer based on the provided PDF."\n'
    "6. Always cite the page number(s) where the information was found."
)


def ask_llm(prompt: str, model: Optional[str] = None, retries: int = 2) -> str:
    """Send prompt to Groq and return the full response. Retries on transient errors."""
    model = model or DEFAULT_MODEL
    client = _get_client()
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Groq attempt %d/%d failed: %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.error("All Groq retries exhausted: %s", exc)
                return f"[Groq error: {exc}]"
    return "[Groq error: unexpected]"


def ask_llm_stream(prompt: str, model: Optional[str] = None) -> Generator[str, None, None]:
    """Stream tokens from Groq. Yields progressively accumulated response strings."""
    model = model or DEFAULT_MODEL
    client = _get_client()
    try:
        stream = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            stream=True,
        )
        accumulated = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                accumulated += delta
                yield accumulated
    except Exception as exc:
        logger.error("Groq streaming error: %s", exc)
        yield f"[Groq error: {exc}]"
