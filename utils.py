"""
utils.py
Language detection and prompt construction utilities.
"""

from langdetect import detect, DetectorFactory

# Fix the random seed for deterministic language detection
DetectorFactory.seed = 0

# Human-readable language names for the prompt instruction
LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "it": "Italian",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "el": "Greek",
    "hu": "Hungarian",
    "cs": "Czech",
    "ro": "Romanian",
    "uk": "Ukrainian",
    "bn": "Bengali",
    "fa": "Persian",
    "he": "Hebrew",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "ur": "Urdu",
}


def detect_language(text: str) -> tuple[str, str]:
    """
    Detect the language of *text*.

    Returns:
        (lang_code, lang_name) — e.g. ("hi", "Hindi")
    """
    try:
        code = detect(text)
        name = LANG_NAMES.get(code, code.upper())
        return code, name
    except Exception:
        return "en", "English"   # safe fallback


def build_prompt(query: str, retrieved_chunks: list[dict], lang_name: str, history: list[list[str]] = None) -> str:
    """
    Construct the user-side LLM prompt (context + question only).
    System-level rules are defined in llm_interface.SYSTEM_PROMPT.

    Parameters
    ----------
    query            : the original user question
    retrieved_chunks : list of {"text": str, "page": int} dicts
    lang_name        : human-readable target language, e.g. "Hindi"
    history          : conversation history as a list of [user_message, bot_message]
    """
    if not retrieved_chunks:
        return ""

    context_parts = [
        f"[Page {c['page']}]\n{c['text']}" for c in retrieved_chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    history_text = ""
    if history:
        history_text = "PREVIOUS CONVERSATION:\n"
        for user_msg, bot_msg in history[-3:]:  # limit to last 3 turns
            if bot_msg:
                history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        history_text += "\n"

    return f"""{history_text}CONTEXT (from uploaded PDF):
{context}

---

QUESTION (in {lang_name}): {query}

ANSWER (in {lang_name}):"""


def format_citations(chunks: list[dict]) -> str:
    """Return a formatted citation string, e.g. 'Pages: 2, 5'."""
    pages = sorted({c["page"] for c in chunks})
    return "Pages: " + ", ".join(str(p) for p in pages)
