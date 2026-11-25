"""Text preprocessing.

Two levels, on purpose:
* clean_text: light cleaning (Unicode, URLs, spaces). Used for Sentence-BERT, which expects natural text.
* normalize_for_tfidf: lowercase, remove punctuation and stop words. Used for TF-IDF.

Negations and words like "off"/"on" are NOT removed, because they change a complaint's meaning
("lights not working", "lights off"). scikit-learn's default stop-word list would remove them.
"""
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

KEEP_WORDS = {"no", "not", "nor", "nothing", "none", "nobody", "never", "cannot", "off", "on",
              "out", "down", "up", "without", "back", "over"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - KEEP_WORDS)
MIN_CHARS, MAX_CHARS = 5, 2000


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_tfidf(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", clean_text(text).lower())
    return " ".join(t for t in text.split() if t not in STOP_WORDS)


def validate_complaint(text: str) -> str:
    """Return cleaned text, or raise ValueError with a message to show the user."""
    cleaned = clean_text(text or "")
    if len(cleaned) < MIN_CHARS:
        raise ValueError(f"Please describe the problem in at least {MIN_CHARS} characters.")
    if len(cleaned) > MAX_CHARS:
        raise ValueError(f"Please keep the description under {MAX_CHARS} characters.")
    if not re.search(r"[A-Za-z]", cleaned):
        raise ValueError("The description must contain words, not only numbers or symbols.")
    return cleaned
