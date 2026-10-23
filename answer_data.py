"""Train-only answer targets; no downloaded tokenizer or language model."""

import unicodedata

MAX_ANSWER_BYTES = 64
PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE = 259
INVALID_ANSWERS = {"", "unanswerable", "unanswered", "no answer"}


def normalize_answer(value):
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).strip().lower().split())


def answer_from_record(record):
    text = normalize_answer(record.get("most_common_answer"))
    return text if text not in INVALID_ANSWERS else ""


def answer_bytes(text):
    """Return a complete UTF-8 answer or None; never silently truncate the target."""
    text = normalize_answer(text)
    encoded = text.encode("utf-8")
    if text in INVALID_ANSWERS or len(encoded) > MAX_ANSWER_BYTES:
        return None
    return [byte + 3 for byte in encoded]
