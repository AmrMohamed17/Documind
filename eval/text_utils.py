# eval/text_utils.py
"""Shared text helpers for the validator AND the eval metrics,
so that 'found' means exactly the same thing in both places."""
import re


REFUSAL_MARKERS = [
    "cannot find the answer",
    "can't find the answer",
    "not find the answer in the provided",
    "no relevant documents",
]


def is_refusal(answer: str) -> bool:
    a = normalize(answer)
    return any(m in a for m in REFUSAL_MARKERS)


def normalize(text: str) -> str:
    """Unify quotes/dashes, strip Markdown markup, collapse whitespace, lowercase."""
    text = (text.replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u2014", "-").replace("\u2013", "-"))
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # [text](url) -> text
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def split_fragments(snippet: str) -> list[str]:
    """A snippet may join spans with '...'. Return the real spans."""
    return [p for p in re.split(r"\s*\.\.\.\s*", snippet) if p.strip()]