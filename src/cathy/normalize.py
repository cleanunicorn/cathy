"""Cleanup between extracted text and the TTS engine: drop front/back matter
nobody wants narrated, and scrub markup artifacts engines would read aloud."""

import re
from urllib.parse import urlparse

# Chapter titles that mark non-narratable front/back matter, matched
# case-insensitively after stripping leading numbering.
SKIP_TITLES = {
    "title page",
    "half title",
    "half title page",
    "copyright",
    "copyright page",
    "contents",
    "table of contents",
    "index",
    "list of illustrations",
    "list of figures",
    "list of tables",
    "bibliography",
    "references",
    "colophon",
    "about the publisher",
}
SKIP_TITLE_PREFIXES = ("also by ", "other books by ", "praise for ")


def _canonical(title: str) -> str:
    return re.sub(r"^[\d\s.:•·—–-]+", "", title.strip().lower())


def _looks_like_copyright(paragraphs: list[str]) -> bool:
    if len(paragraphs) > 15:
        return False
    text = " ".join(paragraphs).lower()
    return "all rights reserved" in text or "isbn" in text


def _looks_like_toc(paragraphs: list[str]) -> bool:
    """A run of many short lines near the book's start reads like a TOC."""
    if len(paragraphs) < 5:
        return False
    short = sum(1 for p in paragraphs if len(p) < 60)
    return short / len(paragraphs) >= 0.8


def skip_reason(index: int, title: str, paragraphs: list[str]) -> str | None:
    """Why this chapter should not be narrated, or None to keep it."""
    canonical = _canonical(title)
    if canonical in SKIP_TITLES or canonical.startswith(SKIP_TITLE_PREFIXES):
        return title
    if _looks_like_copyright(paragraphs):
        return title or "copyright page"
    if index < 3 and not title and _looks_like_toc(paragraphs):
        return "table of contents"
    return None


def drop_front_back_matter(
    chapters: list[tuple[str, list[str]]],
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Split chapters into (kept, names of skipped front/back matter)."""
    kept, skipped = [], []
    for i, (title, paragraphs) in enumerate(chapters):
        reason = skip_reason(i, title, paragraphs)
        if reason is None:
            kept.append((title, paragraphs))
        else:
            skipped.append(reason)
    return kept, skipped


# Soft hyphen, zero-width characters, and BOM vanish; non-breaking space
# becomes a regular one.
_INVISIBLES = str.maketrans("\u00a0", " ", "\u00ad\u200b\u200c\u200d\ufeff")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_EMPHASIS = re.compile(r"([*_`]{1,3})(?!\s)([^*_`]+?)(?<!\s)\1")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\"']+")
_FOOTNOTE = re.compile(r"\[\d{1,3}\]")
# Superscript digits directly after punctuation are footnote markers; after a
# letter or digit they may be a real exponent (m²), so those are kept.
_SUPERSCRIPT_MARK = re.compile(
    r"(?<=[.,;:!?)\]\"\u201d\u2019])[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+"
)
_SEPARATOR = re.compile(r"^[\s*#~•·⁂†‡=_.\-—–]+$")
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
_WHITESPACE = re.compile(r"\s+")


def _url_domain(match: re.Match) -> str:
    """Read a URL as its bare domain — 'see example.com' beats spelling a path.

    Trailing punctuation is sentence punctuation, not URL: keep it."""
    url = match.group(0).rstrip(".,;:!?")
    trailing = match.group(0)[len(url):]
    host = urlparse(url if "://" in url else f"https://{url}").hostname
    return ((host or "").removeprefix("www.") or url) + trailing


def normalize_text(text: str) -> str:
    """Inline cleanup safe for any prose (also applied to chapter titles)."""
    text = text.translate(_INVISIBLES)
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _URL.sub(_url_domain, text)
    text = _FOOTNOTE.sub("", text)
    text = _SUPERSCRIPT_MARK.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_paragraph(text: str) -> str:
    """Clean one paragraph; returns '' for paragraphs that shouldn't be read
    at all (scene-separator glyph runs, stray page numbers)."""
    text = normalize_text(text)
    if _SEPARATOR.match(text) or _PAGE_NUMBER.match(text):
        return ""
    return text


def normalize_chapters(
    chapters: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    """Normalize every paragraph, dropping empties and emptied-out chapters."""
    result = []
    for title, paragraphs in chapters:
        cleaned = [p for p in map(normalize_paragraph, paragraphs) if p]
        if cleaned:
            result.append((normalize_text(title), cleaned))
    return result
