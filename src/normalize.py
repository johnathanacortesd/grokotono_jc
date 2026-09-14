"""Normalización de texto, OCR-fold y limpieza de subtema/tono."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

TONOS = ("Positivo", "Negativo", "Neutro")

DANGLING = {
    "de", "del", "la", "el", "los", "las", "en", "con", "por", "para", "y", "o",
    "u", "e", "a", "al", "un", "una", "unos", "unas", "su", "sus", "que", "se",
    "ha", "han", "porque", "como", "sin", "sobre", "tras", "desde", "hacia",
    "entre", "ni", "lo", "le", "les", "más", "mas",
}

STOPWORDS = DANGLING | {
    "es", "son", "fue", "fueron", "ser", "esta", "este", "estos", "estas",
    "hay", "nos", "me", "te", "tu", "muy", "ya", "si", "no", "tambien",
    "también", "pero", "ante", "bajo", "durante", "mediante", "segun", "según",
}

COLLAGE_HEADS = {
    "mencion", "mención", "presencia", "declaraciones", "noticia", "noticias",
    "alusion", "alusión", "referencia", "cobertura", "actualidad", "gestion",
    "gestión", "temas", "varios", "otros", "general",
}

GENERIC_SUBTEMAS = {
    "hecho informativo", "noticias generales", "gestion institucional",
    "gestión institucional", "actividad institucional", "informacion general",
    "información general", "cobertura informativa", "temas generales",
}


def as_text(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null", "nat"}:
        return ""
    return s


def strip_controls(s: str) -> str:
    return "".join(ch for ch in as_text(s) if (ch >= " " or ch in "\n\t") and ch not in "\ufffe\uffff")


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def fold_text(s: str) -> str:
    t = strip_accents(strip_controls(s)).lower()
    t = re.sub(r"[^a-z0-9ñ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def ocr_fold(s: str) -> str:
    """Fold que absorbe confusiones típicas de OCR en titulares colombianos."""
    t = fold_text(s)
    t = t.replace("0", "o").replace("1", "l").replace("5", "s")
    t = t.replace("rn", "m")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def content_words(s: str, *, ocr: bool = True) -> list[str]:
    folded = ocr_fold(s) if ocr else fold_text(s)
    return [w for w in folded.split() if w not in STOPWORDS and len(w) > 2]


def parse_name_list(raw: str) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,;\n|/]+", str(raw))
    out, seen = [], set()
    for p in parts:
        item = re.sub(r"\s+", " ", p).strip(" -•\t")
        if not item:
            continue
        key = fold_text(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def brand_tokens(*groups: Iterable[str]) -> set[str]:
    toks: set[str] = set()
    for group in groups:
        for item in group or []:
            toks.update(content_words(item))
            f = fold_text(item)
            if f:
                toks.add(f)
    return toks


def _keep_token_case(token: str) -> bool:
    letters = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", "", token)
    if not letters:
        return True
    if letters.isupper() and 2 <= len(letters) <= 6:
        return True
    if token[:1].isupper() and token[1:].islower() and len(letters) >= 2:
        return True
    return False


def sentence_case(phrase: str) -> str:
    """Mayúscula solo en la primera letra; conserva siglas y nombres propios."""
    phrase = re.sub(r"\s+", " ", strip_controls(phrase)).strip(" .;:,-")
    if not phrase:
        return ""
    words = phrase.split()
    titleish = len(words) >= 4 and sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.85
    out = []
    for i, w in enumerate(words):
        if i == 0:
            if w.isupper() and 2 <= len(re.sub(r"[^A-Za-z]", "", w)) <= 6:
                out.append(w)
            else:
                out.append(w[:1].upper() + w[1:])
            continue
        if w.isupper() and 2 <= len(re.sub(r"[^A-Za-z]", "", w)) <= 6:
            out.append(w)
        elif titleish and not _keep_token_case(w) and w.lower() not in {"de", "del", "la", "el", "y", "en", "con"}:
            out.append(w.lower())
        else:
            out.append(w)
    return " ".join(out)


def strip_dangling(words: list[str]) -> list[str]:
    while words and fold_text(words[-1]) in DANGLING:
        words.pop()
    return words


def looks_like_collage(phrase: str) -> bool:
    words = phrase.split()
    if not words:
        return True
    folded = fold_text(phrase)
    if folded in GENERIC_SUBTEMAS:
        return True
    connectors = sum(1 for w in words if fold_text(w) in {"y", "e", "o"})
    if connectors >= 2 and len(words) <= 8:
        return True
    head = fold_text(words[0])
    if head in COLLAGE_HEADS:
        return True
    if re.search(r"[,;|/]", phrase):
        return True
    return False


def looks_like_title_scrap(subtema: str, titulo: str) -> bool:
    """True solo si el subtema es un recorte crudo del titular (no una síntesis)."""
    if not subtema or not titulo:
        return False
    sw = ocr_fold(subtema).split()
    tw = ocr_fold(titulo).split()
    if len(sw) >= 6 and tw[: len(sw)] == sw and len(tw) > len(sw) + 2:
        return True
    return False


def canonicalize_tono(raw: str) -> str:
    t = fold_text(raw).replace(" ", "")
    mapping = {
        "positivo": "Positivo",
        "positive": "Positivo",
        "negativo": "Negativo",
        "negative": "Negativo",
        "neutro": "Neutro",
        "neutral": "Neutro",
        "mixto": "Neutro",
    }
    return mapping.get(t, "Neutro")


def clean_subtema(raw: str, *, titulo: str = "", resumen: str = "", marca: str = "") -> str:
    text = strip_controls(raw)
    text = re.sub(r"[\"'«»“”‘’]", "", text)
    text = re.sub(r"[:;|/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-")
    words = [w for w in text.split() if w]
    if len(words) > 14:
        words = words[:14]
    words = strip_dangling(words)
    phrase = sentence_case(" ".join(words))
    if not phrase or looks_like_collage(phrase) or looks_like_title_scrap(phrase, titulo):
        phrase = _fallback_subtema(resumen, titulo, marca)
    words = strip_dangling(phrase.split())
    phrase = sentence_case(" ".join(words))
    brand_w = set(content_words(marca))
    phrase_w = set(content_words(phrase))
    if not phrase or (brand_w and phrase_w and phrase_w <= brand_w):
        phrase = _fallback_subtema(resumen, titulo, marca)
    return phrase or "Hecho informativo"


def _fallback_subtema(resumen: str, titulo: str, marca: str) -> str:
    source = as_text(resumen) or as_text(titulo)
    source = re.split(r"(?<=[.!?])\s+", source, maxsplit=1)[0]
    source = re.sub(r"^(?:imagenes|en imagenes|fotos|video|en vivo)\s*\|\s*", "", source, flags=re.I)
    words = re.sub(r"[,.;:!?¿¡\"'()\[\]{}]", " ", source).split()
    skip = brand_tokens([marca])
    kept = []
    for w in words:
        if fold_text(w) in skip and len(kept) == 0:
            continue
        kept.append(w)
        if len(kept) >= 10:
            break
    kept = strip_dangling(kept)
    phrase = sentence_case(" ".join(kept))
    if looks_like_collage(phrase) or len(phrase.split()) < 3:
        kept = strip_dangling(re.sub(r"[,.;:!?¿¡\"']", " ", as_text(titulo)).split()[:8])
        phrase = sentence_case(" ".join(kept))
    return phrase


def mentions_target(titulo: str, resumen: str, marca: str, aliases: Sequence[str], voceros: Sequence[str]) -> bool:
    blob = ocr_fold(f"{titulo} {resumen}")
    if not blob:
        return False
    for item in [marca, *aliases, *voceros]:
        f = ocr_fold(item)
        if len(f) < 3:
            continue
        if f in blob:
            return True
        compact = f.replace(" ", "")
        if len(compact) >= 4 and compact in blob.replace(" ", ""):
            return True
    return False


def extract_brand_passages(
    titulo: str,
    resumen: str,
    marca: str,
    aliases: Sequence[str],
    voceros: Sequence[str],
    max_chars: int = 1600,
) -> str:
    """Oraciones que mencionan marca/alias/voceros: son las que deciden el tono."""
    titulo = as_text(titulo)
    resumen = as_text(resumen)
    objetivos = [ocr_fold(x) for x in [marca, *aliases, *voceros] if ocr_fold(x) and len(ocr_fold(x)) >= 3]
    if not resumen:
        return titulo[:220]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", resumen) if s.strip()]
    picked = []
    for i, sent in enumerate(sentences):
        nf = ocr_fold(sent)
        if any(obj in nf for obj in objetivos):
            block = sent
            if len(sent.split()) < 12 and i + 1 < len(sentences):
                block = f"{sent} {sentences[i + 1]}"
            if block not in picked:
                picked.append(block)
    if picked:
        text = " ".join(picked)
        if titulo and ocr_fold(titulo) not in ocr_fold(text):
            if any(obj in ocr_fold(titulo) for obj in objetivos):
                text = f"{titulo}. {text}"
        return text[:max_chars]
    if any(obj in ocr_fold(titulo) for obj in objetivos):
        return f"{titulo}. {resumen[:400]}"[:max_chars]
    return ""
