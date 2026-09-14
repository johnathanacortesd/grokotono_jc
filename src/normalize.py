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


ORG_HEADS = {
    "universidad",
    "universitaria",
    "universitario",
    "gobernacion",
    "alcaldia",
    "instituto",
    "institucion",
    "fundacion",
    "colegio",
    "escuela",
    "empresa",
    "secretaria",
    "ministerio",
    "centro",
    "corporacion",
    "hospital",
    "clinica",
    "rectorado",
}

_STOP_NAME = {"de", "del", "la", "el", "los", "las", "y", "e", "da", "do"}

# Verbos / hechos de gestión o logro: la marca como agente cuenta Positivo
# aunque el texto no traiga adjetivos.
GESTION_VERBS = {
    "entrego", "entrega", "entregaron", "entregara",
    "lanzo", "lanza", "lanzaron", "lanzamiento",
    "inauguro", "inaugura", "inauguraron", "inauguracion",
    "impulso", "impulsa", "impulsaron",
    "aprobo", "aprueba", "aprobaron",
    "firmo", "firma", "firmaron",
    "anuncio", "anuncia", "anunciaron",
    "avanzo", "avanza", "avanzaron",
    "invirtio", "invierte", "invirtieron",
    "obtuvo", "obtiene", "obtuvieron",
    "logro", "logra", "lograron",
    "ocupo", "ocupa",
    "destaco", "destaca", "destacaron",
    "otorgo", "otorga", "otorgaron",
    "graduo", "gradua", "graduaron",
    "suscribio", "suscribe", "suscribieron",
    "adjudico", "adjudica",
    "construyo", "construye",
    "habilito", "habilita",
    "fortalecio", "fortalece",
    "lidero", "lidera",
    "adelanto", "adelanta",
    "gestiono", "gestiona",
    "realizo", "realiza", "realizaron",
    "celebro", "celebra",
    "recibio", "recibe",
    "alcanzo", "alcanza",
    "estreno", "estrena",
    "abrio", "abre",
    "destino", "destina",
    "cofinancio",
    "acredito", "acredita",
    "presento", "presenta",
    "publico", "publica",
    "gano", "gana",
    "concreto", "concreta",
    "beneficio", "beneficia",
    "capacito", "capacita",
    "formo", "forma",
    "amplio", "amplia",
    "renovo", "renueva",
    "modernizo", "moderniza",
    "certifico", "certifica",
    "posiciono", "posiciona",
    "ascendio", "asciende",
    "puso", "dio",
    "aposto", "apuesta",
    "promovio", "promueve",
    "articulo", "articula",
    "vinculo", "vincula",
    "fortalecieron",
    "instalo", "instala",
    "doto", "dota",
    "beco", "beca",
}

GESTION_NOUNS = {
    "convenio", "convenios", "acuerdo", "acuerdos", "alianza", "alianzas",
    "ranking", "acreditacion", "beca", "becas", "infraestructura",
    "inversion", "inversiones", "programa", "programas", "obra", "obras",
    "graduacion", "reconocimiento", "reconocimientos", "gestion",
    "campus", "laboratorios", "investigacion", "convocatoria",
}

NEG_HEADS = {
    "denuncia", "denuncian", "denuncio", "denunciaron",
    "queja", "quejas", "reclamo", "reclaman", "reclamaron",
    "protesta", "protestan", "protestaron", "manifestacion",
    "sancion", "sancionan", "sanciono", "sancionaron",
    "investigan", "investigacion",
    "corrupcion", "irregularidades", "escandalo",
    "demanda", "demandan", "demandaron",
    "critica", "critican", "criticas",
    "cuestionan", "cuestiono",
    "senalan", "senalo",
    "rechazo", "rechazan",
    "polemica",
    "retraso", "retrasos", "falla", "fallas",
    "abandono", "negligencia",
}

LOCATIVE = {"en", "desde", "hacia", "sede", "escenario", "instalaciones"}
LOCATIVE_BEFORE = re.compile(
    r"\b(?:en|desde|hacia|sede(?:\s+de)?)\s+"
    r"(?:el|la|los|las|un|una|su|sus)?"
    r"(?:\s+\w+){0,6}\s+$"
)


def _entity_is_locative(folded: str, entity_folded: str) -> bool:
    idx = folded.find(entity_folded)
    if idx <= 0:
        return False
    return LOCATIVE_BEFORE.search(folded[:idx]) is not None


def _add_unique(out: list[str], seen: set[str], item: str) -> None:
    item = re.sub(r"\s+", " ", as_text(item)).strip(" .,;:-")
    if not item:
        return
    key = ocr_fold(item)
    if len(key) < 3 or key in seen:
        return
    seen.add(key)
    out.append(item)


def name_variants(name: str) -> list[str]:
    """Variantes de una marca/alias: puntuación, nombre corto, 'U. de…', sigla."""
    raw = as_text(name)
    out: list[str] = []
    seen: set[str] = set()
    if not raw:
        return out
    _add_unique(out, seen, raw)
    _add_unique(out, seen, re.sub(r"[.]", " ", raw))
    _add_unique(out, seen, re.sub(r"[.'’]", "", raw))

    folded = ocr_fold(raw)
    words = folded.split()
    if words and words[0] in ORG_HEADS:
        _add_unique(out, seen, words[0])
        rest = words[1:]
        while rest and rest[0] in _STOP_NAME:
            rest = rest[1:]
        if len(rest) >= 2:
            _add_unique(out, seen, " ".join(rest))
        tail = words[1:]
        if tail:
            _add_unique(out, seen, "u " + " ".join(tail))
            _add_unique(out, seen, "u. " + " ".join(tail))

    initials = [w for w in words if w not in _STOP_NAME]
    if len(initials) >= 3:
        ac = "".join(w[0] for w in initials if w)
        if 3 <= len(ac) <= 6:
            _add_unique(out, seen, ac.upper())

    compact = folded.replace(" ", "")
    if 4 <= len(compact) <= 12 and compact != folded:
        _add_unique(out, seen, compact)
    return out


def vocero_variants(name: str) -> list[str]:
    raw = as_text(name)
    out: list[str] = []
    seen: set[str] = set()
    _add_unique(out, seen, raw)
    parts = [p for p in re.split(r"\s+", raw) if p]
    if len(parts) >= 3:
        _add_unique(out, seen, f"{parts[0]} {parts[-1]}")
        _add_unique(out, seen, " ".join(parts[-2:]))
    return out


def focus_names(
    marca: str,
    aliases: Sequence[str] | None = None,
    voceros: Sequence[str] | None = None,
) -> list[str]:
    """Marca + alias + voceros y sus variantes: misma entidad de foco."""
    out: list[str] = []
    seen: set[str] = set()
    for item in name_variants(marca):
        _add_unique(out, seen, item)
    for alias in aliases or []:
        for item in name_variants(alias):
            _add_unique(out, seen, item)
    for vocero in voceros or []:
        for item in vocero_variants(vocero):
            _add_unique(out, seen, item)
            for variant in name_variants(item):
                _add_unique(out, seen, variant)
    return out


def _name_in_blob(name: str, blob: str) -> bool:
    folded_name = ocr_fold(name)
    if len(folded_name) < 3:
        return False
    if folded_name in blob:
        return True
    compact = folded_name.replace(" ", "")
    blob_c = blob.replace(" ", "")
    if len(compact) >= 4 and compact in blob_c:
        return True
    if 3 <= len(compact) <= 5 and " " not in folded_name:
        return re.search(rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])", blob) is not None
    return False


def _hit_names(blob: str, names: Sequence[str]) -> list[str]:
    return [n for n in names if _name_in_blob(n, blob)]


def _split_sentences(text: str) -> list[str]:
    """Parte oraciones sin romper siglas de una letra («U. de Antioquia»)."""
    text = as_text(text)
    if not text:
        return []
    protected = re.sub(r"\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])\.(?=\s)", r"\1·", text)
    return [s.replace("·", ".").strip() for s in re.split(r"(?<=[.!?\n;])\s+", protected) if s.strip()]


def infer_focus_tono(
    titulo: str,
    resumen: str,
    marca: str,
    aliases: Sequence[str],
    voceros: Sequence[str],
) -> str | None:
    """Heurística local: Positivo si el foco es agente de gestión/logro;
    Negativo si la crítica apunta al foco; None si no hay vínculo evaluativo.
    """
    names = focus_names(marca, aliases, voceros)
    blob = ocr_fold(f"{titulo}. {resumen}")
    if not blob or not _hit_names(blob, names):
        return None

    text = as_text(titulo) + ". " + as_text(resumen)
    sentences = _split_sentences(text)
    units = [text]
    for sent in sentences:
        if sent and sent not in units:
            units.append(sent)
    if as_text(titulo) and ocr_fold(as_text(titulo)) not in {ocr_fold(u) for u in units}:
        units.insert(1, as_text(titulo))

    saw_gestion = False
    saw_critica = False
    locative_only = True

    for sent in units:
        folded = ocr_fold(sent)
        hits = _hit_names(folded, names)
        if not hits:
            continue
        words = folded.split()
        wset = set(words)
        has_gestion = bool(wset & GESTION_VERBS) or bool(wset & GESTION_NOUNS)
        has_neg = bool(wset & NEG_HEADS) or any(
            cue in folded for cue in ("contra ", "en contra", "denuncia a", "quejas contra")
        )
        entity_is_locative = any(_entity_is_locative(folded, ocr_fold(n)) for n in hits)
        if has_neg:
            saw_critica = True
            locative_only = False
        if has_gestion and not entity_is_locative:
            saw_gestion = True
            locative_only = False
        if not entity_is_locative:
            locative_only = False

    if saw_critica:
        return "Negativo"
    if saw_gestion:
        return "Positivo"
    if locative_only:
        return None
    return None


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
    return bool(_hit_names(blob, focus_names(marca, aliases, voceros)))


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
    names = focus_names(marca, aliases, voceros)
    if not resumen:
        return titulo[:220] if _hit_names(ocr_fold(titulo), names) else ""
    sentences = _split_sentences(resumen)
    picked = []
    for i, sent in enumerate(sentences):
        nf = ocr_fold(sent)
        if _hit_names(nf, names):
            block = sent
            if len(sent.split()) < 12 and i + 1 < len(sentences):
                block = f"{sent} {sentences[i + 1]}"
            if block not in picked:
                picked.append(block)
    if picked:
        text = " ".join(picked)
        if titulo and ocr_fold(titulo) not in ocr_fold(text):
            if _hit_names(ocr_fold(titulo), names):
                text = f"{titulo}. {text}"
        return text[:max_chars]
    if _hit_names(ocr_fold(f"{titulo} {resumen}"), names):
        return f"{titulo}. {resumen[:400]}".strip(" .")[:max_chars]
    return ""
