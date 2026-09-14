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

# Subtema: frase corta (3–5 palabras). El cuerpo se lee en tramo sustancial.
MAX_SUBTEMA_WORDS = 5
MIN_SUBTEMA_WORDS = 3
DEFAULT_BODY_CHARS = 7000


def as_text(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "null", "nat"}:
        return ""
    return s


def strip_controls(s: str) -> str:
    return "".join(ch for ch in as_text(s) if (ch >= " " or ch in "\n\t") and ch not in "\ufffe\uffff")


def prepare_article(text: str) -> str:
    """CuerpoEs completo: conserva párrafos; limpia controles y saltos de carro."""
    t = strip_controls(as_text(text))
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\t\xa0]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()


def normalize_body_text(text: str, max_chars: int | None = DEFAULT_BODY_CHARS) -> str:
    """Une saltos de maquetación y toma un tramo sustancial del artículo."""
    t = prepare_article(text)
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r" {2,}", " ", t).strip()
    if max_chars and max_chars > 0 and len(t) > max_chars:
        cut = t[:max_chars]
        if " " in cut:
            cut = cut.rsplit(" ", 1)[0]
        t = cut
    return t


def first_content_line(text: str) -> str:
    """Primera línea del cuerpo antes de un salto, sin usarla como subtema."""
    t = strip_controls(as_text(text)).replace("\r\n", "\n").replace("\r", "\n")
    if not t:
        return ""
    return re.sub(r"\s+", " ", t.split("\n", 1)[0]).strip(" .;:-")


def same_folded_phrase(a: str, b: str) -> bool:
    fa, fb = ocr_fold(a), ocr_fold(b)
    if not fa or not fb:
        return False
    return fa == fb


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
    "participo", "participa", "participaron",
    "organizo", "organiza", "organizaron",
    "convoco", "convoca", "convocaron",
    "encabezo", "encabeza",
    "presidio", "preside",
    "dialogo", "dialoga",
    "compromete", "comprometio", "comprometieron",
    "reunio", "reune", "reunieron",
}

GESTION_NOUNS = {
    "convenio", "convenios", "acuerdo", "acuerdos", "alianza", "alianzas",
    "ranking", "acreditacion", "beca", "becas", "infraestructura",
    "inversion", "inversiones", "programa", "programas", "obra", "obras",
    "graduacion", "reconocimiento", "reconocimientos", "gestion", "gestiones",
    "campus", "laboratorios", "investigacion", "convocatoria",
    "encuentro", "encuentros", "evento", "eventos", "compromiso", "compromisos",
    "entrega", "entregas", "lanzamiento", "lanzamientos", "avance", "avances",
    "reunion", "reuniones", "ceremonia", "jornada", "feria", "visita", "visitas",
    "conversatorio", "pacto", "pactos", "dialogo", "mesa",
}

NEG_HEADS = {
    "denuncia", "denuncian", "denuncio", "denunciaron",
    "queja", "quejas", "reclamo", "reclaman", "reclamaron",
    "protesta", "protestan", "protestaron", "manifestacion",
    "sancion", "sancionan", "sanciono", "sancionaron",
    "investigan",
    "corrupcion", "irregularidades", "escandalo",
    "demanda", "demandan", "demandaron",
    "critica", "critican", "criticas",
    "cuestionan", "cuestiono",
    "rechazo", "rechazan",
    "polemica",
    "retraso", "retrasos", "falla", "fallas",
    "abandono", "negligencia",
}

# Colaboración / coautoría en un estudio: el FOCO no es el evaluado.
COLLAB_PHRASES = (
    "con la colaboracion de",
    "en colaboracion con",
    "colaboracion de",
    "colaboracion entre",
    "en coautoria",
    "coautoria con",
    "coautoria de",
    "con el apoyo de",
    "con el respaldo de",
    "elaborado con la colaboracion",
    "elaborado con colaboracion",
    "participo en el estudio",
    "participaron en el estudio",
    "participacion en el estudio",
    "participo en el informe",
    "participaron en el informe",
    "participo en la investigacion",
    "participaron en la investigacion",
)

STUDY_NOUNS = {
    "estudio", "informe", "encuesta", "medicion", "investigacion",
}

PRAISE_CUES = {
    "destaco", "destaca", "destacaron",
    "exalto", "exalta",
    "elogio", "elogia",
    "agradecio", "agradece",
    "reconocio", "reconoce",
    "protagonismo", "liderazgo",
}

DIRECTED_NEG_CUES = (
    "contra ",
    "en contra",
    "denuncia a",
    "quejas contra",
    "investigan a",
    "bajo investigacion",
)

# Recorte de oración del cuerpo (no una etiqueta de 3–5 palabras).
MAX_SUBTEMA_EXTRACT_WORDS = 8
EXTRACT_SPAN_WORDS = 4

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
    """Parte oraciones reales; un salto de maquetación no cuenta como fin."""
    text = normalize_body_text(text, max_chars=None)
    if not text:
        return []
    protected = re.sub(r"\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ])\.(?=\s)", r"\1·", text)
    return [s.replace("·", ".").strip() for s in re.split(r"(?<=[.!?])\s+", protected) if s.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Párrafos del CuerpoEs; líneas cortas de maquetación se unen."""
    article = prepare_article(text)
    if not article:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", article) if p.strip()]
    if len(paras) <= 1 and "\n" in article:
        lines = [ln.strip() for ln in article.split("\n") if ln.strip()]
        if not lines:
            return []
        if max(len(ln) for ln in lines) < 90:
            return [" ".join(lines)]
        return lines
    return paras or ([article] if article else [])


def _passage_units(text: str) -> list[str]:
    """Oraciones por párrafo: unidades para ventanas alrededor de la marca."""
    units: list[str] = []
    for para in _split_paragraphs(text):
        sents = _split_sentences(para)
        if sents:
            units.extend(sents)
        elif para.strip():
            units.append(para.strip())
    return units


def _is_collaborator_sentence(folded: str, wset: set[str]) -> bool:
    if any(p in folded for p in COLLAB_PHRASES):
        return True
    if any(stem in folded for stem in ("participo", "participa", "participaron", "participacion")):
        if wset & STUDY_NOUNS:
            return True
    return False


def _has_praise(folded: str, wset: set[str]) -> bool:
    if wset & PRAISE_CUES:
        return True
    return "papel de" in folded or "rol de" in folded or "exalt" in folded


def _has_directed_critica(folded: str, wset: set[str]) -> bool:
    if any(cue in folded for cue in DIRECTED_NEG_CUES):
        return True
    return bool(wset & NEG_HEADS)


def infer_focus_tono(
    titulo: str,
    resumen: str,
    marca: str,
    aliases: Sequence[str],
    voceros: Sequence[str],
) -> str | None:
    """Heurística local: Positivo si el foco es agente de gestión/logro;
    Negativo si la crítica apunta al foco; None si no hay vínculo evaluativo.

    Se juzga por oración (no por la bolsa de palabras de toda la nota), para
    no pintar el sentimiento del tema (desempleo, crimen…) sobre la marca.
    """
    names = focus_names(marca, aliases, voceros)
    blob = ocr_fold(f"{titulo}. {resumen}")
    if not blob or not _hit_names(blob, names):
        return None

    text = as_text(titulo) + ". " + as_text(resumen)
    sentences = _split_sentences(text)
    units: list[str] = []
    title = as_text(titulo)
    if title:
        units.append(title)
    for sent in sentences:
        if sent and ocr_fold(sent) not in {ocr_fold(u) for u in units}:
            units.append(sent)

    saw_gestion = False
    saw_critica = False
    saw_collab = False
    saw_praise = False
    locative_only = True

    for sent in units:
        folded = ocr_fold(sent)
        hits = _hit_names(folded, names)
        if not hits:
            continue
        words = folded.split()
        wset = set(words)
        entity_is_locative = any(_entity_is_locative(folded, ocr_fold(n)) for n in hits)
        collab = _is_collaborator_sentence(folded, wset)
        critica = _has_directed_critica(folded, wset)
        praise = _has_praise(folded, wset)
        has_gestion = bool(wset & GESTION_VERBS) or bool(wset & GESTION_NOUNS)

        if not entity_is_locative:
            locative_only = False
        if collab:
            saw_collab = True
            if praise:
                saw_praise = True
            if critica and any(cue in folded for cue in DIRECTED_NEG_CUES):
                saw_critica = True
            continue
        if critica:
            saw_critica = True
        if has_gestion and not entity_is_locative:
            saw_gestion = True
        if praise:
            saw_praise = True

    if saw_critica:
        return "Negativo"
    if saw_collab and not saw_praise and not saw_gestion:
        return None
    if saw_praise or saw_gestion:
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
    """True si el subtema es un recorte crudo del titular (no una síntesis)."""
    if not subtema or not titulo:
        return False
    sw = ocr_fold(subtema).split()
    tw = ocr_fold(titulo).split()
    if not sw or not tw:
        return False
    if tw[: len(sw)] == sw and len(tw) > len(sw) + 2 and len(sw) >= 4:
        return True
    return False


def looks_like_title_or_lead(subtema: str, titulo: str, cuerpo: str) -> bool:
    """El subtema no puede ser el título ni la primera línea del cuerpo."""
    if not subtema:
        return False
    if same_folded_phrase(subtema, titulo):
        return True
    lead = first_content_line(cuerpo)
    if lead and same_folded_phrase(subtema, lead):
        return True
    return looks_like_title_scrap(subtema, titulo)


def _is_consecutive_span(needle: Sequence[str], hay: Sequence[str]) -> bool:
    n = len(needle)
    if n < EXTRACT_SPAN_WORDS or len(hay) < n:
        return False
    needle_l = list(needle)
    for i in range(len(hay) - n + 1):
        if list(hay[i : i + n]) == needle_l:
            return True
    return False


def looks_like_body_extract(subtema: str, titulo: str, cuerpo: str) -> bool:
    """True si el subtema es un recorte/cita de una oración cruda, no una etiqueta."""
    phrase = as_text(subtema)
    words = phrase.split()
    if not words:
        return False
    if len(words) > MAX_SUBTEMA_EXTRACT_WORDS:
        return True
    folded = ocr_fold(phrase)
    fw = folded.split()
    if len(fw) < EXTRACT_SPAN_WORDS:
        return False
    sources = [as_text(titulo), first_content_line(cuerpo)]
    sources.extend(_split_sentences(cuerpo)[:48])
    for src in sources:
        sw = ocr_fold(src).split()
        if not sw:
            continue
        if _is_consecutive_span(fw, sw):
            return True
        src_fold = ocr_fold(src)
        if folded in src_fold and len(sw) >= len(fw) + 2:
            return True
    return False


def strip_brand_mentions(
    phrase: str,
    marca: str,
    aliases: Sequence[str] | None = None,
) -> str:
    """Quita marca y alias del subtema: el ángulo de la nota, no la etiqueta."""
    words = [w for w in re.sub(r"\s+", " ", as_text(phrase)).split() if w]
    if not words:
        return ""
    variants: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for item in [marca, *(aliases or [])]:
        for v in name_variants(item) + [as_text(item)]:
            fv = ocr_fold(v).split()
            key = tuple(fv)
            if not fv or key in seen:
                continue
            if len(fv) == 1 and (len(fv[0]) < 3 or fv[0] in ORG_HEADS):
                continue
            seen.add(key)
            variants.append(fv)
    variants.sort(key=len, reverse=True)
    folded_words = [ocr_fold(w) for w in words]
    drop = [False] * len(words)
    for var in variants:
        n = len(var)
        i = 0
        while i <= len(words) - n:
            if folded_words[i : i + n] == var and not any(drop[i : i + n]):
                for j in range(i, i + n):
                    drop[j] = True
                i += n
            else:
                i += 1
    kept = [w for w, d in zip(words, drop) if not d]
    while kept and fold_text(kept[0]) in DANGLING:
        kept.pop(0)
    while kept and fold_text(kept[0]) in ORG_HEADS:
        kept.pop(0)
        while kept and fold_text(kept[0]) in DANGLING:
            kept.pop(0)
    kept = strip_dangling(kept)
    return sentence_case(" ".join(kept))


def _clip_subtema_words(words: Sequence[str]) -> list[str]:
    clipped = [w for w in words if w]
    if len(clipped) > MAX_SUBTEMA_WORDS:
        clipped = clipped[:MAX_SUBTEMA_WORDS]
    return strip_dangling(clipped)


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


def _phrase_is_unusable(phrase: str, titulo: str, cuerpo: str) -> bool:
    if not phrase:
        return True
    if looks_like_collage(phrase) or looks_like_title_or_lead(phrase, titulo, cuerpo):
        return True
    if looks_like_body_extract(phrase, titulo, cuerpo):
        return True
    if len(phrase.split()) > MAX_SUBTEMA_EXTRACT_WORDS:
        return True
    return False


def _finalize_subtema(
    phrase: str,
    *,
    titulo: str,
    resumen: str,
    marca: str,
    aliases: Sequence[str] | None,
) -> str:
    phrase = strip_brand_mentions(phrase, marca, aliases)
    words = _clip_subtema_words(phrase.split())
    return sentence_case(" ".join(words))


def clean_subtema(
    raw: str,
    *,
    titulo: str = "",
    resumen: str = "",
    marca: str = "",
    aliases: Sequence[str] | None = None,
) -> str:
    text = strip_controls(raw)
    text = re.sub(r"[\"'«»“”‘’]", "", text)
    text = re.sub(r"[:;|/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-")
    raw_joined = " ".join(text.split())
    raw_is_extract = (
        len(raw_joined.split()) > MAX_SUBTEMA_EXTRACT_WORDS
        or looks_like_body_extract(raw_joined, titulo, resumen)
    )
    phrase = ""
    if not raw_is_extract:
        phrase = _finalize_subtema(
            raw_joined,
            titulo=titulo,
            resumen=resumen,
            marca=marca,
            aliases=aliases,
        )
    brand_w = set(content_words(marca))
    for alias in aliases or []:
        brand_w |= set(content_words(alias))
    phrase_w = set(content_words(phrase))
    needs_fallback = (
        not phrase
        or _phrase_is_unusable(phrase, titulo, resumen)
        or len(phrase.split()) < MIN_SUBTEMA_WORDS
        or (brand_w and phrase_w and phrase_w <= brand_w)
    )
    if needs_fallback:
        phrase = _fallback_subtema(resumen, titulo, marca, aliases)
        phrase = _finalize_subtema(
            phrase, titulo=titulo, resumen=resumen, marca=marca, aliases=aliases
        )
    if _phrase_is_unusable(phrase, titulo, resumen):
        alt = _fallback_subtema(resumen, titulo, marca, aliases, offset=4)
        alt = _finalize_subtema(
            alt, titulo=titulo, resumen=resumen, marca=marca, aliases=aliases
        )
        if alt and not _phrase_is_unusable(alt, titulo, resumen):
            phrase = alt
    return phrase or "Hecho informativo"


def _source_tokens(resumen: str, titulo: str) -> tuple[list[str], str]:
    raw = as_text(resumen) or as_text(titulo)
    lead = first_content_line(raw)
    rest = ""
    body_nl = raw.replace("\r\n", "\n").replace("\r", "\n")
    if "\n" in body_nl:
        rest = body_nl.split("\n", 1)[1]
    if len(ocr_fold(rest)) >= 24:
        source = normalize_body_text(rest, max_chars=DEFAULT_BODY_CHARS)
    else:
        source = normalize_body_text(raw, max_chars=DEFAULT_BODY_CHARS)
    source = re.sub(
        r"^(?:imagenes|en imagenes|fotos|video|en vivo)\s*\|\s*",
        "",
        source,
        flags=re.I,
    )
    tokens = re.sub(r"[,.;:!?¿¡\"'()\[\]{}]", " ", source).split()
    return tokens, lead


_ANGLE_HEADS = (
    ("informe", "Informe"),
    ("estudio", "Estudio"),
    ("encuesta", "Encuesta"),
    ("entrega", "Entrega"),
    ("lanzamiento", "Lanzamiento"),
    ("encuentro", "Encuentro"),
    ("convenio", "Convenio"),
    ("protesta", "Protesta"),
    ("sancion", "Sanción"),
    ("becas", "Becas"),
    ("beca", "Becas"),
    ("desempleo", "Desempleo"),
    ("ranking", "Ranking"),
    ("acreditacion", "Acreditación"),
    ("diplomado", "Diplomados"),
    ("pae", "PAE"),
)

_ANALYTICAL_SKIP = {
    "llega", "llego", "segun", "nuevo", "nueva", "nuevos", "nuevas",
    "mil", "ciento", "asi", "revela", "revelo", "otro", "otra",
    "titular", "distinto", "distinta", "sobre", "anuncia", "anuncio",
    "hay", "mas", "mitad", "ano", "anos", "porcentaje",
}


def _analytical_fallback(
    resumen: str,
    titulo: str,
    marca: str,
    aliases: Sequence[str] | None = None,
) -> str:
    """Etiqueta nominal 3–5 palabras; no recorta una oración del cuerpo."""
    skip = brand_tokens([marca], aliases or []) | _ANALYTICAL_SKIP | STOPWORDS
    blob_words = set(fold_text(f"{titulo} {resumen}").split())
    head = ""
    for key, label in _ANGLE_HEADS:
        if key in blob_words:
            head = label
            break

    def take_words(source: str, limit: int) -> list[str]:
        kept: list[str] = []
        seen: set[str] = set()
        for w in re.sub(r"[,.;:!?¿¡\"'()\[\]%]", " ", as_text(source)).split():
            fw = fold_text(w)
            if not fw or fw in skip or fw.isdigit() or fw in ORG_HEADS:
                continue
            if head and fw == fold_text(head):
                continue
            if fw in GESTION_VERBS:
                continue
            if fw in seen:
                continue
            seen.add(fw)
            kept.append(w if (w.isupper() and 2 <= len(w) <= 6) else w.lower())
            if len(kept) >= limit:
                break
        return kept

    rest = take_words(titulo, 2 if head else 4)
    if len(rest) < (2 if head else MIN_SUBTEMA_WORDS):
        for w in take_words(resumen, 4):
            if fold_text(w) not in {fold_text(x) for x in rest}:
                rest.append(w)
            if len(rest) >= (2 if head else 4):
                break

    if head:
        parts = [head]
        if rest:
            parts.append("de")
            parts.extend(rest)
    else:
        parts = list(rest)
    words = strip_dangling(parts)
    if len(words) > MAX_SUBTEMA_WORDS:
        words = strip_dangling(words[:MAX_SUBTEMA_WORDS])
    phrase = sentence_case(" ".join(words))
    return strip_brand_mentions(phrase, marca, aliases)


def _fallback_subtema(
    resumen: str,
    titulo: str,
    marca: str,
    aliases: Sequence[str] | None = None,
    offset: int = 0,
) -> str:
    tokens, lead = _source_tokens(resumen, titulo)
    skip = brand_tokens([marca], aliases or [])
    avoid = {ocr_fold(titulo), ocr_fold(lead)} - {""}

    def build(start: int) -> str:
        kept: list[str] = []
        i = max(0, start)
        while i < len(tokens) and len(kept) < MAX_SUBTEMA_WORDS:
            w = tokens[i]
            i += 1
            if fold_text(w) in skip and len(kept) == 0:
                continue
            kept.append(w)
        kept = strip_dangling(kept)
        while kept and fold_text(kept[0]) in DANGLING | ORG_HEADS:
            kept.pop(0)
            kept = strip_dangling(kept)
        phrase = sentence_case(" ".join(kept))
        return strip_brand_mentions(phrase, marca, aliases)

    analytical = _analytical_fallback(resumen, titulo, marca, aliases)
    if (
        analytical
        and len(analytical.split()) >= MIN_SUBTEMA_WORDS
        and ocr_fold(analytical) not in avoid
        and not looks_like_collage(analytical)
        and not looks_like_title_scrap(analytical, titulo)
        and not looks_like_title_or_lead(analytical, titulo, resumen)
        and not looks_like_body_extract(analytical, titulo, resumen)
    ):
        return analytical

    for start in (offset, offset + 3, offset + 6, 1, 5, 8):
        phrase = build(start)
        words = _clip_subtema_words(phrase.split())
        phrase = sentence_case(" ".join(words))
        if (
            phrase
            and len(phrase.split()) >= MIN_SUBTEMA_WORDS
            and ocr_fold(phrase) not in avoid
            and not looks_like_collage(phrase)
            and not looks_like_title_scrap(phrase, titulo)
            and not looks_like_body_extract(phrase, titulo, resumen)
        ):
            return phrase

    title_toks = re.sub(r"[,.;:!?¿¡\"']", " ", as_text(titulo)).split()
    kept = []
    for w in title_toks:
        if fold_text(w) in skip:
            continue
        kept.append(w)
        if len(kept) >= MAX_SUBTEMA_WORDS:
            break
    kept = strip_dangling(kept)
    phrase = strip_brand_mentions(sentence_case(" ".join(kept)), marca, aliases)
    words = _clip_subtema_words(phrase.split())
    phrase = sentence_case(" ".join(words))
    if (
        phrase
        and ocr_fold(phrase) not in avoid
        and not looks_like_body_extract(phrase, titulo, resumen)
        and not looks_like_title_or_lead(phrase, titulo, resumen)
    ):
        return phrase
    return analytical or phrase


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
    max_chars: int = 4000,
) -> str:
    """Ventanas de oraciones/párrafos alrededor de marca, alias y voceros.

    Recorre el CuerpoEs completo (saltos de línea permitidos). No se queda
    en la primera línea ni recorta el artículo antes de buscar menciones.
    Esas ventanas alimentan el TONO (cómo se trata al FOCO), no el subtema.
    """
    titulo = as_text(titulo)
    article = prepare_article(resumen)
    names = focus_names(marca, aliases, voceros)
    if not article:
        return titulo[:280] if _hit_names(ocr_fold(titulo), names) else ""

    units = _passage_units(article)
    if not units:
        units = [normalize_body_text(article, max_chars=None)] if article else []

    hit_idx = [i for i, unit in enumerate(units) if _hit_names(ocr_fold(unit), names)]
    if not hit_idx:
        if _hit_names(ocr_fold(titulo), names):
            return titulo[:280]
        return ""

    ranges: list[list[int]] = []
    n = len(units)
    for i in hit_idx:
        pad_after = 2 if len(units[i].split()) < 10 else 1
        lo = max(0, i - 1)
        hi = min(n, i + 1 + pad_after)
        if ranges and lo <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])

    blocks = [" ".join(units[lo:hi]).strip() for lo, hi in ranges]
    text = "\n".join(b for b in blocks if b)
    if titulo and _hit_names(ocr_fold(titulo), names):
        if ocr_fold(titulo) not in ocr_fold(text):
            text = f"{titulo}.\n{text}"
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text.strip()
