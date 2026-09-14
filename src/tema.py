"""Agrupa subtemas ya finalizados en un tema_AI más general.

No reescribe subtema_AI: solo lee las cadenas ya limpias y les asigna un
tema un poco más amplio (máx. 4 palabras, sentence case, sin marca).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Sequence

from src.group import _DSU
from src.normalize import (
    as_text,
    content_words,
    fold_text,
    ocr_fold,
    sentence_case,
    strip_brand_mentions,
    strip_dangling,
)

TEMA_MIN_WORDS = 2
TEMA_MAX_WORDS = 4

# Disparadores (texto plegado) → etiqueta un poco más general que el subtema.
TEMA_FAMILIES: list[tuple[frozenset[str], str]] = [
    (frozenset({"beca", "becas", "sostenimiento", "subsidio", "subsidios"}), "Becas y apoyos estudiantiles"),
    (frozenset({"diplomado", "diplomados", "formacion", "capacitacion"}), "Formación y diplomados"),
    (frozenset({"obra", "obras", "laboratorio", "laboratorios", "infraestructura", "bloque"}), "Obras e infraestructura"),
    (frozenset({"convenio", "convenios", "movilidad", "alianza", "alianzas", "cooperacion"}), "Convenios y alianzas"),
    (frozenset({"ranking", "qs", "acreditacion", "puesto"}), "Rankings y acreditación"),
    (frozenset({"pae", "alimentacion", "escolar"}), "Alimentación escolar"),
    (frozenset({"protesta", "protestas", "matricula", "alza"}), "Protestas y matrícula"),
    (frozenset({"sancion", "sanciones", "exsecretario", "procuraduria"}), "Sanciones institucionales"),
    (frozenset({"denuncia", "denuncias", "irregularidades", "contratacion"}), "Denuncias de contratación"),
    (frozenset({"encuentro", "encuentros", "evento", "eventos", "reunion", "reuniones"}), "Encuentros y eventos"),
    (frozenset({"compromiso", "compromisos", "pacto", "pactos"}), "Compromisos institucionales"),
    (frozenset({"foro", "foros", "conversatorio", "campus"}), "Foros y eventos"),
    (frozenset({"variante", "vial", "viales", "carretera"}), "Obras viales"),
    (frozenset({"gallinas", "ponedoras", "alimentaria"}), "Seguridad alimentaria"),
    (frozenset({"investigacion", "ciencia", "cientifica"}), "Investigación y ciencia"),
    (frozenset({"graduacion", "grados", "egresados"}), "Grados y egresados"),
    (frozenset({"inversion", "inversiones", "presupuesto", "recursos"}), "Inversión y recursos"),
    (frozenset({"convocatoria", "convocatorias"}), "Convocatorias abiertas"),
    (frozenset({"visita", "visitas"}), "Visitas institucionales"),
    (frozenset({"lanzamiento", "lanzamientos", "programa", "programas"}), "Programas y lanzamientos"),
    (frozenset({"suicidio", "suicidios", "jovenes"}), "Informes de salud"),
    (frozenset({"desempleo", "empleo", "ocupacion"}), "Informes de empleo"),
    (frozenset({"canal", "dique", "proyecto"}), "Proyectos de infraestructura"),
]

ACTION_HEADS = {
    "entrega", "entregas", "lanzamiento", "lanzamientos", "avance", "avances",
    "aprobacion", "inicio", "denuncia", "sancion", "revision", "impulso",
    "firma", "inauguracion", "encuentro", "evento", "gestion", "informe",
}

SUBTEMA_RATIO = 0.76
SUBTEMA_LOOSE_RATIO = 0.62


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def subtemas_similar(a: str, b: str) -> bool:
    fa, fb = ocr_fold(a), ocr_fold(b)
    if not fa or not fb:
        return False
    if fa == fb or fa in fb or fb in fa:
        if min(len(fa), len(fb)) >= 12:
            return True
    wa, wb = set(content_words(a)), set(content_words(b))
    overlap = wa & wb
    if len(overlap) >= 2:
        return True
    if len(overlap) >= 1 and _ratio(fa, fb) >= SUBTEMA_LOOSE_RATIO:
        return True
    if _ratio(fa, fb) >= SUBTEMA_RATIO:
        return True
    return False


def cluster_subtema_indices(subtemas: Sequence[str]) -> list[list[int]]:
    n = len(subtemas)
    dsu = _DSU(n)
    exact: dict[str, int] = {}
    for i, sub in enumerate(subtemas):
        key = ocr_fold(sub)
        if not key:
            continue
        if key in exact:
            dsu.union(i, exact[key])
        else:
            exact[key] = i

    inv: dict[str, list[int]] = defaultdict(list)
    words_list: list[set[str]] = []
    for i, sub in enumerate(subtemas):
        ws = set(content_words(sub))
        words_list.append(ws)
        for w in ws:
            inv[w].append(i)

    for i, ws in enumerate(words_list):
        hits: Counter[int] = Counter()
        for w in ws:
            for j in inv.get(w, ()):
                if j > i:
                    hits[j] += 1
        for j, _c in hits.items():
            if subtemas_similar(subtemas[i], subtemas[j]):
                dsu.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return [sorted(v) for v in groups.values()]


def _clip_tema(phrase: str) -> str:
    words = strip_dangling(as_text(phrase).split())
    if len(words) > TEMA_MAX_WORDS:
        words = strip_dangling(words[:TEMA_MAX_WORDS])
    return sentence_case(" ".join(words))


def _family_label(phrases: Sequence[str]) -> str | None:
    bag: Counter[str] = Counter()
    for p in phrases:
        bag.update(content_words(p))
    if not bag:
        return None
    scored: list[tuple[int, int, str]] = []
    for triggers, label in TEMA_FAMILIES:
        hit = sum(bag[t] for t in triggers if t in bag)
        if hit:
            scored.append((hit, len(triggers), label))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def broaden_subtema(subtema: str) -> str:
    """Tema un poco más general que un subtema suelto (sin hermanos)."""
    phrase = as_text(subtema)
    family = _family_label([phrase])
    if family:
        return family
    words = strip_dangling(phrase.split())
    if words and fold_text(words[0]) in ACTION_HEADS and len(words) >= 3:
        rest = words[1:]
        if rest and fold_text(rest[0]) in {"de", "del", "la", "el"}:
            rest = rest[1:]
            if rest and fold_text(rest[0]) in {"de", "del", "la", "el"}:
                rest = rest[1:]
        rest = strip_dangling(rest)
        if len(rest) >= 2:
            broadened = _clip_tema(" ".join(rest[:TEMA_MAX_WORDS]))
            if ocr_fold(broadened) != ocr_fold(phrase) and len(broadened.split()) >= TEMA_MIN_WORDS:
                return broadened
        if len(rest) == 1:
            return _clip_tema(f"{rest[0]} institucional")
    cw = content_words(phrase)
    if len(cw) >= 2:
        return _clip_tema(f"{cw[0]} y {cw[1]}")
    if cw:
        return _clip_tema(f"{cw[0]} institucional")
    return _clip_tema(phrase) or "Hecho informativo"


def _shared_tema(phrases: Sequence[str]) -> str:
    family = _family_label(phrases)
    if family:
        return family
    if len(phrases) == 1:
        return broaden_subtema(phrases[0])
    counts: Counter[str] = Counter()
    for p in phrases:
        counts.update(content_words(p))
    common = [w for w, _n in counts.most_common(4) if w]
    if len(common) >= 2:
        return _clip_tema(f"{common[0]} y {common[1]}")
    if common:
        return broaden_subtema(max(phrases, key=len))
    return broaden_subtema(phrases[0])


def clean_tema(
    raw: str,
    *,
    marca: str = "",
    aliases: Sequence[str] | None = None,
) -> str:
    phrase = strip_brand_mentions(as_text(raw), marca, aliases)
    phrase = _clip_tema(phrase)
    if len(phrase.split()) < TEMA_MIN_WORDS:
        extra = broaden_subtema(raw or phrase)
        extra = strip_brand_mentions(extra, marca, aliases)
        phrase = _clip_tema(extra) or phrase
    return phrase or "Hecho informativo"


def assign_temas(
    subtemas: Sequence[str],
    *,
    marca: str = "",
    aliases: Sequence[str] | None = None,
) -> list[str]:
    """Un tema_AI por fila: el mismo en cada clúster de subtemas parecidos.

    Recibe subtemas ya finalizados. No los modifica.
    """
    cleaned = [
        strip_brand_mentions(as_text(s), marca, aliases) or as_text(s)
        for s in subtemas
    ]
    clusters = cluster_subtema_indices(cleaned)
    out = [""] * len(cleaned)
    for members in clusters:
        label = clean_tema(
            _shared_tema([cleaned[i] for i in members]),
            marca=marca,
            aliases=aliases,
        )
        for i in members:
            out[i] = label
    return out
