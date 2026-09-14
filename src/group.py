"""Agrupación local OCR-aware y propagación Positivo-first."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from src.normalize import (
    TONOS,
    clean_subtema,
    content_words,
    fold_text,
    ocr_fold,
    strip_dangling,
)

TITLE_RATIO = 0.86
RESUMEN_RATIO = 0.82
NGRAM_RATIO = 0.78
MIN_NGRAMS = 24
MIN_TITLE_OVERLAP = 3


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    r1 = SequenceMatcher(None, a, b).ratio()
    sa = " ".join(sorted(a.split()))
    sb = " ".join(sorted(b.split()))
    r2 = SequenceMatcher(None, sa, sb).ratio()
    return max(r1, r2)


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    compact = ocr_fold(text).replace(" ", "")
    if len(compact) < n:
        return set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _ngram_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def titles_similar(a: str, b: str, exclude: set[str] | None = None) -> bool:
    fa, fb = ocr_fold(a), ocr_fold(b)
    if not fa or not fb:
        return False
    if fa == fb or fa in fb or fb in fa:
        if min(len(fa), len(fb)) >= 18:
            return True
    if _ratio(fa, fb) >= TITLE_RATIO:
        return True
    wa = set(content_words(a)) - (exclude or set())
    wb = set(content_words(b)) - (exclude or set())
    overlap = wa & wb
    if len(overlap) >= 4:
        return True
    if len(overlap) >= MIN_TITLE_OVERLAP and _ratio(fa, fb) >= 0.72:
        return True
    return False


def resumenes_similar(a: str, b: str) -> bool:
    fa, fb = ocr_fold(a), ocr_fold(b)
    if len(fa) < 40 or len(fb) < 40:
        return False
    if _ratio(fa[:500], fb[:500]) >= RESUMEN_RATIO:
        return True
    ga, gb = _char_ngrams(a[:800]), _char_ngrams(b[:800])
    if min(len(ga), len(gb)) >= MIN_NGRAMS and _ngram_overlap(ga, gb) >= NGRAM_RATIO:
        return True
    return False


def cluster_indices(
    titles: Sequence[str],
    resumenes: Sequence[str],
    exclude_tokens: Iterable[str] | None = None,
) -> list[list[int]]:
    n = len(titles)
    dsu = _DSU(n)
    exclude = {fold_text(t) for t in (exclude_tokens or []) if t}
    exclude |= {w for tok in list(exclude) for w in tok.split()}

    exact: dict[str, int] = {}
    prefixes: dict[str, list[int]] = defaultdict(list)
    for i, title in enumerate(titles):
        key = ocr_fold(title)
        if len(key) >= 12:
            if key in exact:
                dsu.union(i, exact[key])
            else:
                exact[key] = i
        pref = key[:24]
        if len(pref) >= 18:
            prefixes[pref].append(i)
    for idxs in prefixes.values():
        root = idxs[0]
        for j in idxs[1:]:
            if titles_similar(titles[root], titles[j], exclude):
                dsu.union(root, j)

    inv: dict[str, list[int]] = defaultdict(list)
    words_list = []
    for i, title in enumerate(titles):
        ws = [w for w in content_words(title) if w not in exclude]
        words_list.append(set(ws))
        for w in set(ws):
            inv[w].append(i)

    for i, ws in enumerate(words_list):
        hits: Counter[int] = Counter()
        for w in ws:
            for j in inv.get(w, ()):
                if j > i:
                    hits[j] += 1
        for j, c in hits.items():
            if c >= 2 and titles_similar(titles[i], titles[j], exclude):
                dsu.union(i, j)

    grams_list = [_char_ngrams(r[:800]) for r in resumenes]
    ginv: dict[str, list[int]] = defaultdict(list)
    for i, grams in enumerate(grams_list):
        if len(grams) < MIN_NGRAMS:
            continue
        for g in grams:
            ginv[g].append(i)

    seen_pairs: set[tuple[int, int]] = set()
    for i, grams in enumerate(grams_list):
        if len(grams) < MIN_NGRAMS:
            continue
        hits = Counter()
        for g in grams:
            for j in ginv.get(g, ()):
                if j > i:
                    hits[j] += 1
        for j, inter in hits.items():
            if (i, j) in seen_pairs:
                continue
            seen_pairs.add((i, j))
            other = grams_list[j]
            if len(other) < MIN_NGRAMS:
                continue
            if inter / min(len(grams), len(other)) >= NGRAM_RATIO or resumenes_similar(
                resumenes[i], resumenes[j]
            ):
                dsu.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return [sorted(v) for v in groups.values()]


def positivo_first(tonos: Sequence[str]) -> str:
    vals = [canonicalize_if_needed(t) for t in tonos if canonicalize_if_needed(t) in TONOS]
    if not vals:
        return "Neutro"
    if any(t == "Positivo" for t in vals):
        return "Positivo"
    counts = Counter(vals)
    best_n = max(counts.values())
    tied = [t for t, n in counts.items() if n == best_n]
    if len(tied) == 1:
        return tied[0]
    return "Neutro"


def canonicalize_if_needed(tono: str) -> str:
    from src.normalize import canonicalize_tono

    return canonicalize_tono(tono) if tono not in TONOS else tono


def pick_best_subtema(cands: Sequence[str], *, titulo: str = "", resumen: str = "", marca: str = "") -> str:
    cleaned = []
    for c in cands:
        s = clean_subtema(c, titulo=titulo, resumen=resumen, marca=marca)
        if s:
            cleaned.append(s)
    if not cleaned:
        return clean_subtema("", titulo=titulo, resumen=resumen, marca=marca)
    counts = Counter(cleaned)
    max_n = max(counts.values())
    tied = [s for s, n in counts.items() if n == max_n]

    def score(s: str) -> tuple:
        words = strip_dangling(s.split())
        n = len(words)
        in_range = 1 if 3 <= n <= 12 else 0
        return (in_range, n, -len(s), s)

    return max(tied, key=score)


def propagate_labels(
    tonos: list[str],
    subtemas: list[str],
    titles: Sequence[str],
    resumenes: Sequence[str],
    *,
    marca: str = "",
    exclude_tokens: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Unifica subtema+tono por clúster (título O resumen) y Positivo-first.

    1) Noticias con título o resumen parecido (OCR-aware) → mismo subtema y tono.
    2) Positivo gana dentro del clúster.
    3) Positivo gana también entre filas que ya quedaron con el mismo subtema.
    """
    n = len(tonos)
    out_tono = [canonicalize_if_needed(t) for t in tonos]
    out_sub = list(subtemas)
    clusters = cluster_indices(titles, resumenes, exclude_tokens)

    for members in clusters:
        if len(members) == 1:
            i = members[0]
            out_sub[i] = clean_subtema(
                out_sub[i], titulo=titles[i], resumen=resumenes[i], marca=marca
            )
            continue
        canon_sub = pick_best_subtema(
            [out_sub[i] for i in members],
            titulo=titles[members[0]],
            resumen=max((resumenes[i] for i in members), key=lambda x: len(str(x))),
            marca=marca,
        )
        canon_tono = positivo_first([out_tono[i] for i in members])
        for i in members:
            out_sub[i] = canon_sub
            out_tono[i] = canon_tono

    by_sub: dict[str, list[int]] = defaultdict(list)
    for i, sub in enumerate(out_sub):
        key = ocr_fold(sub)
        if key:
            by_sub[key].append(i)
    for idxs in by_sub.values():
        if len(idxs) < 2:
            continue
        canon_tono = positivo_first([out_tono[i] for i in idxs])
        for i in idxs:
            out_tono[i] = canon_tono

    return out_tono, out_sub
