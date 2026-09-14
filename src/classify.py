"""Lotes JSON a OpenAI y postproceso de tono/subtema."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

from openai import OpenAI

from src.group import propagate_labels
from src.normalize import (
    as_text,
    brand_tokens,
    canonicalize_tono,
    clean_subtema,
    extract_brand_passages,
    infer_focus_tono,
    normalize_body_text,
)
from src.prompts import SYSTEM_PROMPT, build_user_prompt

ProgressFn = Callable[[float, str], None]

DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"
BODY_MAX_CHARS = 7000
PASSAGE_MAX_CHARS = 4000

# Tarifas configurables gpt-4.1-nano (USD por 1 millón de tokens).
INPUT_USD_PER_1M_TOKENS = 0.10
OUTPUT_USD_PER_1M_TOKENS = 0.40


@dataclass
class ClassifyStats:
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    batches: int = 0
    model: str = DEFAULT_MODEL

    @property
    def cost_input_usd(self) -> float:
        return (self.prompt_tokens / 1_000_000) * INPUT_USD_PER_1M_TOKENS

    @property
    def cost_output_usd(self) -> float:
        return (self.completion_tokens / 1_000_000) * OUTPUT_USD_PER_1M_TOKENS

    @property
    def cost_total_usd(self) -> float:
        return self.cost_input_usd + self.cost_output_usd


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
    inp = (prompt_tokens / 1_000_000) * INPUT_USD_PER_1M_TOKENS
    out = (completion_tokens / 1_000_000) * OUTPUT_USD_PER_1M_TOKENS
    return inp, out, inp + out


def _usage_from_response(resp: Any) -> tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt, completion


def _parse_json_content(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"resultados": data}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    raise ValueError("La respuesta del modelo no es JSON válido.")


def _index_resultados(payload: dict) -> dict[int, dict]:
    rows = payload.get("resultados") or payload.get("results") or payload.get("items") or []
    if isinstance(payload, dict) and not rows and "tono" in payload:
        rows = [payload]
    out = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        out[idx] = row
    return out


def classify_batch(
    client: OpenAI,
    items: Sequence[dict],
    *,
    marca: str,
    aliases: Sequence[str],
    voceros: Sequence[str],
    model: str,
    candidatos: Sequence[str] | None = None,
    retries: int = 3,
    usage_sink: ClassifyStats | None = None,
) -> dict[int, dict]:
    user = build_user_prompt(
        items, marca=marca, aliases=aliases, voceros=voceros, candidatos=candidatos
    )
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=min(4000, 90 * max(len(items), 1) + 400),
            )
            prompt_t, completion_t = _usage_from_response(resp)
            if usage_sink is not None:
                usage_sink.prompt_tokens += prompt_t
                usage_sink.completion_tokens += completion_t
            content = resp.choices[0].message.content or ""
            return _index_resultados(_parse_json_content(content))
        except Exception as exc:  # noqa: BLE001 — se reintenta y se degrada
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"No se pudo clasificar el lote: {last_err}")


def _draft_row(
    raw: dict | None,
    titulo: str,
    resumen: str,
    marca: str,
    aliases: Sequence[str],
    voceros: Sequence[str],
    pasajes: str | None = None,
) -> tuple[str, str]:
    raw = raw or {}
    passages = (
        pasajes
        if pasajes is not None
        else extract_brand_passages(titulo, resumen, marca, aliases, voceros)
    )
    tono = canonicalize_tono(str(raw.get("tono") or raw.get("tone") or "Neutro"))
    sub = clean_subtema(
        str(raw.get("subtema") or raw.get("sub_tema") or raw.get("subtema_AI") or ""),
        titulo=titulo,
        resumen=resumen,
        marca=marca,
        aliases=aliases,
    )
    if not passages:
        hinted = infer_focus_tono(titulo, "", marca, aliases, voceros)
        tono = hinted if hinted in {"Positivo", "Negativo"} else "Neutro"
        return tono, sub
    hinted = infer_focus_tono(titulo, passages, marca, aliases, voceros)
    # El modelo a veces pinta el tema (desempleo, crimen…) como Negativo:
    # solo se conserva si la crítica apunta al FOCO.
    if tono == "Negativo" and hinted != "Negativo":
        tono = hinted if hinted == "Positivo" else "Neutro"
    elif tono == "Neutro" and hinted in {"Positivo", "Negativo"}:
        tono = hinted
    return tono, sub


def classify_rows(
    titles: Sequence[str],
    resumenes: Sequence[str],
    *,
    marca: str,
    aliases: Sequence[str] | None = None,
    voceros: Sequence[str] | None = None,
    api_key: str,
    model: str = DEFAULT_MODEL,
    batch_size: int = 10,
    progress: ProgressFn | None = None,
) -> tuple[list[str], list[str], ClassifyStats]:
    aliases = list(aliases or [])
    voceros = list(voceros or [])
    stats = ClassifyStats(model=model or DEFAULT_MODEL)
    t0 = time.perf_counter()
    client = OpenAI(api_key=api_key)
    n = len(titles)
    batch_size = max(1, min(int(batch_size or 10), 25))
    drafts_tono = ["Neutro"] * n
    drafts_sub = [""] * n
    candidatos: list[str] = []

    total_batches = (n + batch_size - 1) // batch_size if n else 1
    done_batches = 0
    cls_start = 0.04
    cls_span = 0.86

    if progress:
        progress(0.02, "Preparando lotes…")

    for start in range(0, n, batch_size):
        chunk_ids = list(range(start, min(start + batch_size, n)))
        items = []
        for i in chunk_ids:
            titulo = as_text(titles[i])
            cuerpo_raw = as_text(resumenes[i])
            cuerpo = normalize_body_text(cuerpo_raw, max_chars=BODY_MAX_CHARS)
            pasajes = extract_brand_passages(
                titulo, cuerpo_raw, marca, aliases, voceros, max_chars=PASSAGE_MAX_CHARS
            )
            items.append(
                {
                    "id": i,
                    "titulo": titulo[:280],
                    "pasajes": pasajes,
                    "resumen": cuerpo,
                }
            )
        lo, hi = chunk_ids[0] + 1, chunk_ids[-1] + 1
        if progress:
            frac = cls_start + (done_batches / max(total_batches, 1)) * cls_span
            progress(
                min(frac, 0.90),
                f"Clasificando lote {done_batches + 1} de {total_batches} · filas {lo}–{hi} de {n}…",
            )
        try:
            mapped = classify_batch(
                client,
                items,
                marca=marca,
                aliases=aliases,
                voceros=voceros,
                model=model,
                candidatos=candidatos,
                usage_sink=stats,
            )
        except Exception:
            mapped = {}
        for i in chunk_ids:
            tono, sub = _draft_row(
                mapped.get(i),
                as_text(titles[i]),
                as_text(resumenes[i]),
                marca,
                aliases,
                voceros,
                pasajes=items[i - start]["pasajes"],
            )
            drafts_tono[i] = tono
            drafts_sub[i] = sub
            if sub:
                candidatos.append(sub)
        done_batches += 1
        stats.batches = done_batches
        if progress:
            frac = cls_start + (done_batches / max(total_batches, 1)) * cls_span
            progress(
                min(frac, 0.90),
                f"Lote {done_batches} de {total_batches} listo · {hi} de {n} filas.",
            )

    if progress:
        progress(0.92, "Agrupando títulos y resúmenes parecidos (OCR)…")

    exclude = brand_tokens([marca], aliases, voceros)
    out_tono, out_sub = propagate_labels(
        drafts_tono,
        drafts_sub,
        [as_text(t) for t in titles],
        [as_text(r) for r in resumenes],
        marca=marca,
        aliases=aliases,
        exclude_tokens=exclude,
    )
    stats.elapsed_s = time.perf_counter() - t0
    if progress:
        progress(1.0, "Clasificación terminada.")
    return out_tono, out_sub, stats


def classify_dataframe(
    df,
    title_col: str,
    resumen_col: str,
    **kwargs: Any,
):
    titles = df[title_col].tolist()
    resumenes = df[resumen_col].tolist()
    tonos, subtemas, stats = classify_rows(titles, resumenes, **kwargs)
    out = df.copy()
    drop = [c for c in ("tono_AI", "tema_AI", "subtema_AI") if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    out["tono_AI"] = tonos
    out["subtema_AI"] = subtemas
    cols = [c for c in out.columns if c not in {"tono_AI", "subtema_AI"}]
    return out[cols + ["tono_AI", "subtema_AI"]], stats
