"""Lectura y escritura de Excel de menciones."""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import pandas as pd

TITLE_HINTS = (
    "titulo",
    "título",
    "titular",
    "headline",
    "title",
    "titulos",
    "encabezado",
)
RESUMEN_HINTS = (
    "resumen",
    "resumen - aclaracion",
    "resumen - aclaración",
    "resumen aclaracion",
    "cuerpo",
    "cuerposes",
    "contenido",
    "texto",
    "bajada",
    "lead",
    "copete",
    "descripcion",
    "descripción",
    "sintesis",
    "síntesis",
    "noticia",
)


def list_sheets(data: bytes | BytesIO) -> list[str]:
    bio = data if isinstance(data, BytesIO) else BytesIO(data)
    xl = pd.ExcelFile(bio, engine="openpyxl")
    return list(xl.sheet_names)


def read_xlsx(data: bytes | BytesIO, sheet: str | int = 0) -> pd.DataFrame:
    bio = data if isinstance(data, BytesIO) else BytesIO(data)
    df = pd.read_excel(bio, sheet_name=sheet, engine="openpyxl", dtype=object)
    df.columns = [str(c).strip() if c is not None else "" for c in df.columns]
    return df


def _norm_col(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def guess_column(columns: Sequence[str], hints: Sequence[str]) -> str | None:
    cols = list(columns)
    norms = {_norm_col(c): c for c in cols}
    for h in hints:
        if h in norms:
            return norms[h]
    for h in hints:
        for n, orig in norms.items():
            if h in n:
                return orig
    return cols[0] if cols else None


def guess_title_column(columns: Sequence[str]) -> str | None:
    return guess_column(columns, TITLE_HINTS)


def guess_resumen_column(columns: Sequence[str]) -> str | None:
    guessed = guess_column(columns, RESUMEN_HINTS)
    title = guess_title_column(columns)
    if guessed and title and guessed == title:
        for c in columns:
            if c != title:
                return c
    return guessed


def dataframe_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Menciones")
    return buf.getvalue()
