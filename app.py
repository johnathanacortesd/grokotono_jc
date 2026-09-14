"""App Streamlit: tono y subtema de noticias anclados a una marca."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.classify import DEFAULT_MODEL, classify_dataframe
from src.io_xlsx import (
    dataframe_to_xlsx_bytes,
    guess_resumen_column,
    guess_title_column,
    list_sheets,
    read_xlsx,
)
from src.normalize import parse_name_list

st.set_page_config(
    page_title="Tono y subtema por marca",
    layout="wide",
    initial_sidebar_state="expanded",
)


def leer_api_key() -> str:
    """Lee OPENAI_API_KEY en la raíz de secrets o openai.api_key anidado."""
    secrets = {}
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    for key in ("OPENAI_API_KEY", "openai_api_key"):
        try:
            val = secrets[key]
            if val:
                return str(val).strip()
        except Exception:
            pass

    for section in ("openai", "OpenAI", "OPENAI"):
        nested = None
        try:
            nested = secrets[section]
        except Exception:
            nested = None
        if nested is None:
            continue
        for key in ("api_key", "API_KEY", "OPENAI_API_KEY", "openai_api_key"):
            try:
                val = nested[key]
            except Exception:
                try:
                    val = nested.get(key)  # type: ignore[attr-defined]
                except Exception:
                    val = None
            if val:
                return str(val).strip()

    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def main() -> None:
    st.title("Tono y subtema por marca")
    st.caption(
        "Clasifica noticias en tono_AI y subtema_AI anclados a la marca, sus alias y voceros. "
        "No mide el sentimiento general de la nota."
    )

    with st.expander("Cómo se mide el tono (léelo antes de clasificar)", expanded=True):
        st.markdown(
            """
**El tono es un juicio sobre la marca, no sobre la historia.**

Una nota puede hablar de un tema grave (un accidente, un delito, una inundación) y
seguir siendo **Neutro** para la marca si esta no es el blanco de la crítica.
**Positivo** / **Negativo** / **Neutro** solo aplican a lo que se dice de la
**marca principal**, sus **alias** o sus **voceros propios**.

El **subtema** es una frase nominal corta y completa en español colombiano que
condensa el *Resumen* (no un recorte del título ni un collage de palabras clave).
Noticias con título o resumen parecido —incluidos errores de OCR— quedan con el
mismo subtema y el mismo tono. Si alguna del grupo es **Positivo**, el grupo
queda Positivo.
            """
        )

    api_key = leer_api_key()
    if not api_key:
        st.error(
            "No se encontró la clave de OpenAI. En Streamlit Cloud agrégala en "
            "**App settings → Secrets** como `OPENAI_API_KEY` "
            '(también se acepta `[openai]` + `api_key`). En local puedes usar '
            "`.streamlit/secrets.toml` o la variable de entorno `OPENAI_API_KEY`."
        )
        st.stop()

    with st.sidebar:
        st.header("Marca y modelo")
        marca = st.text_input(
            "Marca principal",
            placeholder="Gobernación de Sucre",
            help="Entidad sobre la que se juzga el tono.",
        )
        aliases_raw = st.text_area(
            "Alias",
            placeholder="Gobernación\nSucre\nel departamento",
            help="Otras formas de nombrarla en medios: sigla, apodo, 'el departamento'. Una por línea o separadas por coma.",
        )
        voceros_raw = st.text_area(
            "Voceros propios (opcional)",
            placeholder="Lucy García\nLucy García Montes",
            help="Personas cuya mención cuenta como mención de la marca para el tono.",
        )
        modelo = st.text_input("Modelo OpenAI", value=DEFAULT_MODEL)
        lote = st.number_input("Tamaño de lote", min_value=1, max_value=25, value=10, step=1)
        max_filas = st.number_input(
            "Máximo de filas (0 = todas)",
            min_value=0,
            max_value=100_000,
            value=0,
            step=50,
            help="Útil para una prueba corta antes de correr el archivo completo.",
        )

    uploaded = st.file_uploader("Sube un archivo .xlsx de menciones", type=["xlsx"])
    if uploaded is None:
        st.info("Carga un Excel para elegir las columnas de **Título** y **Resumen**.")
        return

    raw = uploaded.getvalue()
    try:
        sheets = list_sheets(raw)
    except Exception as exc:
        st.error(f"No se pudo leer el Excel: {exc}")
        return

    hoja = sheets[0]
    if len(sheets) > 1:
        hoja = st.selectbox("Hoja", sheets, index=0)

    try:
        df = read_xlsx(raw, sheet=hoja)
    except Exception as exc:
        st.error(f"No se pudo abrir la hoja: {exc}")
        return

    if df.empty:
        st.warning("La hoja está vacía.")
        return

    cols = list(df.columns)
    c1, c2 = st.columns(2)
    title_default = guess_title_column(cols) or cols[0]
    resumen_default = guess_resumen_column(cols) or cols[min(1, len(cols) - 1)]
    with c1:
        title_col = st.selectbox(
            "Columna de Título",
            cols,
            index=cols.index(title_default) if title_default in cols else 0,
        )
    with c2:
        resumen_col = st.selectbox(
            "Columna de Resumen",
            cols,
            index=cols.index(resumen_default) if resumen_default in cols else 0,
        )

    st.caption(f"{len(df)} filas en **{hoja}**. Se conservan todas las columnas originales.")
    st.dataframe(df.head(8), use_container_width=True)

    if not marca.strip():
        st.warning("Escribe la **marca principal** en la barra lateral.")
        return

    if st.button("Clasificar", type="primary"):
        trabajo = df.copy()
        if max_filas:
            trabajo = trabajo.head(int(max_filas))
        aliases = parse_name_list(aliases_raw)
        voceros = parse_name_list(voceros_raw)
        barra = st.progress(0.0, text="Preparando lotes…")
        estado = st.empty()

        def on_progress(frac: float, msg: str) -> None:
            barra.progress(min(max(frac, 0.0), 1.0), text=msg)
            estado.write(msg)

        try:
            resultado = classify_dataframe(
                trabajo,
                title_col,
                resumen_col,
                marca=marca.strip(),
                aliases=aliases,
                voceros=voceros,
                api_key=api_key,
                model=modelo.strip() or DEFAULT_MODEL,
                batch_size=int(lote),
                progress=on_progress,
            )
        except Exception as exc:
            barra.empty()
            st.error(f"Falló la clasificación: {exc}")
            return

        st.session_state["resultado"] = resultado
        st.session_state["archivo_nombre"] = uploaded.name

    resultado = st.session_state.get("resultado")
    if isinstance(resultado, pd.DataFrame) and not resultado.empty:
        st.subheader("Resultado")
        st.dataframe(
            resultado[[c for c in (title_col, resumen_col, "tono_AI", "subtema_AI") if c in resultado.columns]].head(30),
            use_container_width=True,
        )
        conteo = resultado["tono_AI"].value_counts().to_dict() if "tono_AI" in resultado.columns else {}
        st.caption(
            " · ".join(f"{k}: {conteo.get(k, 0)}" for k in ("Positivo", "Negativo", "Neutro"))
            or "Sin conteo de tono."
        )
        base = (st.session_state.get("archivo_nombre") or "menciones").rsplit(".", 1)[0]
        st.download_button(
            "Descargar Excel clasificado",
            data=dataframe_to_xlsx_bytes(resultado),
            file_name=f"{base}_tono_subtema.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
