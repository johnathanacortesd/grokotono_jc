"""App Streamlit: tono, tema y subtema de noticias anclados a una marca."""

from __future__ import annotations

import hashlib
import hmac
import html
import os

import pandas as pd
import streamlit as st

from src.classify import (
    DEFAULT_MODEL,
    INPUT_USD_PER_1M_TOKENS,
    OUTPUT_USD_PER_1M_TOKENS,
    classify_dataframe,
)
from src.io_xlsx import (
    dataframe_to_xlsx_bytes,
    guess_cuerpo_column,
    guess_title_column,
    list_sheets,
    read_xlsx,
)
from src.normalize import parse_name_list
from src.usage import load_recent_runs, maybe_notify_email, record_run

st.set_page_config(
    page_title="grokotono",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

_APP_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");

html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], .stMarkdown, label, p, h1, h2, h3 {
  font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif !important;
}

.block-container {
  padding-top: 0.7rem !important;
  padding-bottom: 3.2rem !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
  max-width: 598px !important;
}

footer, .stDeployButton, [data-testid="stDecoration"],
[data-testid="stAppDeployButton"] {
  display: none !important;
  visibility: hidden !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

section[data-testid="stSidebar"],
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
  display: none !important;
  width: 0 !important;
  min-width: 0 !important;
}

h1, h2, h3 {
  letter-spacing: -0.045em;
  font-weight: 800 !important;
  line-height: 1.15 !important;
}

.gx-brand { margin: 0 0 0.35rem 0; }
.gx-mark {
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: -0.07em;
  line-height: 1.05;
}
.gx-mark span { color: #FF6A00; }
.gx-tag {
  margin: 0.2rem 0 0.85rem 0;
  color: inherit;
  opacity: 0.62;
  font-size: 0.86rem;
  font-weight: 500;
  letter-spacing: -0.01em;
}

.gx-kicker {
  margin: 1.15rem 0 0.4rem 0;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #FF6A00;
  line-height: 1.2;
}
.gx-lead {
  margin: 0 0 0.85rem 0;
  font-size: 0.95rem;
  font-weight: 500;
  letter-spacing: -0.02em;
  line-height: 1.35;
  opacity: 0.78;
}

[data-testid="stForm"] {
  background: var(--st-secondary-background-color, var(--secondary-background-color, transparent));
  border: 1px solid var(--st-border-color, var(--border-color, rgba(255,106,0,0.28)));
  border-radius: 16px !important;
  padding: 0.85rem 1rem 1rem 1rem !important;
  margin: 0.35rem 0 1.1rem 0 !important;
  box-shadow: none !important;
}

[data-testid="stFileUploader"] {
  background: var(--st-secondary-background-color, var(--secondary-background-color, transparent));
  border: 1px solid var(--st-border-color, var(--border-color, rgba(255,106,0,0.28)));
  border-radius: 16px !important;
  padding: 0.35rem 0.55rem 0.55rem 0.55rem !important;
  margin: 0 0 0.85rem 0 !important;
}
[data-testid="stFileUploader"] section {
  position: relative !important;
  z-index: 0 !important;
}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploadDropzone"] {
  position: relative !important;
  z-index: 0 !important;
  border-radius: 12px !important;
}

[data-testid="stMetric"] {
  background: var(--st-secondary-background-color, var(--secondary-background-color, transparent));
  border: 1px solid var(--st-border-color, var(--border-color, rgba(255,106,0,0.28)));
  border-radius: 14px;
  padding: 0.7rem 0.8rem;
}
[data-testid="stMetricLabel"] { opacity: 0.68; font-weight: 650 !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { font-weight: 800 !important; letter-spacing: -0.045em; }

div[data-testid="stVerticalBlock"]:has(.gx-summary-flag) [data-testid="stMetric"] {
  padding: 0.5rem 0.65rem;
}
div[data-testid="stVerticalBlock"]:has(.gx-summary-flag) [data-testid="stMetricValue"] {
  font-size: 1.05rem !important;
  line-height: 1.15 !important;
}
div[data-testid="stVerticalBlock"]:has(.gx-summary-flag) [data-testid="stMetricLabel"] {
  font-size: 0.62rem !important;
}
.gx-summary-flag { display: none !important; height: 0 !important; margin: 0 !important; }

div[data-testid="stProgress"] {
  margin: 0.15rem 0 0.55rem 0 !important;
  padding: 0 !important;
}
div[data-testid="stProgress"] > div,
div[data-testid="stProgressBar"] > div {
  border-radius: 999px !important;
  height: 8px !important;
  overflow: hidden !important;
}
div[data-testid="stProgress"] p,
div[data-testid="stProgress"] span,
div[data-testid="stProgress"] [data-testid="stMarkdownContainer"],
div[data-testid="stProgressBar"] p,
[data-testid="stProgress"] label {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  overflow: hidden !important;
  position: static !important;
}
div[data-testid="stProgress"] > div > div,
div[data-testid="stProgressBar"] > div > div,
div[role="progressbar"] > div {
  background: linear-gradient(90deg, #FF6A00, #FFB347) !important;
  border-radius: 999px !important;
  transition: width 0.4s ease !important;
}

.gx-progress-status {
  display: block;
  margin: 0 0 0.15rem 0;
  padding: 0.15rem 0 0.05rem 0;
  font-size: 0.84rem;
  font-weight: 500;
  letter-spacing: -0.01em;
  line-height: 1.4;
  opacity: 0.72;
}

.stButton > button {
  border-radius: 999px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
  min-height: 2.55rem !important;
}
.stButton > button[kind="primary"] {
  background: #FF6A00 !important;
  color: #fff !important;
  border: 0 !important;
}
.stButton > button[kind="secondary"] {
  border: 1px solid var(--st-border-color, var(--border-color, rgba(127,127,127,0.35))) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div {
  border-radius: 12px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  outline: 2px solid #FF6A00 !important;
  border-color: #FF6A00 !important;
  box-shadow: 0 0 0 3px rgba(255, 106, 0, 0.22) !important;
}

div[data-testid="stExpander"] {
  border: 1px solid var(--st-border-color, var(--border-color, rgba(255,106,0,0.22)));
  border-radius: 14px !important;
  margin: 0 0 0.9rem 0 !important;
}

[data-testid="stCaption"] { opacity: 0.68; }

.stAlert { border-radius: 14px !important; }
</style>
"""


def inject_css() -> None:
    st.markdown(_APP_CSS, unsafe_allow_html=True)


def _secrets_obj():
    try:
        return st.secrets
    except Exception:
        return {}


def leer_secret(nombre: str) -> str:
    secrets = _secrets_obj()
    for key in (nombre, nombre.lower(), nombre.upper()):
        try:
            val = secrets[key]
            if val:
                return str(val).strip()
        except Exception:
            pass
    return (os.environ.get(nombre) or os.environ.get(nombre.upper()) or "").strip()


def leer_api_key() -> str:
    secrets = _secrets_obj()
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


def _clave_ok(ingresada: str, esperada: str) -> bool:
    a = hashlib.sha256((ingresada or "").encode("utf-8")).digest()
    b = hashlib.sha256((esperada or "").encode("utf-8")).digest()
    return hmac.compare_digest(a, b)


def format_elapsed(seconds: float) -> str:
    s = max(0.0, float(seconds))
    mins = s / 60.0
    if s < 10:
        return f"{s:.1f} s ({mins:.2f} min)"
    return f"{s:.0f} s ({mins:.1f} min)"


def format_usd(value: float) -> str:
    v = max(0.0, float(value))
    if v == 0:
        return "$0.00"
    if v < 0.01:
        return f"${v:.6f}"
    return f"${v:.4f}"


def brand_header() -> None:
    st.markdown(
        '<div class="gx-brand"><div class="gx-mark">grok<span>tono</span></div>'
        '<p class="gx-tag">tono · tema · subtema · marca</p></div>',
        unsafe_allow_html=True,
    )


def kicker(text: str) -> None:
    st.markdown(f'<p class="gx-kicker">{html.escape(text)}</p>', unsafe_allow_html=True)


def require_password() -> None:
    esperada = leer_secret("APP_PASSWORD")
    if not esperada:
        brand_header()
        st.error(
            "No se encontró **APP_PASSWORD** en los secrets. "
            "En Streamlit Cloud agrégalo en **App settings → Secrets**. "
            "En local usa `.streamlit/secrets.toml` o la variable de entorno "
            "`APP_PASSWORD`. La clave no va en el código."
        )
        st.stop()

    if st.session_state.get("auth_ok"):
        return

    brand_header()
    st.markdown(
        '<p class="gx-lead">Entra con la clave de la app para clasificar.</p>',
        unsafe_allow_html=True,
    )
    with st.form("gate", clear_on_submit=False, border=False, enter_to_submit=True):
        ingresada = st.text_input("Clave", type="password")
        entrar = st.form_submit_button("Entrar", type="primary", use_container_width=True)
    if entrar:
        if _clave_ok(ingresada, esperada):
            st.session_state["auth_ok"] = True
            st.rerun()
        st.error("Clave incorrecta.")
    st.stop()


def render_download(resultado: pd.DataFrame, archivo_nombre: str, *, key: str) -> None:
    base = (archivo_nombre or "menciones").rsplit(".", 1)[0]
    st.download_button(
        "Descargar Excel clasificado",
        data=dataframe_to_xlsx_bytes(resultado),
        file_name=f"{base}_tono_tema_subtema.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
        key=key,
    )


def render_summary(resultado: pd.DataFrame, stats: dict) -> None:
    st.markdown('<p class="gx-summary-flag"></p>', unsafe_allow_html=True)
    conteo = resultado["tono_AI"].value_counts().to_dict() if "tono_AI" in resultado.columns else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positivo", int(conteo.get("Positivo", 0)))
    c2.metric("Negativo", int(conteo.get("Negativo", 0)))
    c3.metric("Neutro", int(conteo.get("Neutro", 0)))
    c4.metric("Filas", int(len(resultado)))

    elapsed = format_elapsed(stats.get("elapsed_s") or 0)
    t1, t2 = st.columns(2)
    t1.metric("Tiempo", elapsed)
    t2.metric(
        "Tokens",
        f"{int(stats.get('prompt_tokens') or 0):,} in / {int(stats.get('completion_tokens') or 0):,} out".replace(",", "."),
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Costo input", format_usd(stats.get("cost_input_usd") or 0))
    k2.metric("Costo output", format_usd(stats.get("cost_output_usd") or 0))
    k3.metric("Costo total", format_usd(stats.get("cost_total_usd") or 0))
    st.caption(
        f"Estimado USD · gpt-4.1-nano · input ${INPUT_USD_PER_1M_TOKENS:.2f}/1M · "
        f"output ${OUTPUT_USD_PER_1M_TOKENS:.2f}/1M."
    )


def render_resultado(
    resultado: pd.DataFrame,
    stats: dict,
    archivo_nombre: str,
    *,
    download_key: str,
) -> None:
    render_download(resultado, archivo_nombre, key=download_key)
    render_summary(resultado, stats)


def render_uso_expander() -> None:
    with st.expander("Uso / clientes", expanded=False):
        st.caption(
            "Registro local de corridas (`data/uso_clientes.csv`). "
            "Tras cada corrida se avisa por correo a "
            "`cortesalexander8@gmail.com` si hay SMTP o Resend en secrets "
            "(anulable con `USAGE_NOTIFY_EMAIL`). "
            "En Streamlit Cloud el disco suele reiniciarse salvo que haya "
            "almacenamiento persistente."
        )
        try:
            rows = load_recent_runs(12)
        except Exception as exc:
            st.caption(f"No se pudo leer el log: {exc}")
            return
        if not rows:
            st.caption("Aún no hay corridas registradas en este servidor.")
            return
        preview = pd.DataFrame(rows)
        keep = [c for c in (
            "timestamp", "marca", "aliases", "n_rows",
            "positivo", "negativo", "neutro", "model", "elapsed_s", "cost_usd",
        ) if c in preview.columns]
        st.dataframe(preview[keep], hide_index=True, use_container_width=True)


def log_successful_run(resultado: pd.DataFrame, stats: dict, marca: str, aliases: list[str], model: str) -> None:
    conteo = resultado["tono_AI"].value_counts().to_dict() if "tono_AI" in resultado.columns else {}
    try:
        row = record_run(
            marca=marca,
            aliases=aliases,
            n_rows=len(resultado),
            tono_counts=conteo,
            model=model,
            elapsed_s=float(stats.get("elapsed_s") or 0),
            cost_usd=stats.get("cost_total_usd"),
        )
    except Exception:
        return
    try:
        maybe_notify_email(row, secrets=_secrets_obj())
    except Exception:
        pass


def main() -> None:
    inject_css()
    require_password()
    brand_header()
    st.markdown(
        '<p class="gx-lead">Clasifica tono_AI, tema_AI y subtema_AI anclados al foco. '
        "El tono mira solo cómo aparece la marca; el subtema sigue saliendo del cuerpo completo.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Regla de tono, tema y subtema", expanded=False):
        st.markdown(
            """
**El tono es un juicio sobre la marca, no sobre la historia.**

- **Positivo** si el foco (marca / alias / voceros) es agente de un encuentro,
  evento, gestión, entrega o avance — aunque el texto no traiga adjetivos.
- **Negativo** si la crítica o la queja apunta al foco. Colaborar en un estudio
  sobre un tema duro (p. ej. desempleo) **no** es Negativo.
- **Neutro** solo si no hay vínculo evaluativo (sede, escenario, o la historia es de otro).

El **subtema** es una frase nominal de **3 a 5 palabras**, sin el nombre de la marca,
distinta del título y de la primera línea del cuerpo. Prefiere la columna **CuerpoEs**.
El **tema** agrupa subtemas ya listos en una etiqueta un poco más general (máx. 4 palabras).
Noticias parecidas (OCR incluido) quedan con el mismo subtema y tono; **Positivo** gana.
            """
        )

    api_key = leer_api_key()
    if not api_key:
        st.error(
            "No se encontró la clave de OpenAI. En Streamlit Cloud agrégala en "
            "**App settings → Secrets** como `OPENAI_API_KEY` "
            "(también se acepta `[openai]` + `api_key`). En local puedes usar "
            "`.streamlit/secrets.toml` o la variable de entorno `OPENAI_API_KEY`."
        )
        st.stop()

    kicker("Archivo")
    uploaded = st.file_uploader(
        "Excel de menciones (.xlsx)",
        type=["xlsx"],
        label_visibility="visible",
    )
    if uploaded is None:
        resultado_prev = st.session_state.get("resultado")
        stats_prev = st.session_state.get("stats") or {}
        if isinstance(resultado_prev, pd.DataFrame) and not resultado_prev.empty:
            kicker("Listo")
            render_resultado(
                resultado_prev,
                stats_prev,
                st.session_state.get("archivo_nombre") or "menciones",
                download_key="dl_top_nofile",
            )
        st.caption("Sube el archivo. Después eliges columnas y el foco, en este mismo hilo.")
        render_uso_expander()
        if st.button("Cerrar sesión"):
            st.session_state["auth_ok"] = False
            st.rerun()
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
    kicker("Columnas")
    title_default = guess_title_column(cols) or cols[0]
    cuerpo_default = guess_cuerpo_column(cols) or cols[min(1, len(cols) - 1)]
    c1, c2 = st.columns(2)
    with c1:
        title_col = st.selectbox(
            "Columna de Título",
            cols,
            index=cols.index(title_default) if title_default in cols else 0,
        )
    with c2:
        cuerpo_col = st.selectbox(
            "Columna de cuerpo (CuerpoEs / Resumen)",
            cols,
            index=cols.index(cuerpo_default) if cuerpo_default in cols else 0,
            help="Prefiere CuerpoEs (texto completo con saltos). Resumen sirve si no hay cuerpo.",
        )
    st.caption(f"{len(df)} filas · **{hoja}** · se conservan las columnas originales.")

    kicker("Foco y ajuste")
    with st.form("foco_y_ajuste", clear_on_submit=False, border=False, enter_to_submit=False):
        marca = st.text_input(
            "Marca principal",
            placeholder="Universidad de Antioquia",
            help="Entidad sobre la que se juzga el tono. No va en el subtema.",
        )
        aliases_raw = st.text_area(
            "Alias",
            placeholder="UdeA\nU. de Antioquia\nla U",
            help="Sigla, nombre corto u otras formas. Una por línea o separadas por coma.",
            height=84,
        )
        voceros_raw = st.text_area(
            "Voceros propios (opcional)",
            placeholder="John Jairo Arboleda Céspedes",
            help="Personas cuya mención cuenta como mención de la marca.",
            height=72,
        )
        m1, m2, m3 = st.columns(3)
        with m1:
            modelo = st.text_input("Modelo OpenAI", value=DEFAULT_MODEL)
        with m2:
            lote = st.number_input("Tamaño de lote", min_value=1, max_value=25, value=10, step=1)
        with m3:
            max_filas = st.number_input(
                "Máximo de filas (0 = todas)",
                min_value=0,
                max_value=100_000,
                value=0,
                step=50,
                help="Útil para una prueba corta antes de correr el archivo completo.",
            )
        clasificar = st.form_submit_button("Clasificar", type="primary", use_container_width=True)

    if clasificar:
        if not marca.strip():
            st.warning("Escribe la **marca principal** en el bloque de foco, debajo del archivo.")
        else:
            trabajo = df.copy()
            if max_filas:
                trabajo = trabajo.head(int(max_filas))
            aliases = parse_name_list(aliases_raw)
            voceros = parse_name_list(voceros_raw)
            kicker("Progreso")
            estado = st.empty()
            estado.markdown(
                '<p class="gx-progress-status">Preparando lotes…</p>',
                unsafe_allow_html=True,
            )
            barra = st.progress(0.0)

            def on_progress(frac: float, msg: str) -> None:
                barra.progress(min(max(frac, 0.0), 1.0))
                estado.markdown(
                    f'<p class="gx-progress-status">{html.escape(msg)}</p>',
                    unsafe_allow_html=True,
                )

            try:
                resultado, stats = classify_dataframe(
                    trabajo,
                    title_col,
                    cuerpo_col,
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
                estado.empty()
                st.error(f"Falló la clasificación: {exc}")
                return

            st.session_state["resultado"] = resultado
            st.session_state["archivo_nombre"] = uploaded.name
            st.session_state["stats"] = {
                "elapsed_s": stats.elapsed_s,
                "prompt_tokens": stats.prompt_tokens,
                "completion_tokens": stats.completion_tokens,
                "cost_input_usd": stats.cost_input_usd,
                "cost_output_usd": stats.cost_output_usd,
                "cost_total_usd": stats.cost_total_usd,
            }
            log_successful_run(
                resultado,
                st.session_state["stats"],
                marca.strip(),
                aliases,
                modelo.strip() or DEFAULT_MODEL,
            )
            barra.progress(1.0)
            estado.markdown(
                '<p class="gx-progress-status">Clasificación terminada.</p>',
                unsafe_allow_html=True,
            )

    resultado = st.session_state.get("resultado")
    stats = st.session_state.get("stats") or {}
    if isinstance(resultado, pd.DataFrame) and not resultado.empty:
        kicker("Listo")
        render_resultado(
            resultado,
            stats,
            st.session_state.get("archivo_nombre") or uploaded.name,
            download_key="dl_after_progress",
        )

    render_uso_expander()

    if st.button("Cerrar sesión"):
        st.session_state["auth_ok"] = False
        st.rerun()


if __name__ == "__main__":
    main()
