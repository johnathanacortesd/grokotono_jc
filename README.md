# Tono y subtema por marca (Streamlit)

App para clasificar **tono** y **subtema** de noticias anclados a una **marca**,
sus **alias** y **voceros** (no al sentimiento general de la nota).

Usa OpenAI (`gpt-4.1-nano-2025-04-14` por defecto) en lotes JSON y, después,
agrupa en local títulos o resúmenes parecidos (incluyendo errores típicos de
OCR). Dentro de cada grupo —y entre filas que ya compartan el mismo subtema—
el tono es **Positivo-first**: si alguna mención es Positivo, el grupo queda
Positivo.

Pensada para analistas de medios en Colombia. El archivo de entrada es un
`.xlsx` de menciones; la salida es el mismo Excel con `tono_AI` y `subtema_AI`.

## Qué hace

1. Pides la clave de acceso (`APP_PASSWORD`)
2. Subes un `.xlsx`
3. Eliges las columnas de **Título** y **Resumen**
4. Indicas marca, alias y voceros (barra lateral)
5. Genera `tono_AI` (`Positivo` | `Negativo` | `Neutro`) y `subtema_AI`
6. Ves un resumen compacto (conteo de tono, tiempo, tokens y costo) y
   descargas el Excel. No hay tablas de vista previa.

## Secrets (Streamlit Cloud)

En **App settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
APP_PASSWORD = "..."
```

También se acepta la forma anidada de OpenAI:

```toml
[openai]
api_key = "sk-..."
```

Sin `APP_PASSWORD` o sin `OPENAI_API_KEY` la app se detiene con un mensaje
claro. **No hay claves ni contraseñas en el repositorio.**

## Costo API (gpt-4.1-nano)

Constantes en `src/classify.py` (USD por 1 millón de tokens):

- Input: **$0.10 / 1M** (`INPUT_USD_PER_1M_TOKENS`)
- Output: **$0.40 / 1M** (`OUTPUT_USD_PER_1M_TOKENS`)

Al terminar, la app muestra tokens in/out, costo input, costo output, costo
total y el tiempo total en segundos y minutos (`42 s (0.7 min)`).

## Deploy en Streamlit Cloud

El código está en [johnathanacortesd/grokotono_jc](https://github.com/johnathanacortesd/grokotono_jc).

1. En [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Repository: `johnathanacortesd/grokotono_jc`. Branch: `main` (o la rama del PR).
3. **Main file path:** `app.py`
4. En *Advanced settings* elige **Python 3.12**.
5. Pega los secrets `OPENAI_API_KEY` y `APP_PASSWORD`.
6. Deploy.

En la app, el tema claro/oscuro de Streamlit (Settings) usa acento cian tipo X/Grok.

## Uso local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# .streamlit/secrets.toml:
# OPENAI_API_KEY = "sk-..."
# APP_PASSWORD = "..."
streamlit run app.py
```

Pruebas de postproceso (sin clave de OpenAI):

```bash
python3 -m unittest tests.test_postprocess -v
```

## Reglas (resumen)

- **Positivo** si el foco (marca / alias / voceros) es agente de una gestión o
  logro aunque el texto no traiga adjetivos: *la Universidad entregó…*,
  *avanzó la obra…*, *lanzó el programa…*, rankings, becas, convenios.
- **Negativo** si la crítica o la queja apunta al foco.
- **Neutro** solo si no hay vínculo evaluativo (sede/escenario, o la historia
  es de otro). No uses Neutro para gestiones «solo descriptivas» del foco.
- Nombre largo, nombre corto, sigla y voceros listados = la misma entidad.
- **Subtema:** frase lógica que condensa el resumen. Sentence case; se
  conservan siglas. No collage ni recorte del título.
- Noticias iguales o parecidas (título **o** resumen, con OCR) → mismo
  subtema y mismo tono. Si alguna es Positivo, el grupo queda Positivo.

## Estructura

```
app.py
requirements.txt
README.md
.streamlit/config.toml
src/
  __init__.py
  classify.py
  group.py
  io_xlsx.py
  normalize.py
  prompts.py
tests/
  test_postprocess.py
```
