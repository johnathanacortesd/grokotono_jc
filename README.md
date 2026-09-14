# Tono y subtema por marca (Streamlit)

App para clasificar **tono** y **subtema** de noticias anclados a una **marca**,
sus **alias** y **voceros** (no al sentimiento general de la nota).

Usa OpenAI (`gpt-4.1-nano-2025-04-14` por defecto) en lotes JSON y, después,
agrupa en local títulos o resúmenes parecidos (incluyendo errores típicos de
OCR). Dentro de cada grupo —y entre filas que ya compartan el mismo subtema—
el tono es **Positivo-first**: si alguna mención es Positivo, el grupo queda
Positivo.

Pensada para analistas de medios en Colombia (flujo tipo Gobernación de Sucre /
vocería). El archivo de entrada es un `.xlsx` de menciones; la salida es el
mismo Excel con `tono_AI` y `subtema_AI`.

## Qué hace

1. Subes un `.xlsx`
2. Eliges las columnas de **Título** y **Resumen**
3. Indicas marca, alias y voceros (barra lateral)
4. Genera `tono_AI` (`Positivo` | `Negativo` | `Neutro`) y `subtema_AI`
   (frase nominal corta y completa en español colombiano)
5. Descargas el Excel con **todas** las columnas originales + las dos nuevas

## Secrets (Streamlit Cloud)

En **App settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
```

También se acepta la forma anidada:

```toml
[openai]
api_key = "sk-..."
```

Sin clave la app se detiene con un mensaje claro. No hay claves de ejemplo en
el repositorio.

## Deploy en Streamlit Cloud

1. Publica este repo en GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. **Main file path:** `app.py`
4. En *Advanced settings* elige **Python 3.12**.
5. Pega el secret `OPENAI_API_KEY`.
6. Deploy.

## Uso local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# opcional: export OPENAI_API_KEY=sk-...
# o crea .streamlit/secrets.toml con OPENAI_API_KEY
streamlit run app.py
```

Pruebas de postproceso (sin clave de OpenAI):

```bash
python3 -m unittest tests.test_postprocess -v
```

## Reglas (resumen)

- **Positivo / Negativo / Neutro** solo respecto a la marca, alias o voceros.
  El tema de la nota (un delito, una tragedia, una cifra nacional) no decide
  el tono. Ante duda, Neutro.
- **Subtema:** frase lógica que condensa el resumen. Sentence case (mayúscula
  solo en la primera letra); se conservan siglas (PAE, ANI, EPS). No collage
  de keywords ni recorte del título. No termina en *de, la, el, en, con, por,
  para, y, del, ha, porque…*
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
