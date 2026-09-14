# Tono y subtema por marca (Streamlit)

App para clasificar **tono** y **subtema** de noticias anclados a una **marca**, sus **alias** y **voceros** (no al sentimiento general de la nota).

Usa OpenAI (`gpt-4.1-nano-2025-04-14` por defecto) + agrupación local de títulos/resúmenes parecidos (OCR-aware) con **mismo subtema y mismo tono**, priorizando **Positivo** dentro del grupo.

## Qué hace

1. Subes un `.xlsx`
2. Eliges columnas de **título** y **resumen**
3. Indicas marca, alias y voceros
4. Genera `tono_AI` (`Positivo` | `Negativo` | `Neutro`) y `subtema_AI` (frase corta y completa)
5. Descargas el Excel con todas las columnas originales + las dos nuevas

## Secrets (Streamlit Cloud)

En **App settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
```

## Deploy

1. Crea un repo en GitHub y sube estos archivos
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. Main file: `app.py`
4. Pega el secret `OPENAI_API_KEY`
5. Deploy

## Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# opcional: export OPENAI_API_KEY=sk-...
# o crea .streamlit/secrets.toml con OPENAI_API_KEY
streamlit run app.py
```

## Reglas (resumen)

- **Positivo / Negativo / Neutro** solo respecto a la marca/alias/voceros
- Subtema: frase lógica que condensa el resumen (no collage del título)
- Noticias iguales o parecidas → mismo subtema y tono; si alguna es Positivo, el grupo queda Positivo

## Estructura

```
app.py
requirements.txt
README.md
.streamlit/config.toml
src/
  classify.py
  group.py
  io_xlsx.py
  normalize.py
  prompts.py
```
