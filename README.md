# Tono, tema y subtema por marca (Streamlit)

App para clasificar **tono**, **tema** y **subtema** de noticias anclados a una
**marca**, sus **alias** y **voceros** (no al sentimiento general de la nota).

Usa OpenAI (`gpt-4.1-nano-2025-04-14` por defecto) en lotes JSON. El tono y el
subtema se deciden sobre **pasajes** del `CuerpoEs` (ventanas alrededor de cada
mención de marca / alias / voceros), con el título como apoyo. Después agrupa
en local títulos o cuerpos parecidos (OCR) y, aparte, agrupa subtemas
parecidos en un `tema_AI` más general. Dentro de cada grupo de noticia —y entre
filas que ya compartan el mismo subtema— el tono es **Positivo-first**.

Pensada para analistas de medios en Colombia. El archivo de entrada es un
`.xlsx` de menciones; la salida es el mismo Excel con `tono_AI`, `tema_AI` y
`subtema_AI` (en ese orden).

## Qué hace

1. Pides la clave de acceso (`APP_PASSWORD`)
2. Subes un `.xlsx` (primero, en la columna principal)
3. Eliges las columnas de **Título** y **cuerpo** (`CuerpoEs` se prefiere;
   `Resumen` sirve si no hay cuerpo). El cuerpo puede ir **completo**, con
   saltos de línea.
4. Indicas marca, alias, voceros y ajustes **debajo** del archivo (no en
   una barra lateral)
5. Extrae pasajes de mención, genera `tono_AI`
   (`Positivo` | `Negativo` | `Neutro`), `tema_AI` y `subtema_AI`
6. Ves un resumen compacto (conteo de tono, tiempo, tokens y costo) y
   descargas el Excel **justo debajo del progreso**. No hay tablas de vista
   previa.

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

Tema claro: fondo limpio con acento **naranja**. Oscuro: negro profundo y
naranja (energía tipo Grok/X, acento de marca naranja). En Settings de
Streamlit se cambia claro/oscuro.

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

- Se lee el **CuerpoEs completo** (los saltos de línea son válidos). El
  clasificador no usa el artículo entero como sentimiento: extrae **pasajes**
  con oraciones/párrafos alrededor de cada hit de marca, alias o vocero.
- **Tono y subtema** salen de esos pasajes; el **título** es apoyo.
- Si no hay pasajes de mención → **Neutro**, salvo que el título evalúe
  claramente al foco.
- **Positivo** si el foco es agente de un **encuentro, evento, gestión,
  entrega, lanzamiento, avance o compromiso**, aunque el texto no traiga
  adjetivos: *la Universidad entregó…*, *realizó un encuentro…*, *avanzó la
  obra…*, *lanzó el programa…*, rankings, becas, convenios.
- **Negativo** si la crítica o la queja apunta al foco.
- **Neutro** si solo es sede/escenario, o la historia es de otro. No uses
  Neutro para gestiones «solo descriptivas» del foco.
- Nombre largo, nombre corto, sigla y voceros listados = la misma entidad.
- **Subtema:** frase nominal de **3 a 5 palabras** a partir de los pasajes.
  Sentence case; se conservan siglas. **No menciones la marca.** No copies el
  título ni la primera línea del cuerpo. Noticias iguales o parecidas (OCR)
  → mismo subtema y mismo tono; **Positivo** gana.
- **Tema (`tema_AI`):** más amplio que el subtema. Subtemas iguales o
  parecidos quedan con el **mismo** tema. Si un subtema no tiene hermanos,
  igual recibe un tema específico un poco más general (p. ej. subtema
  «Entrega de becas de sostenimiento» → tema «Becas y apoyos estudiantiles»).
  Español de Colombia, sentence case, corto (~2 a 5 palabras), sin relleno
  de marca.

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
  tema.py
tests/
  test_postprocess.py
```
