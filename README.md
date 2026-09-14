# Tono, tema y subtema por marca (Streamlit)

App para clasificar **tono**, **tema** y **subtema** de noticias anclados a una
**marca**, sus **alias** y **voceros** (no al sentimiento general de la nota).

Usa OpenAI (`gpt-4.1-nano-2025-04-14` por defecto) en lotes JSON. El **subtema**
se decide con el título + el **CuerpoEs completo** (misma ruta que el commit
`18b79f6`). El **tono** se juzga solo por cómo aparecen marca / alias / voceros
en los pasajes de mención (el título es apoyo). Después agrupa en local títulos
o cuerpos parecidos (OCR) y, **aparte**, agrupa los subtemas ya finalizados en
un `tema_AI` más general. Dentro de cada grupo de noticia —y entre filas que ya
compartan el mismo subtema— el tono es **Positivo-first**.

Pensada para analistas de medios en Colombia. El archivo de entrada es un
`.xlsx` de menciones; la salida es el mismo Excel con `tono_AI`, `tema_AI` y
`subtema_AI` (en ese orden). El `subtema_AI` no se reescribe al calcular el tema.

## Qué hace

1. Pides la clave de acceso (`APP_PASSWORD`)
2. Subes un `.xlsx` (primero, en la columna principal)
3. Eliges las columnas de **Título** y **cuerpo** (`CuerpoEs` se prefiere;
   `Resumen` sirve si no hay cuerpo)
4. Indicas marca, alias, voceros y ajustes **debajo** del archivo (no en
   una barra lateral)
5. Genera `tono_AI` (`Positivo` | `Negativo` | `Neutro`), agrupa subtemas en
   `tema_AI` y deja `subtema_AI` como en el paso anterior
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

### Aviso opcional por correo (uso / clientes)

Tras **cada corrida exitosa** se appende una fila en `data/uso_clientes.csv`
(timestamp, marca, alias, filas, conteos de tono, modelo, tiempo, costo) y,
si hay canal SMTP o Resend en secrets, se envía un correo de aviso. El
expander **Uso / clientes** muestra las últimas filas si el archivo existe.

El destino por defecto (confirmado por A.C.) es
**`cortesalexander8@gmail.com`**. Se puede anular con `USAGE_NOTIFY_EMAIL`.
**No hay contraseñas SMTP ni API keys en el código**; van solo en secrets.

```toml
# Opcional: anula el destino por defecto (cortesalexander8@gmail.com)
# USAGE_NOTIFY_EMAIL = "otro@correo.com"

# Opción A — SMTP
SMTP_HOST = "smtp.ejemplo.com"
SMTP_PORT = "587"
SMTP_USER = "usuario"
SMTP_PASSWORD = "..."          # solo en secrets, nunca en el repo
SMTP_FROM = "grokotono@ejemplo.com"

# Opción B — Resend (si está RESEND_API_KEY se usa este canal)
RESEND_API_KEY = "re_..."      # solo en secrets, nunca en el repo
# SMTP_FROM o USAGE_NOTIFY_FROM = remitente verificado en Resend
```

Sin `SMTP_HOST` y sin `RESEND_API_KEY` solo se escribe el CSV. Un fallo de
correo no interrumpe la clasificación.

**Caveat Streamlit Cloud:** el sistema de archivos es efímero. El CSV se pierde
al reiniciar el contenedor salvo que uses un disco persistente (o copies el
log a otro almacenamiento). En un servidor propio el archivo sí queda.

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

Tema claro: fondo limpio con acento **naranja**. Oscuro: negro y naranja.
En Settings de Streamlit se cambia claro/oscuro.

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

- **Tono** = cómo se trata al FOCO en los pasajes de mención + título. No es
  el sentimiento de la noticia. Colaborar en un estudio sobre desempleo
  (u otro problema social) **no** es Negativo para la marca.
- **Positivo** si el foco es agente de un **encuentro, evento, gestión,
  entrega o avance**, aunque el texto no traiga adjetivos.
- **Negativo** si la crítica o la queja apunta al foco.
- **Neutro** si solo es sede/escenario, o la historia es de otro. No uses
  Neutro para gestiones «solo descriptivas» del foco.
- Nombre largo, nombre corto, sigla y voceros listados = la misma entidad.
- **Subtema (congelado desde `18b79f6`):** frase nominal de **3 a 5 palabras**
  a partir del **cuerpo completo** (preferir **CuerpoEs**). Sentence case;
  se conservan siglas. **No menciones la marca.** No copies el título ni la
  primera línea. Noticias iguales o parecidas (OCR) → mismo subtema y mismo
  tono; **Positivo** gana. El cálculo de `tema_AI` **no** regenera ni
  «mejora» el subtema.
- **Tema (`tema_AI`):** más amplio que el subtema (máx. **4 palabras**).
  Subtemas iguales o parecidos quedan con el **mismo** tema. Si un subtema
  es único, igual recibe un tema un poco más general. Español de Colombia,
  sentence case, sin relleno de marca.

## Estructura

```
app.py
requirements.txt
README.md
.streamlit/config.toml
data/
  .gitkeep
  uso_clientes.csv   # se crea en runtime (no va al git)
src/
  __init__.py
  classify.py
  group.py
  io_xlsx.py
  normalize.py
  prompts.py
  tema.py
  usage.py
tests/
  test_postprocess.py
```
