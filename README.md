# Tono y subtema por marca (Streamlit)

App para clasificar **tono** y **subtema** de noticias anclados a una
**marca**, sus **alias** y **voceros** (no al sentimiento general de la nota).

Usa OpenAI (`gpt-4.1-nano-2025-04-14` por defecto) en lotes JSON. El **subtema**
se decide con el título + el **CuerpoEs completo** (misma ruta que el commit
`18b79f6`). El **tono** se juzga solo por cómo aparecen marca / alias / voceros
en los pasajes de mención (el título es apoyo). Después agrupa en local títulos
o cuerpos parecidos (OCR). Dentro de cada grupo de noticia —y entre filas que ya
compartan el mismo subtema— el tono es **Positivo-first**.

Pensada para analistas de medios en Colombia. El archivo de entrada es un
`.xlsx` de menciones; la salida es el mismo Excel con `tono_AI` y
`subtema_AI` (en ese orden). No hay columna `tema_AI`.

## Qué hace

1. Pides la clave de acceso (`APP_PASSWORD`)
2. Subes un `.xlsx` (primero, en la columna principal)
3. Eliges las columnas de **Título** y **cuerpo** (`CuerpoEs` se prefiere;
   `Resumen` sirve si no hay cuerpo)
4. Indicas marca, alias, voceros y ajustes **debajo** del archivo (no en
   una barra lateral)
5. Genera `tono_AI` (`Positivo` | `Negativo` | `Neutro`) y `subtema_AI`
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

### Aviso por correo (interno; no hay expander en la UI)

Tras **cada corrida exitosa** se appende `data/uso_clientes.csv` (timestamp,
marca, alias, filas, conteos de tono, modelo, tiempo, costo, `email_status`)
y **siempre se intenta** enviar un correo a
**`cortesalexander8@gmail.com`**. Un fallo de correo **no** interrumpe la
clasificación: el estado queda en el CSV y en un log de servidor
(`[grokotono] email_status=…`). El log de uso **no** se muestra en la app.

Destino por defecto: **`cortesalexander8@gmail.com`**. Anulable con
`USAGE_NOTIFY_EMAIL`. **No hay contraseñas SMTP ni API keys en el código.**

Camino más simple: **Resend**. Pega esto en Secrets:

```toml
RESEND_API_KEY = "re_..."
USAGE_NOTIFY_EMAIL = "cortesalexander8@gmail.com"  # optional override
USAGE_NOTIFY_FROM = "onboarding@resend.dev"  # or verified domain
```

También se aceptan formas anidadas, por ejemplo `[resend] api_key = "re_..."`.
Si `RESEND_API_KEY` no está, se usa SMTP (`SMTP_HOST`, `SMTP_PORT`,
`SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`). Si no hay ni Resend ni SMTP, el
CSV anota `email_status=skipped_no_secrets` (una línea, para diagnosticar).

```toml
# Alternativa SMTP (solo si no hay RESEND_API_KEY)
SMTP_HOST = "smtp.ejemplo.com"
SMTP_PORT = "587"
SMTP_USER = "usuario"
SMTP_PASSWORD = "..."          # solo en secrets, nunca en el repo
SMTP_FROM = "grokotono@ejemplo.com"
```

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
5. Pega los secrets `OPENAI_API_KEY`, `APP_PASSWORD` y, para el aviso de
   uso, `RESEND_API_KEY` (ver TOML más arriba).
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
- **Subtema:** frase nominal de **3 a 5 palabras** que condensa el ángulo
  de la nota a partir del **cuerpo completo** (preferir **CuerpoEs**; se
  unen los saltos de línea de maquetación y se lee un tramo sustancial,
  no solo la primera línea). Sentence case; se conservan siglas. **No
  menciones la marca** en el subtema. No copies el título ni la primera
  línea del cuerpo. Noticias iguales o parecidas (OCR) → mismo subtema y
  mismo tono; **Positivo** gana. Misma ruta que `18b79f6`.

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
  usage.py
tests/
  test_postprocess.py
```
