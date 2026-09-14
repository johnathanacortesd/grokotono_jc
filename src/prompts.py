"""Prompts de clasificación (tono anclado a marca + subtema nominal)."""

from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = """Eres analista senior de monitoreo de medios en Colombia.
Etiquetas menciones de prensa en español colombiano.

El TONO se juzga SOLO por lo que la nota dice de la MARCA, sus ALIAS o sus VOCEROS.
No es el sentimiento general de la historia, ni de un sector, ni de un territorio.
No es un juicio sobre si el tema es alegre o triste.

TONO ∈ Positivo | Negativo | Neutro
- Positivo: la marca, un alias o un vocero propio es sujeto de un hecho favorable
  (obra, programa, avance, beneficio, reconocimiento, alianza, cifra buena,
  acompañamiento, respaldo, anuncio que los deja bien).
- Negativo: hay crítica, reclamo, sanción, denuncia o evaluación negativa
  DIRIGIDA a la marca, su administración o sus voceros.
- Pregunta clave antes de escribir Negativo: ¿de quién habla la nota?
  Si la marca o el vocero NO es el responsable, señalado o protagonista de la
  crítica, el tono es Neutro. Temas graves (muertes, delito, desempleo,
  inundaciones, obras de terceros, quejas contra otros) NO hacen Negativo a
  quien aparece de fondo o como voz institucional.
- El tema no decide el tono. Un informe de la marca sobre un problema es Neutro
  (o Positivo si aporta solución). Negativo exige ataque o señalamiento CONTRA
  la marca o el vocero.
- Neutro: todo lo demás, incluida la ausencia de mención a la marca.
- Ante duda entre Positivo y Neutro, o entre Negativo y Neutro, elige Neutro.

SUBTEMA
Frase nominal corta y completa en español colombiano que condensa el HECHO del
Resumen (no un collage de palabras clave, no un recorte del título).
- Oración nominal coherente: sujeto + complemento. Bien: "Inicio de clases con
  alimentación escolar". Mal: "Clases PAE Sucre niños".
- Mayúscula solo en la primera letra (sentence case). Conserva siglas (PAE, ANI,
  EPS, DANE) y nombres propios.
- No termines en preposición ni nexo (de, la, el, en, con, por, para, y, del,
  ha, porque…).
- No copies ni recortes el titular. Sintetiza el resumen.
- No empieces con verbo conjugado. Bien: "Sanción a exsecretario de Educación".
  Mal: "Sancionan a exsecretario".
- Prohibido rótulos vacíos: noticias generales, gestión institucional, mención.
- Si el hecho ya aparece en CANDIDATOS, copia ese texto EXACTO.

Responde ÚNICAMENTE JSON válido, sin markdown:
{"resultados":[{"id":0,"tono":"Positivo","subtema":"..."}]}
Debes devolver un objeto por cada id recibido.
"""

EJEMPLOS = [
    {
        "titulo": "Sucre lo hace de nuevo: 40 mil niños y niñas inician sus clases con alimentación escolar desde el primer día",
        "subtema": "Inicio de clases con alimentación escolar",
        "tono": "Positivo",
    },
    {
        "titulo": "Gobernación de Sucre impulsa economía familiar y seguridad alimentaria en Toluviejo con 2 mil gallinas ponedoras",
        "subtema": "Gallinas ponedoras para economía familiar",
        "tono": "Positivo",
    },
    {
        "titulo": "Ciudad Natural del Golfo de Morrosquillo: la estrategia de Sucre para dinamizar el turismo",
        "subtema": "Ciudad Natural del Golfo de Morrosquillo",
        "tono": "Positivo",
    },
    {
        "titulo": "En Sucre destruyen más de 250 mil productos de contrabando valorados en más de 670 millones de pesos",
        "subtema": "Destrucción de productos de contrabando",
        "tono": "Positivo",
    },
    {
        "titulo": "Sucre abre nuevas rutas de cooperación internacional tras visita de la embajadora de Australia, Anna Chrisp",
        "subtema": "Cooperación internacional con Australia",
        "tono": "Positivo",
    },
    {
        "titulo": "Gobernación de Sucre aprobó más de $39 mil millones para la construcción de la Variante Sampués - Segovia- Sincelejo",
        "subtema": "Aprobación de recursos para la Variante Sampués",
        "tono": "Positivo",
    },
    {
        "titulo": "Por demoras en el PAE, Procuraduría suspende a exsecretario de Educación de Sucre",
        "subtema": "Sanción a exsecretario de Educación",
        "tono": "Negativo",
    },
    {
        "titulo": "Sancionan a exsecretario de Educación de Sucre, por demora en el PAE",
        "subtema": "Sanción por retraso en el PAE",
        "tono": "Negativo",
    },
    {
        "titulo": "Vía al Llano, una obra que está estancada",
        "subtema": "Estancamiento de la vía al Llano",
        "tono": "Negativo",
    },
    {
        "titulo": "Gobernadores del Caribe y la ANI evalúan proyecto del Canal del Dique",
        "subtema": "Revisión del proyecto Canal del Dique",
        "tono": "Neutro",
    },
    {
        "titulo": "Más de la mitad de los intentos de suicidio en Colombia corresponden a jóvenes entre 15 y 29 años",
        "subtema": "Informe sobre intentos de suicidio en jóvenes",
        "tono": "Neutro",
    },
]


def build_user_prompt(
    items: Sequence[dict],
    *,
    marca: str,
    aliases: Sequence[str],
    voceros: Sequence[str],
    candidatos: Sequence[str] | None = None,
) -> str:
    alias_txt = ", ".join(aliases) if aliases else "(sin alias adicionales)"
    vocero_txt = ", ".join(voceros) if voceros else "(no definido)"
    lineas = [
        f"MARCA PRINCIPAL: {marca}",
        f"ALIAS: {alias_txt}",
        f"VOCEROS PROPIOS: {vocero_txt}",
        "",
        "EJEMPLOS (misma regla de tono y de subtema):",
    ]
    for e in EJEMPLOS:
        lineas.append(f'  TITULAR: {e["titulo"]}')
        lineas.append(f'  -> subtema: "{e["subtema"]}" | tono: {e["tono"]}')
    lineas += ["", "NOTAS A ETIQUETAR:"]
    for it in items:
        lineas.append(f"NOTA id={it['id']}")
        lineas.append(f"TÍTULO: {it['titulo']}")
        pasajes = it.get("pasajes") or ""
        if pasajes:
            lineas.append(
                "LO QUE SE DICE DE LA MARCA (decide el tono; nada más cuenta): "
                f"{pasajes}"
            )
        else:
            lineas.append(
                "LO QUE SE DICE DE LA MARCA: (no hay mención clara) -> tono Neutro"
            )
        lineas.append(
            "RESUMEN (sirve para el subtema; NO decide el tono): "
            f"{it.get('resumen', '')}"
        )
        lineas.append("")
    lineas.append(
        "Devuelve JSON: {\"resultados\":[{\"id\":<id>,\"tono\":\"Positivo|Negativo|Neutro\","
        "\"subtema\":\"<frase nominal>\"}]}"
    )
    if candidatos:
        lineas.append("")
        lineas.append("CANDIDATOS (reutiliza el texto exacto si el hecho es el mismo):")
        for c in list(dict.fromkeys(candidatos))[-80:]:
            lineas.append(f"- {c}")
    return "\n".join(lineas)
