"""Prompts de clasificación (tono anclado a marca + subtema nominal)."""

from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = """Eres analista senior de monitoreo de medios en Colombia.
Etiquetas menciones de prensa en español colombiano.

FOCO = MARCA PRINCIPAL + todos sus ALIAS + VOCEROS PROPIOS. Son la misma entidad.
Trata igual el nombre largo, el nombre corto, la sigla, apodos y el vocero listado.
Ejemplo: «Universidad de Antioquia», «U. de Antioquia», «UdeA» y el rector listado = mismo FOCO.

El TONO se juzga SOLO por cómo aparece el FOCO (marca, alias, voceros) en
los PASAJES DEL FOCO (oraciones del CuerpoEs que lo mencionan) y, como
apoyo, el TÍTULO. No es el sentimiento general de la historia, ni de un
sector, ni de un territorio. Una nota puede ser negativa en su tema
(desempleo, crimen, inflación) y aun así Neutro o Positivo para el FOCO.

TONO ∈ Positivo | Negativo | Neutro

POSITIVO — úsalo cuando el FOCO es agente, protagonista o beneficiario de un hecho
que lo deja bien, AUNQUE el texto sea seco y no traiga adjetivos
(«excelente», «destacado», «exitoso», etc.). Cuentan como Positivo:
- gestiones, encuentros, eventos, entregas y avances del FOCO:
  «la Universidad entregó…», «avanzó la obra de…»,
  «lanzó el programa…», «firmó el convenio…», «inauguró…», «aprobó recursos…»,
  «puso en marcha…», «destinó inversión…», «realizó un encuentro…»,
  «participó en el evento…»
- logros: rankings, acreditación, becas, infraestructura, programas, investigación,
  graduaciones, trabajo con comunidad, alianzas, convocatorias que abre el FOCO
- reconocimiento, inversión, cifras buenas, acompañamiento o respaldo al FOCO
Si el FOCO HACE la gestión, el tono es Positivo. No pidas adjetivos para etiquetar.

NEGATIVO — crítica, queja, denuncia, sanción, protesta, retraso atribuido o
evaluación negativa DIRIGIDA al FOCO (marca, alias o voceros). El FOCO es el
señalado, no un tercero. NUNCA marques Negativo solo porque el tema de la
noticia sea socialmente malo si eso no ataca al FOCO.

NEUTRO — úsalo POCO. Solo cuando NO hay vínculo evaluativo con el FOCO:
- mención de sede o escenario («el foro se realizó en la Universidad»)
- la historia es de otra persona o entidad; el FOCO aparece de fondo o como dato
- listado incidental sin gestión ni juicio
- COLABORACIÓN en un estudio o informe ajeno sin crítica ni elogio
  («con la colaboración de [marca]», «participó en el estudio»). Default Neutro.
  Positivo solo si se exalta el rol. Nunca Negativo por las cifras del problema
  (p. ej. desempleo).
NO uses Neutro para gestiones institucionales del FOCO por ser «solo descriptivas».
Un informe del FOCO sobre un problema ajeno puede ser Neutro; si el FOCO aporta
solución o anuncia gestión propia, es Positivo.

Ante duda Positivo vs Neutro: si el FOCO es agente de gestión, logro o acto
oficial → Positivo. Si solo colabora en un estudio ajeno sin elogio → Neutro.
Ante duda Negativo vs Neutro: si la crítica no apunta al FOCO → Neutro;
si apunta al FOCO → Negativo.

SUBTEMA
Frase nominal CORTA y completa (máximo 6 palabras; no recortes a mitad de sentido)
en español colombiano que condensa el ÁNGULO del hecho a partir del CUERPO
completo (CuerpoEs o Resumen). No es un collage, no es un recorte del título
y no es la primera línea del cuerpo.
- Máximo 6 palabras. Prefiere una frase corta completa (4 a 6 está bien).
  Bien: "Entrega de becas de sostenimiento". Bien: "Inicio de clases con
  alimentación escolar". Bien: "Cocha Molina celebró". Mal: una oración
  larga, un extracto del cuerpo o un titular reescrito.
- NO menciones la MARCA, ni alias, ni el nombre de la institución en el
  subtema. El subtema es el tema/ángulo de la noticia, no una etiqueta de
  marca. Mal: "Universidad de Antioquia entrega becas". Bien: "Entrega de
  becas de sostenimiento".
- Analiza TODO el CUERPO que se te entrega (varios párrafos). Los saltos de línea
  son de maquetación: NO los trates como fin de oración ni copies la
  primera línea antes de un \\n.
- El subtema DEBE ser distinto del TÍTULO y distinto de la primera línea
  del cuerpo. Inventa una frase lógica condensada del sentido completo.
- Oración nominal coherente: sujeto + complemento. Bien: "Inicio de clases
  con PAE". Mal: "Clases PAE Sucre niños".
- Mayúscula solo en la primera letra (sentence case). Conserva siglas (PAE,
  ANI, EPS, DANE) y nombres propios que NO sean la marca.
- No termines en preposición ni nexo (de, la, el, en, con, por, para, y,
  del, ha, porque…).
- No empieces con verbo conjugado. Bien: "Sanción a exsecretario".
  Mal: "Sancionan a exsecretario".
- Prohibido rótulos vacíos: noticias generales, gestión institucional,
  mención.
- Si el hecho ya aparece en CANDIDATOS, copia ese texto EXACTO (sigue
  siendo corto y sin marca).

Responde ÚNICAMENTE JSON válido, sin markdown:
{"resultados":[{"id":0,"tono":"Positivo","subtema":"..."}]}
Debes devolver un objeto por cada id recibido.
"""

EJEMPLOS = [
    {
        "titulo": "La Universidad entregó 400 becas de sostenimiento a estudiantes de estratos 1 y 2",
        "subtema": "Entrega de becas de sostenimiento",
        "tono": "Positivo",
    },
    {
        "titulo": "Avanzó la obra del nuevo bloque de laboratorios en el campus",
        "subtema": "Avance de obra de laboratorios",
        "tono": "Positivo",
    },
    {
        "titulo": "La U. de Antioquia lanzó el programa de diplomados virtuales para docentes",
        "subtema": "Lanzamiento de diplomados virtuales",
        "tono": "Positivo",
    },
    {
        "titulo": "Universidad Pontificia Bolivariana firmó convenio de movilidad con institución de España",
        "subtema": "Convenio de movilidad con España",
        "tono": "Positivo",
    },
    {
        "titulo": "La institución ocupó el puesto 8 en el ranking QS de universidades colombianas",
        "subtema": "Puesto 8 en ranking QS nacional",
        "tono": "Positivo",
    },
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
        "titulo": "Gobernación de Sucre aprobó más de $39 mil millones para la construcción de la Variante Sampués - Segovia- Sincelejo",
        "subtema": "Aprobación de recursos para la Variante Sampués",
        "tono": "Positivo",
    },
    {
        "titulo": "Estudiantes protestan contra la Universidad por alza en los derechos de matrícula",
        "subtema": "Protesta por alza de matrícula",
        "tono": "Negativo",
    },
    {
        "titulo": "Por demoras en el PAE, Procuraduría suspende a exsecretario de Educación de Sucre",
        "subtema": "Sanción a exsecretario",
        "tono": "Negativo",
    },
    {
        "titulo": "Denuncian irregularidades en contratación de la institución",
        "subtema": "Denuncia de irregularidades en contratación",
        "tono": "Negativo",
    },
    {
        "titulo": "El foro de periodismo se realizó en el auditorio de la Universidad",
        "subtema": "Foro de periodismo en el campus",
        "tono": "Neutro",
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
    {
        "titulo": "La Universidad realizó un encuentro con líderes comunales del Aburrá",
        "subtema": "Encuentro con líderes comunales",
        "tono": "Positivo",
    },
    {
        "titulo": "Desempleo juvenil en Barranquilla llega al 18,8 % según un nuevo informe",
        "subtema": "Informe de desempleo juvenil",
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
        f"ALIAS (misma entidad que la marca): {alias_txt}",
        f"VOCEROS PROPIOS (misma entidad que la marca): {vocero_txt}",
        "",
        "Recuerda: nombre largo, nombre corto, sigla y voceros listados = el mismo FOCO.",
        "Si el FOCO es el que entrega, lanza, avanza, firma, inaugura, invierte o anuncia, tono Positivo.",
        "Neutro solo si es sede/escenario o la historia es de otro.",
        "TONO: solo cómo los PASAJES DEL FOCO (y el título de apoyo) tratan al FOCO, no el sentimiento de la noticia.",
        "Colaborar en un estudio sobre desempleo (u otro problema social) NO es Negativo para el FOCO.",
        "Encuentros, eventos, gestiones, entregas y avances del FOCO → Positivo.",
        "SUBTEMA: máximo 6 palabras, frase corta completa, sin el nombre de la marca/alias, distinto del título y de la primera línea.",
        "Analiza el CUERPO completo (ya viene con saltos de línea unidos). No copies el titular ni el arranque.",
        "",
        "EJEMPLOS (misma regla de tono y de subtema; el subtema NUNCA nombra la marca):",
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
                "LO QUE SE DICE DEL FOCO (marca/alias/voceros; decide el TONO, no el subtema): "
                f"{pasajes}"
            )
        else:
            lineas.append(
                "LO QUE SE DICE DEL FOCO: no hay mención clara al nombre, alias ni voceros. "
                "Usa Neutro salvo que el título/resumen sí nombren una variante del FOCO "
                "como agente de una gestión."
            )
        lineas.append(
            "CUERPO (CuerpoEs o Resumen; texto completo ya normalizado. "
            "Analiza todo el bloque, no solo la primera oración ni la primera línea): "
            f"{it.get('resumen', '')}"
        )
        lineas.append("")
    lineas.append(
        "Devuelve JSON: {\"resultados\":[{\"id\":<id>,\"tono\":\"Positivo|Negativo|Neutro\","
        "\"subtema\":\"<frase nominal de máximo 6 palabras, sin marca>\"}]}"
    )
    if candidatos:
        lineas.append("")
        lineas.append("CANDIDATOS (reutiliza el texto exacto si el hecho es el mismo):")
        for c in list(dict.fromkeys(candidatos))[-80:]:
            lineas.append(f"- {c}")
    return "\n".join(lineas)
