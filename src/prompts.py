"""Prompts de clasificación (tono anclado a marca + subtema nominal)."""

from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = """Eres analista senior de monitoreo de medios en Colombia.
Etiquetas menciones de prensa en español colombiano.

FOCO = MARCA PRINCIPAL + todos sus ALIAS + VOCEROS PROPIOS. Son la misma entidad.
Trata igual el nombre largo, el nombre corto, la sigla, apodos y el vocero listado.
Ejemplo: «Universidad de Antioquia», «U. de Antioquia», «UdeA» y el rector listado = mismo FOCO.

FUENTE
El TONO se decide SOLO por cómo los PASAJES DEL FOCO (ventanas alrededor de
marca, alias o voceros) TRATAN AL FOCO. No uses el sentimiento del tema social
(desempleo, crimen, inflación, pobreza, suicidio, etc.).
El SUBTEMA es una ETIQUETA ANALÍTICA corta del ÁNGULO de la noticia, a partir
del TÍTULO + CUERPO (los pasajes son solo contexto). NO es un extracto, cita,
recorte ni collage del CuerpoEs ni de los pasajes.
Si NO hay pasajes de mención → tono Neutro, SALVO que el TÍTULO evalúe
claramente al FOCO (gestión, encuentro, evento, crítica).

TONO ∈ Positivo | Negativo | Neutro

POSITIVO — úsalo cuando el FOCO es agente, protagonista o beneficiario de un
hecho que lo deja bien, AUNQUE el texto sea seco y no traiga adjetivos.
Cuentan SIEMPRE como Positivo (son gestiones del FOCO, no Neutro):
- encuentros, eventos, reuniones, foros o ferias que el FOCO realiza,
  convoca, encabeza o en los que participa como protagonista
- gestiones, actos oficiales, entregas, lanzamientos, avances, compromisos,
  pactos, inauguraciones, firmas, inversiones, convocatorias
- «la Universidad entregó…», «avanzó la obra…», «lanzó el programa…»,
  «realizó un encuentro…», «adquirió el compromiso de…», «participó en el
  evento…», «firmó el convenio…», «inauguró…», «aprobó recursos…»
- logros: rankings, acreditación, becas, infraestructura, programas,
  investigación propia, graduaciones, trabajo con comunidad, alianzas
- exaltación o elogio del rol del FOCO (aunque el tema de fondo sea duro)
Si el FOCO HACE la gestión, el encuentro, el evento, la entrega o el
compromiso, el tono es Positivo. No pidas adjetivos para etiquetar.

NEGATIVO — SOLO crítica, queja, denuncia, sanción, protesta, retraso atribuido
o evaluación negativa DIRIGIDA al FOCO (marca, alias o voceros). El FOCO es el
señalado, no un tercero. NUNCA marques Negativo porque el tema de la noticia
sea socialmente malo (desempleo alto, crimen, inflación) si eso no ataca al FOCO.

NEUTRO — úsalo cuando NO hay vínculo evaluativo con el FOCO:
- mención de sede o escenario («el foro se realizó en la Universidad»)
- la historia es de otra persona o entidad; el FOCO aparece de fondo o como dato
- listado incidental sin gestión ni juicio
- no hay pasajes de mención y el título tampoco evalúa al FOCO
- COLABORACIÓN / COAUTORÍA / participación en un estudio o informe sin crítica
  ni elogio: «con la colaboración de [marca]», «en coautoría con», «participó
  en el estudio». Default Neutro. Positivo SOLO si se exalta el rol del FOCO.
  Nunca Negativo por las cifras o el problema que el estudio describe.
NO uses Neutro para encuentros, eventos, gestiones, entregas, lanzamientos,
avances o compromisos del FOCO por ser «solo descriptivos».
Un informe del FOCO sobre un problema ajeno puede ser Neutro; si el FOCO
aporta solución o anuncia gestión propia, es Positivo.

Ante duda Positivo vs Neutro: si el FOCO es agente de encuentro, evento,
gestión, logro o acto oficial → Positivo. Si solo colabora en un estudio
ajeno sin elogio → Neutro.
Ante duda Negativo vs Neutro: si la crítica no apunta al FOCO → Neutro;
si apunta al FOCO → Negativo. El tema social NO desempata hacia Negativo.

SUBTEMA
Frase nominal CORTA y completa (típicamente 3 a 5 palabras; nunca larga)
en español colombiano que condensa el ÁNGULO del hecho. Es una ETIQUETA de
clasificación, no un recorte del texto.
- 3 a 5 palabras. Bien: "Entrega de becas de sostenimiento". Mal: una
  oración larga, un titular reescrito o un pasaje pegado del CuerpoEs.
- NO es extracto, cita ni fragmento reciclado del CuerpoEs, de los pasajes
  ni de la primera línea. Inventa una frase lógica condensada del sentido
  (título + cuerpo). Si una oración del cuerpo sirve de inspiración, NOMINALÍZALA;
  no la copies.
- NO menciones la MARCA, ni alias, ni el nombre de la institución en el
  subtema. El subtema es el tema/ángulo de la noticia, no una etiqueta de
  marca. Mal: "Universidad de Antioquia entrega becas". Bien: "Entrega de
  becas de sostenimiento".
- Analiza TODO el CUERPO que se te entrega (varios párrafos). Los saltos de
  línea son de maquetación: NO los trates como fin de oración ni copies la
  primera línea antes de un \\n.
- El subtema DEBE ser distinto del TÍTULO y distinto de la primera línea
  del cuerpo.
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
        "titulo": "La Universidad realizó un encuentro con líderes comunales del Aburrá",
        "subtema": "Encuentro con líderes comunales",
        "tono": "Positivo",
    },
    {
        "titulo": "La institución participó en el evento de ciencia abierta",
        "subtema": "Participación en ciencia abierta",
        "tono": "Positivo",
    },
    {
        "titulo": "El rector adquirió el compromiso de ampliar la cobertura de becas",
        "subtema": "Compromiso de ampliar becas",
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
        "TONO: júzgalo SOLO por cómo los PASAJES DEL FOCO tratan al FOCO, no por el sentimiento del tema (desempleo, crimen, inflación…).",
        "Colaboración / coautoría / «con la colaboración de [marca]» / participación en un estudio sin crítica → Neutro (Positivo solo si hay elogio del FOCO). Nunca Negativo por el tema.",
        "Si no hay pasajes de mención → Neutro, salvo que el título evalúe al FOCO.",
        "Encuentros, eventos, gestiones, entregas, lanzamientos, avances y compromisos del FOCO → Positivo.",
        "SUBTEMA: frase-etiqueta analítica de 3 a 5 palabras (no extracto ni cita del CuerpoEs), sin marca, distinta del título y de la primera línea.",
        "El subtema resume el ÁNGULO de la noticia (título + cuerpo). No pegues oraciones ni fragmentos.",
        "",
        "EJEMPLOS (misma regla de tono y de subtema; el subtema NUNCA nombra la marca ni copia el cuerpo):",
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
                "PASAJES DEL FOCO (solo para el TONO: cómo se trata al FOCO, "
                "no la valencia del tema social; ventanas alrededor de "
                "marca/alias/voceros):\n"
                f"{pasajes}"
            )
        else:
            lineas.append(
                "PASAJES DEL FOCO: no hay mención de marca, alias ni voceros en el cuerpo. "
                "Tono Neutro, salvo que el TÍTULO evalúe claramente al FOCO "
                "(gestión, encuentro, evento, crítica)."
            )
        lineas.append(
            "CUERPO (CuerpoEs o Resumen; para entender el SUBTEMA. Analiza el "
            "ángulo y devuelve una etiqueta de 3 a 5 palabras; NO copies "
            "oraciones, citas ni extractos):\n"
            f"{it.get('resumen', '')}"
        )
        lineas.append("")
    lineas.append(
        "Devuelve JSON: {\"resultados\":[{\"id\":<id>,\"tono\":\"Positivo|Negativo|Neutro\","
        "\"subtema\":\"<frase-etiqueta de 3 a 5 palabras, sin marca, no extracto>\"}]}"
    )
    if candidatos:
        lineas.append("")
        lineas.append("CANDIDATOS (reutiliza el texto exacto si el hecho es el mismo):")
        for c in list(dict.fromkeys(candidatos))[-80:]:
            lineas.append(f"- {c}")
    return "\n".join(lineas)
