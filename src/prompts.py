"""Prompts de clasificación (tono anclado a marca + subtema nominal)."""

from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = """Eres analista senior de monitoreo de medios en Colombia.
Etiquetas menciones de prensa en español colombiano.

FOCO = MARCA PRINCIPAL + todos sus ALIAS + VOCEROS PROPIOS. Son la misma entidad.
Trata igual el nombre largo, el nombre corto, la sigla, apodos y el vocero listado.
Ejemplo: «Universidad de Antioquia», «U. de Antioquia», «UdeA» y el rector listado = mismo FOCO.

FUENTE PRINCIPAL
El TONO y el SUBTEMA se deciden PRIMARIO a partir de los PASAJES DEL FOCO
(ventanas de oraciones/párrafos alrededor de cada mención de marca, alias o
voceros) y, como apoyo, del TÍTULO.
NO uses el sentimiento general de la historia, ni de un sector, ni de un
territorio. Si el cuerpo es largo, ignora lo que no esté en los pasajes ni
en el título.
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
  investigación, graduaciones, trabajo con comunidad, alianzas
Si el FOCO HACE la gestión, el encuentro, el evento, la entrega o el
compromiso, el tono es Positivo. No pidas adjetivos para etiquetar.

NEGATIVO — crítica, queja, denuncia, sanción, protesta, retraso atribuido o
evaluación negativa DIRIGIDA al FOCO (marca, alias o voceros). El FOCO es el
señalado, no un tercero.

NEUTRO — úsalo POCO. Solo cuando NO hay vínculo evaluativo con el FOCO:
- mención de sede o escenario («el foro se realizó en la Universidad»)
- la historia es de otra persona o entidad; el FOCO aparece de fondo o como dato
- listado incidental sin gestión ni juicio
- no hay pasajes de mención y el título tampoco evalúa al FOCO
NO uses Neutro para encuentros, eventos, gestiones, entregas, lanzamientos,
avances o compromisos del FOCO por ser «solo descriptivos».
Un informe del FOCO sobre un problema ajeno puede ser Neutro; si el FOCO
aporta solución o anuncia gestión propia, es Positivo.

Ante duda Positivo vs Neutro: si el FOCO es agente de encuentro, evento,
gestión, logro o acto oficial → Positivo.
Ante duda Negativo vs Neutro: si la crítica no apunta al FOCO → Neutro;
si apunta al FOCO → Negativo.

SUBTEMA
Frase nominal CORTA y completa (típicamente 3 a 5 palabras; nunca larga)
en español colombiano que condensa el ÁNGULO del hecho a partir de los
PASAJES DEL FOCO (y el título como apoyo). No es un collage, no es un
recorte del título y no es la primera línea del cuerpo.
- 3 a 5 palabras. Bien: "Entrega de becas de sostenimiento". Mal: una
  oración larga o un titular reescrito.
- NO menciones la MARCA, ni alias, ni el nombre de la institución en el
  subtema. El subtema es el tema/ángulo de la noticia, no una etiqueta de
  marca. Mal: "Universidad de Antioquia entrega becas". Bien: "Entrega de
  becas de sostenimiento".
- El subtema DEBE ser distinto del TÍTULO y distinto de la primera línea
  del cuerpo. Inventa una frase lógica condensada del sentido de los pasajes.
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
        "Decide TONO y SUBTEMA sobre todo con los PASAJES DEL FOCO; el TÍTULO es apoyo.",
        "Si no hay pasajes de mención → Neutro, salvo que el título evalúe al FOCO.",
        "Encuentros, eventos, gestiones, entregas, lanzamientos, avances y compromisos del FOCO → Positivo.",
        "SUBTEMA: 3 a 5 palabras, sin el nombre de la marca/alias, distinto del título y de la primera línea.",
        "",
        "EJEMPLOS (misma regla de tono y de subtema; el subtema NUNCA nombra la marca):",
    ]
    for e in EJEMPLOS:
        lineas.append(f'  TITULAR: {e["titulo"]}')
        lineas.append(f'  -> subtema: "{e["subtema"]}" | tono: {e["tono"]}')
    lineas += ["", "NOTAS A ETIQUETAR:"]
    for it in items:
        lineas.append(f"NOTA id={it['id']}")
        lineas.append(f"TÍTULO (apoyo): {it['titulo']}")
        pasajes = it.get("pasajes") or ""
        if pasajes:
            lineas.append(
                "PASAJES DEL FOCO (fuente PRINCIPAL de tono y subtema; "
                "ventanas alrededor de marca/alias/voceros):\n"
                f"{pasajes}"
            )
        else:
            lineas.append(
                "PASAJES DEL FOCO: no hay mención de marca, alias ni voceros en el cuerpo. "
                "Tono Neutro, salvo que el TÍTULO evalúe claramente al FOCO "
                "(gestión, encuentro, evento, crítica)."
            )
        lineas.append("")
    lineas.append(
        "Devuelve JSON: {\"resultados\":[{\"id\":<id>,\"tono\":\"Positivo|Negativo|Neutro\","
        "\"subtema\":\"<frase nominal de 3 a 5 palabras, sin marca>\"}]}"
    )
    if candidatos:
        lineas.append("")
        lineas.append("CANDIDATOS (reutiliza el texto exacto si el hecho es el mismo):")
        for c in list(dict.fromkeys(candidatos))[-80:]:
            lineas.append(f"- {c}")
    return "\n".join(lineas)
