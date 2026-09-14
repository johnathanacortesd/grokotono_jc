"""Pruebas locales de postproceso (sin API)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.classify import ClassifyStats, classify_dataframe, classify_rows, estimate_cost_usd
from src.group import cluster_indices, positivo_first, propagate_labels
from src.io_xlsx import dataframe_to_xlsx_bytes, guess_cuerpo_column, guess_resumen_column, guess_title_column, read_xlsx
from src.normalize import (
    canonicalize_tono,
    clean_subtema,
    first_content_line,
    focus_names,
    fold_text,
    infer_focus_tono,
    mentions_target,
    name_variants,
    normalize_body_text,
    ocr_fold,
    same_folded_phrase,
    sentence_case,
    strip_brand_mentions,
    strip_dangling,
)
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.tema import assign_temas, broaden_subtema, cluster_subtema_indices


class NormalizeTests(unittest.TestCase):
    def test_dangling_endings(self):
        words = strip_dangling("Alimentación escolar de la".split())
        self.assertEqual(words[-1].lower(), "escolar")

    def test_clean_subtema_strips_dangling_ending(self):
        out = clean_subtema(
            "Avance de obra de",
            titulo="Otro titular sobre laboratorios en el campus",
            resumen=(
                "Avanzó la obra de laboratorios en el campus central.\n"
                "Se instalaron equipos nuevos de investigación aplicada."
            ),
            marca="UdeA",
        )
        self.assertTrue(out)
        self.assertNotIn(fold_text(out.split()[-1]), {
            "de", "del", "la", "el", "en", "con", "por", "para", "y",
        })

    def test_subtema_rejects_title_equal(self):
        titulo = "La Universidad entregó 400 becas de sostenimiento"
        cuerpo = (
            "La rectoría presentó cifras de cobertura del semestre.\n"
            "El programa de sostenimiento benefició a estudiantes de estratos 1 y 2 "
            "con giros mensuales durante el calendario académico."
        )
        out = clean_subtema(
            titulo,
            titulo=titulo,
            resumen=cuerpo,
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
        )
        self.assertFalse(same_folded_phrase(out, titulo))
        self.assertNotEqual(ocr_fold(out), ocr_fold(titulo))
        self.assertLessEqual(len(out.split()), 6)

    def test_subtema_rejects_first_line_equal(self):
        first = "La rectoría presentó el informe anual de gestión"
        cuerpo = (
            first
            + "\nEl resto del artículo habla de acreditación de alta calidad "
            "y nuevos programas de posgrado en ingeniería aplicada."
        )
        titulo = "Otro titular distinto sobre investigación aplicada"
        out = clean_subtema(
            first,
            titulo=titulo,
            resumen=cuerpo,
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
        )
        self.assertEqual(first_content_line(cuerpo), first)
        self.assertFalse(same_folded_phrase(out, first))
        self.assertNotEqual(ocr_fold(out), ocr_fold(first))
        self.assertFalse(same_folded_phrase(out, titulo))

    def test_strip_brand_mentions_helper(self):
        out = strip_brand_mentions(
            "Universidad de Antioquia entrega becas de sostenimiento",
            "Universidad de Antioquia",
            ["UdeA"],
        )
        folded = fold_text(out)
        self.assertNotIn("antioquia", folded)
        self.assertNotIn("udea", folded)
        self.assertIn("becas", folded)

    def test_clean_subtema_omits_marca(self):
        out = clean_subtema(
            "Universidad de Antioquia entrega becas de sostenimiento",
            titulo="Otra nota sobre ranking QS nacional de universidades",
            resumen=(
                "Se entregaron becas de sostenimiento a estudiantes de estratos 1 y 2.\n"
                "El giro cubre alimentación y transporte durante el semestre en curso."
            ),
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
        )
        self.assertNotIn("antioquia", fold_text(out))
        self.assertLessEqual(len(out.split()), 6)
        self.assertGreaterEqual(len(out.split()), 3)

    def test_normalize_body_joins_newlines(self):
        raw = "Primera línea del lead\n\nSegundo párrafo con más contexto del hecho."
        out = normalize_body_text(raw)
        self.assertNotIn("\n", out)
        self.assertIn("Segundo párrafo", out)
        self.assertIn("Primera línea", out)

    def test_sentence_case_keeps_acronym(self):
        self.assertEqual(
            sentence_case("sanción por retraso en el PAE"),
            "Sanción por retraso en el PAE",
        )

    def test_subtema_not_keyword_collage(self):
        out = clean_subtema(
            "Educación, Salud, Infraestructura",
            titulo="Gobernación anuncia PAE",
            resumen="La Gobernación de Sucre inició el PAE desde el primer día de clases.",
            marca="Gobernación de Sucre",
        )
        self.assertNotIn(",", out)
        self.assertGreaterEqual(len(out.split()), 3)

    def test_clean_subtema_allows_six_word_phrase(self):
        phrase = "Inicio de clases con alimentación escolar"
        self.assertEqual(len(phrase.split()), 6)
        out = clean_subtema(
            phrase,
            titulo="Otro titular distinto sobre el PAE departamental",
            resumen=(
                "El departamento arrancó el PAE el primer día de clases para 40 mil niños.\n"
                "Segundo párrafo de contexto institucional sobre cobertura alimentaria."
            ),
            marca="Gobernación de Sucre",
        )
        self.assertEqual(out, phrase)
        self.assertEqual(len(out.split()), 6)

    def test_clean_subtema_rejects_cocha_molina_celebro(self):
        raw = "Cocha Molina celebró"
        out = clean_subtema(
            raw,
            titulo="Otro titular de un acto protocolario en el campus",
            resumen=(
                "La rectoría adelantó un acto protocolario con la comunidad académica.\n"
                "Hubo reconocimientos a docentes y estudiantes del semestre en curso."
            ),
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
        )
        self.assertNotEqual(ocr_fold(out), ocr_fold(raw))
        self.assertNotEqual(fold_text(out.split()[-1]), "celebro")
        self.assertLessEqual(len(out.split()), 6)
        self.assertGreaterEqual(len(out.split()), 3)

    def test_tono_canonical(self):
        self.assertEqual(canonicalize_tono("POSITIVO"), "Positivo")
        self.assertEqual(canonicalize_tono("mixto"), "Neutro")


class GroupTests(unittest.TestCase):
    def test_ocr_titles_cluster(self):
        titles = [
            "Gobernación de Sucre impulsa el PAE escolar desde el primer día",
            "Gobemacion de Sucre impulsa el PAE escolar desde el primer dia",
            "Variante Sampués recibe más de 39 mil millones",
        ]
        resumenes = [
            "La Gobernación confirmó alimentación escolar desde el primer día de clases.",
            "La Gobemacion confirmo alimentacion escolar desde el primer dia de clases.",
            "Se aprobaron recursos para la variante Sampués-Segovia-Sincelejo.",
        ]
        groups = cluster_indices(titles, resumenes, exclude_tokens={"gobernacion", "sucre"})
        clustered = {frozenset(g) for g in groups if len(g) > 1}
        self.assertIn(frozenset({0, 1}), clustered)

    def test_positivo_first_in_cluster_and_subtema(self):
        titles = [
            "40 mil niños inician clases con alimentación escolar en Sucre",
            "Niños y niñas inician clases con alimentación escolar desde el primer día",
            "Informe nacional de suicidio juvenil",
            "Estudio de suicidio en jóvenes de 15 a 29 años",
        ]
        resumenes = [
            "El departamento arrancó el PAE el primer día de clases para 40 mil niños.",
            "Arranque del PAE el primer dia de clases para ninos y ninas.",
            "Un informe nacional alerta sobre intentos de suicidio en jóvenes.",
            "El estudio reitera la alerta sobre intentos de suicidio en jovenes.",
        ]
        tonos = ["Neutro", "Positivo", "Neutro", "Negativo"]
        subs = [
            "Inicio de clases con alimentación escolar",
            "Alimentación escolar en el primer día",
            "Informe sobre intentos de suicidio en jóvenes",
            "Informe sobre intentos de suicidio en jóvenes",
        ]
        out_t, out_s = propagate_labels(
            tonos,
            subs,
            titles,
            resumenes,
            marca="Gobernación de Sucre",
            exclude_tokens={"gobernacion", "sucre"},
        )
        self.assertEqual(out_t[0], "Positivo")
        self.assertEqual(out_t[1], "Positivo")
        self.assertEqual(out_s[0], out_s[1])
        self.assertEqual(out_t[2], out_t[3])
        self.assertEqual(out_t[2], "Neutro")

    def test_positivo_first_helper(self):
        self.assertEqual(positivo_first(["Neutro", "Negativo", "Positivo"]), "Positivo")
        self.assertEqual(positivo_first(["Neutro", "Negativo", "Neutro"]), "Neutro")


class MentionGuardTests(unittest.TestCase):
    def test_no_mention_is_not_brand(self):
        self.assertFalse(
            mentions_target(
                "Alcalde de Cartagena revisó el Canal del Dique",
                "Autoridades del Caribe evaluaron la obra con la ANI.",
                "Gobernación de Sucre",
                ["el departamento"],
                ["Lucy García"],
            )
        )

    def test_vocero_counts(self):
        self.assertTrue(
            mentions_target(
                "Lucy García anunció el Hospital Madre",
                "La mandataria lideró la inauguración en Sincelejo.",
                "Gobernación de Sucre",
                [],
                ["Lucy García"],
            )
        )


class ClassifyPipelineTests(unittest.TestCase):
    def test_drafts_then_positivo_first_group(self):
        titles = [
            "40 mil niños inician clases con alimentación escolar en Sucre",
            "Niños inician clases con alimentación escolar desde el primer día",
        ]
        resumenes = [
            "La Gobernación de Sucre arrancó el PAE el primer día de clases.",
            "Gobemacion de Sucre arranco el PAE el primer dia de clases.",
        ]

        def fake_batch(client, items, **kwargs):
            return {
                items[0]["id"]: {
                    "id": items[0]["id"],
                    "tono": "Neutro",
                    "subtema": "Inicio de clases con alimentación escolar",
                },
                items[1]["id"]: {
                    "id": items[1]["id"],
                    "tono": "Positivo",
                    "subtema": "Alimentación escolar en el primer día",
                },
            }

        with patch("src.classify.OpenAI"), patch(
            "src.classify.classify_batch", side_effect=fake_batch
        ):
            from src.classify import classify_rows

            tonos, subs, temas, _stats = classify_rows(
                titles,
                resumenes,
                marca="Gobernación de Sucre",
                aliases=["Sucre"],
                voceros=["Lucy García"],
                api_key="test",
                batch_size=10,
            )
        self.assertEqual(tonos[0], "Positivo")
        self.assertEqual(tonos[1], "Positivo")
        self.assertEqual(subs[0], subs[1])
        self.assertTrue(temas[0])
        self.assertEqual(temas[0], temas[1])

    def test_sends_substantial_normalized_cuerpo(self):
        titulo = "Entrega de apoyos de sostenimiento en el campus"
        lead = "Lead corto de la nota."
        rest = "Párrafo de desarrollo del hecho institucional con detalle. " * 90
        cuerpo = lead + "\n" + rest
        captured = {}

        def fake_batch(client, items, **kwargs):
            captured["resumen"] = items[0]["resumen"]
            return {
                items[0]["id"]: {
                    "id": items[0]["id"],
                    "tono": "Positivo",
                    "subtema": "Entrega de apoyos",
                }
            }

        with patch("src.classify.OpenAI"), patch(
            "src.classify.classify_batch", side_effect=fake_batch
        ):
            classify_rows(
                [titulo],
                [cuerpo],
                marca="Universidad de Antioquia",
                aliases=["UdeA"],
                voceros=[],
                api_key="test",
                batch_size=10,
            )
        self.assertNotIn("\n", captured["resumen"])
        self.assertGreater(len(captured["resumen"]), 1200)
        self.assertLessEqual(len(captured["resumen"]), 7100)
        self.assertIn("desarrollo del hecho", captured["resumen"])


class IoTests(unittest.TestCase):
    def test_guess_and_roundtrip(self):
        df = pd.DataFrame(
            {
                "Título": ["Nota uno"],
                "Resumen": ["Hecho de prueba"],
                "Medio": ["El Heraldo"],
            }
        )
        raw = dataframe_to_xlsx_bytes(df)
        back = read_xlsx(raw)
        self.assertEqual(guess_title_column(back.columns), "Título")
        self.assertEqual(guess_resumen_column(back.columns), "Resumen")
        self.assertEqual(list(back.columns), list(df.columns))

    def test_prefers_cuerposes_over_resumen(self):
        cols = ["Título", "Resumen", "CuerpoEs", "Medio"]
        self.assertEqual(guess_cuerpo_column(cols), "CuerpoEs")
        self.assertEqual(guess_resumen_column(cols), "CuerpoEs")


class FocusMatchingTests(unittest.TestCase):
    def test_marca_variants_include_short_and_u_de(self):
        names = name_variants("Universidad de Antioquia")
        folded = " ".join(names).lower()
        self.assertTrue(any("u de antioquia" in v.lower() for v in names))
        self.assertIn("universidad", folded)

    def test_alias_and_vocero_are_same_focus(self):
        names = focus_names(
            "Universidad de Antioquia",
            ["UdeA", "U. de Antioquia"],
            ["John Jairo Arboleda Céspedes"],
        )
        blob_alias = "la udea lanzo un programa de becas"
        blob_short = "la u de antioquia firmo un convenio"
        blob_vocero = "john jairo arboleda cespedes anuncio la acreditacion"
        self.assertTrue(mentions_target(blob_alias, "", "Universidad de Antioquia", ["UdeA"], []))
        self.assertTrue(mentions_target(blob_short, "", "Universidad de Antioquia", ["U. de Antioquia"], []))
        self.assertTrue(
            mentions_target(
                blob_vocero,
                "",
                "Universidad de Antioquia",
                [],
                ["John Jairo Arboleda Céspedes"],
            )
        )
        compact = " ".join(names).lower()
        self.assertIn("udea", compact.replace(".", ""))

    def test_upb_acronym(self):
        vars_ = [v.lower() for v in name_variants("Universidad Pontificia Bolivariana")]
        self.assertTrue(any(v.replace(" ", "") == "upb" for v in vars_))


class TonoHeuristicaTests(unittest.TestCase):
    marca = "Universidad de Antioquia"
    aliases = ["UdeA", "U. de Antioquia"]
    voceros = ["John Jairo Arboleda Céspedes"]

    def test_gestion_sin_adjetivos_es_positivo(self):
        self.assertEqual(
            infer_focus_tono(
                "La Universidad entregó 400 becas de sostenimiento",
                "La Universidad de Antioquia entregó becas a estudiantes de estratos 1 y 2.",
                self.marca,
                self.aliases,
                self.voceros,
            ),
            "Positivo",
        )
        self.assertEqual(
            infer_focus_tono(
                "Avanzó la obra del nuevo bloque de laboratorios",
                "La U. de Antioquia avanzó la obra de laboratorios en el campus.",
                self.marca,
                self.aliases,
                self.voceros,
            ),
            "Positivo",
        )
        self.assertEqual(
            infer_focus_tono(
                "La UdeA lanzó el programa de diplomados virtuales",
                "La institución lanzó diplomados para docentes.",
                self.marca,
                self.aliases,
                self.voceros,
            ),
            "Positivo",
        )

    def test_critica_al_foco_es_negativo(self):
        self.assertEqual(
            infer_focus_tono(
                "Estudiantes protestan contra la Universidad por alza de matrícula",
                "Hay quejas contra la UdeA por el incremento de derechos pecunarios.",
                self.marca,
                self.aliases,
                self.voceros,
            ),
            "Negativo",
        )

    def test_sede_es_neutro(self):
        self.assertIsNone(
            infer_focus_tono(
                "El foro de periodismo se realizó en el auditorio de la Universidad",
                "El evento se realizó en el auditorio de la Universidad de Antioquia.",
                self.marca,
                self.aliases,
                self.voceros,
            )
        )

    def test_draft_upgrade_neutro_to_positivo(self):
        from src.classify import _draft_row

        tono, _sub = _draft_row(
            {"tono": "Neutro", "subtema": "Entrega de becas de sostenimiento"},
            "La Universidad entregó 400 becas de sostenimiento",
            "La Universidad de Antioquia entregó becas a estudiantes de estratos 1 y 2.",
            self.marca,
            self.aliases,
            self.voceros,
        )
        self.assertEqual(tono, "Positivo")

    def test_encuentro_evento_es_positivo(self):
        self.assertEqual(
            infer_focus_tono(
                "Encuentro con líderes comunales",
                "La Universidad de Antioquia realizó un encuentro con líderes comunales del Aburrá.",
                self.marca,
                self.aliases,
                self.voceros,
            ),
            "Positivo",
        )
        self.assertEqual(
            infer_focus_tono(
                "Evento de ciencia abierta",
                "La UdeA participó en el evento de ciencia abierta.",
                self.marca,
                self.aliases,
                self.voceros,
            ),
            "Positivo",
        )


class PromptTests(unittest.TestCase):
    def test_prompt_biases_gestion_to_positivo(self):
        self.assertIn("Si el FOCO HACE la gestión, el tono es Positivo", SYSTEM_PROMPT)
        self.assertIn("NEUTRO — úsalo POCO", SYSTEM_PROMPT)
        self.assertIn("3 a 6 palabras", SYSTEM_PROMPT)
        self.assertIn("NO menciones la MARCA", SYSTEM_PROMPT)
        self.assertIn("saltos de línea", SYSTEM_PROMPT)
        self.assertIn("encuentros", SYSTEM_PROMPT.lower())
        self.assertIn("eventos", SYSTEM_PROMPT.lower())
        self.assertIn("colaboración", SYSTEM_PROMPT.lower())
        self.assertIn("desempleo", SYSTEM_PROMPT.lower())
        self.assertNotIn("Ante duda entre Positivo y Neutro, o entre Negativo y Neutro, elige Neutro", SYSTEM_PROMPT)
        user = build_user_prompt(
            [{"id": 0, "titulo": "Nota", "pasajes": "La UdeA realizó un encuentro.", "resumen": "Cuerpo largo de la nota."}],
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
            voceros=[],
        )
        self.assertIn("CUERPO (CuerpoEs o Resumen; texto completo ya normalizado.", user)
        self.assertIn("Cuerpo largo de la nota.", user)
        self.assertIn("Encuentros, eventos, gestiones", user)
        self.assertIn("SUBTEMA: 3 a 6 palabras, sin el nombre de la marca/alias, distinto del título y de la primera línea.", user)


class CostTests(unittest.TestCase):
    def test_nano_rates(self):
        inp, out, total = estimate_cost_usd(10_000, 2_000)
        self.assertAlmostEqual(inp, 0.001)
        self.assertAlmostEqual(out, 0.0008)
        self.assertAlmostEqual(total, 0.0018)
        stats = ClassifyStats(prompt_tokens=10_000, completion_tokens=2_000)
        self.assertAlmostEqual(stats.cost_total_usd, 0.0018)


class ThemeCssTests(unittest.TestCase):
    def test_css_does_not_force_white_app_background(self):
        from pathlib import Path

        from app import _APP_CSS

        self.assertNotIn("background: #FFFFFF", _APP_CSS)
        self.assertNotIn("background: #ffffff", _APP_CSS)
        self.assertNotIn("1d9bf0", _APP_CSS.lower())
        self.assertIn("ff6a00", _APP_CSS.lower())
        self.assertIn("gx-summary-flag", _APP_CSS)
        self.assertIn("stprogress", _APP_CSS.lower())
        self.assertIn("gx-progress-status", _APP_CSS)
        src = Path("app.py").read_text(encoding="utf-8")
        self.assertNotIn("st.sidebar", src)
        self.assertIsNone(__import__("re").search(r"\.progress\([^)]*text\s*=", src))
        self.assertIn("CuerpoEs", src)
        self.assertIn("foco_y_ajuste", src)
        self.assertIn("dl_after_progress", src)
        self.assertNotIn("Uso / clientes", src)
        self.assertNotIn("render_uso_expander", src)
        cfg = Path(".streamlit/config.toml").read_text(encoding="utf-8")
        self.assertIn("FF6A00", cfg)
        self.assertIn("#000000", cfg)


UNINORTE = "Universidad del Norte"
UNINORTE_ALIASES = ["Uninorte"]
DESEMPLEO_TITULO = "Desempleo juvenil en Barranquilla llega al 18,8 %"
DESEMPLEO_EXTRACT = (
    "Así lo revela un informe de Goyn Barranquilla y NuestraBarranquilla, "
    "elaborado con la colaboración de la Universidad del Norte, la "
    "Universidad Simón Bolívar y Fundesarrollo"
)
DESEMPLEO_CUERPO = (
    "El desempleo de los jóvenes en el distrito alcanzó el 18,8 por ciento.\n\n"
    + DESEMPLEO_EXTRACT
    + "."
)

# Bloque SUBTEMA del commit 18b79f6. El techo de palabras es lo único que
# puede diferir (5 → 6); el resto del texto debe coincidir carácter a carácter.
FROZEN_SUBTEMA_PROMPT_18B79F6 = """SUBTEMA
Frase nominal CORTA y completa (típicamente 3 a 5 palabras; nunca larga)
en español colombiano que condensa el ÁNGULO del hecho a partir del CUERPO
completo (CuerpoEs o Resumen). No es un collage, no es un recorte del título
y no es la primera línea del cuerpo.
- 3 a 5 palabras. Bien: "Entrega de becas de sostenimiento". Mal: una
  oración larga o un titular reescrito.
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

"""

FROZEN_SUBTEMA_PROMPT = FROZEN_SUBTEMA_PROMPT_18B79F6.replace(
    "3 a 5 palabras", "3 a 6 palabras"
)

FROZEN_CLEAN_SUBTEMA = {
    (
        "Avance de obra de",
        "Otro titular sobre laboratorios en el campus",
        "Avanzó la obra de laboratorios en el campus central.\n"
        "Se instalaron equipos nuevos de investigación aplicada.",
        "UdeA",
        None,
    ): "Avance de obra",
    (
        "Universidad de Antioquia entrega becas de sostenimiento",
        "Otra nota sobre ranking QS nacional de universidades",
        "Se entregaron becas de sostenimiento a estudiantes de estratos 1 y 2.\n"
        "El giro cubre alimentación y transporte durante el semestre en curso.",
        "Universidad de Antioquia",
        ("UdeA",),
    ): "Entrega becas de sostenimiento",
    (
        "Informe de desempleo juvenil",
        DESEMPLEO_TITULO,
        DESEMPLEO_CUERPO,
        UNINORTE,
        ("Uninorte",),
    ): "Informe de desempleo juvenil",
}


class CollaboratorTonoTests(unittest.TestCase):
    def test_university_collaborator_on_unemployment_study_is_not_negativo(self):
        hinted = infer_focus_tono(
            DESEMPLEO_TITULO,
            DESEMPLEO_CUERPO,
            UNINORTE,
            UNINORTE_ALIASES,
            [],
        )
        self.assertNotEqual(hinted, "Negativo")
        self.assertIn(hinted, {None, "Positivo"})

        from src.classify import _draft_row

        tono, sub = _draft_row(
            {
                "tono": "Negativo",
                "subtema": DESEMPLEO_EXTRACT,
            },
            DESEMPLEO_TITULO,
            DESEMPLEO_CUERPO,
            UNINORTE,
            UNINORTE_ALIASES,
            [],
        )
        self.assertNotEqual(tono, "Negativo")
        self.assertIn(tono, {"Neutro", "Positivo"})
        self.assertGreaterEqual(len(sub.split()), 3)
        self.assertLessEqual(len(sub.split()), 6)
        self.assertFalse(same_folded_phrase(sub, DESEMPLEO_TITULO))
        self.assertFalse(same_folded_phrase(sub, first_content_line(DESEMPLEO_CUERPO)))
        folded = fold_text(sub)
        self.assertNotIn("norte", folded)
        self.assertNotIn("uninorte", folded)

    def test_pipeline_keeps_collaborator_out_of_negativo(self):
        def fake_batch(client, items, **kwargs):
            return {
                items[0]["id"]: {
                    "id": items[0]["id"],
                    "tono": "Negativo",
                    "subtema": "Informe de desempleo juvenil",
                }
            }

        with patch("src.classify.OpenAI"), patch(
            "src.classify.classify_batch", side_effect=fake_batch
        ):
            tonos, subs, temas, _stats = classify_rows(
                [DESEMPLEO_TITULO],
                [DESEMPLEO_CUERPO],
                marca=UNINORTE,
                aliases=UNINORTE_ALIASES,
                voceros=[],
                api_key="test",
                batch_size=10,
            )
        self.assertNotEqual(tonos[0], "Negativo")
        self.assertIn(tonos[0], {"Neutro", "Positivo"})
        self.assertEqual(subs[0], "Informe de desempleo juvenil")
        self.assertTrue(temas[0])
        self.assertNotEqual(fold_text(temas[0]), fold_text(subs[0]))

    def test_praise_of_collaborator_can_be_positivo(self):
        cuerpo = (
            "El informe sobre desempleo juvenil exaltó el rol de la "
            "Universidad del Norte en la medición."
        )
        self.assertEqual(
            infer_focus_tono(
                DESEMPLEO_TITULO,
                cuerpo,
                UNINORTE,
                UNINORTE_ALIASES,
                [],
            ),
            "Positivo",
        )


class FrozenSubtema18b79f6Tests(unittest.TestCase):
    def test_subtema_prompt_is_18b79f6_except_six_word_ceiling(self):
        start = SYSTEM_PROMPT.index("SUBTEMA\n")
        end = SYSTEM_PROMPT.index("Responde ÚNICAMENTE")
        block = SYSTEM_PROMPT[start:end]
        self.assertEqual(block, FROZEN_SUBTEMA_PROMPT)
        self.assertEqual(
            block.replace("3 a 6 palabras", "3 a 5 palabras"),
            FROZEN_SUBTEMA_PROMPT_18B79F6,
        )
        self.assertIn("típicamente 3 a 6 palabras; nunca larga", block)
        self.assertIn("3 a 6 palabras. Bien:", block)
        self.assertNotIn("no recortes a mitad de sentido", block)
        self.assertNotIn("Prefiere una frase corta completa", block)
        self.assertNotIn("extracto del cuerpo", block)
        self.assertNotIn("Cocha Molina celebró", SYSTEM_PROMPT)
        self.assertIn("NO menciones la MARCA", SYSTEM_PROMPT)
        self.assertIn("saltos de línea", SYSTEM_PROMPT)
        self.assertNotIn("tema_ai", SYSTEM_PROMPT.lower())

    def test_user_prompt_still_sends_full_cuerpo_for_subtema(self):
        user = build_user_prompt(
            [{"id": 0, "titulo": "Titular", "pasajes": "Pasaje del foco.", "resumen": "CUERPO_COMPLETO_XYZ"}],
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
            voceros=[],
        )
        self.assertIn("CUERPO_COMPLETO_XYZ", user)
        self.assertIn("CUERPO (CuerpoEs o Resumen; texto completo ya normalizado.", user)
        self.assertIn("Analiza todo el bloque, no solo la primera oración ni la primera línea)", user)
        self.assertIn("SUBTEMA: 3 a 6 palabras, sin el nombre de la marca/alias, distinto del título y de la primera línea.", user)
        self.assertIn("Analiza el CUERPO completo (ya viene con saltos de línea unidos). No copies el titular ni el arranque.", user)
        self.assertIn("frase nominal de 3 a 6 palabras, sin marca", user)
        self.assertNotIn("máximo 6 palabras, frase corta completa", user)

    def test_clean_subtema_snapshots_18b79f6(self):
        for (raw, titulo, resumen, marca, aliases), expected in FROZEN_CLEAN_SUBTEMA.items():
            out = clean_subtema(
                raw,
                titulo=titulo,
                resumen=resumen,
                marca=marca,
                aliases=list(aliases) if aliases else None,
            )
            self.assertEqual(out, expected, msg=repr(raw))

    def test_six_word_analytical_label_is_kept(self):
        phrases = [
            "Inicio de clases con alimentación escolar",
            "Puesto 8 en ranking QS nacional",
            "Revisión del proyecto Canal del Dique",
        ]
        for phrase in phrases:
            self.assertEqual(len(phrase.split()), 6, msg=phrase)
            out = clean_subtema(
                phrase,
                titulo="Otro titular distinto sobre gestión departamental",
                resumen=(
                    "El departamento arrancó el PAE el primer día de clases para 40 mil niños.\n"
                    "Segundo párrafo de contexto institucional sobre cobertura alimentaria."
                ),
                marca="Gobernación de Sucre",
            )
            self.assertEqual(out, phrase)
            self.assertEqual(len(out.split()), 6)

    def test_style_stays_analytical_label_not_extract(self):
        out = clean_subtema(
            DESEMPLEO_EXTRACT,
            titulo=DESEMPLEO_TITULO,
            resumen=DESEMPLEO_CUERPO,
            marca=UNINORTE,
            aliases=UNINORTE_ALIASES,
        )
        self.assertGreaterEqual(len(out.split()), 3)
        self.assertLessEqual(len(out.split()), 6)
        self.assertFalse(same_folded_phrase(out, DESEMPLEO_EXTRACT))
        self.assertFalse(same_folded_phrase(out, DESEMPLEO_TITULO))
        self.assertFalse(same_folded_phrase(out, first_content_line(DESEMPLEO_CUERPO)))
        clipped_extract = " ".join(DESEMPLEO_EXTRACT.split()[:6])
        self.assertFalse(same_folded_phrase(out, clipped_extract))
        self.assertNotIn(",", out)

    def test_pick_best_subtema_six_words_in_range(self):
        import inspect

        from src.group import pick_best_subtema

        src = inspect.getsource(pick_best_subtema)
        self.assertIn("3 <= n <= 6", src)
        self.assertNotIn("3 <= n <= 5", src)
        six = "Inicio de clases con alimentación escolar"
        out = pick_best_subtema(
            [six, six],
            titulo="Otro titular distinto sobre el PAE departamental",
            resumen=(
                "El departamento arrancó el PAE el primer día de clases para 40 mil niños.\n"
                "Segundo párrafo de contexto institucional."
            ),
            marca="Gobernación de Sucre",
        )
        self.assertEqual(out, six)
        self.assertEqual(len(out.split()), 6)

    def test_clean_subtema_helpers_unchanged(self):
        import inspect

        from src.normalize import (
            MAX_SUBTEMA_WORDS,
            MIN_SUBTEMA_WORDS,
            _clip_subtema_words,
            _fallback_subtema,
            _finalize_subtema,
            _phrase_is_unusable,
        )

        self.assertEqual(MAX_SUBTEMA_WORDS, 6)
        self.assertEqual(MIN_SUBTEMA_WORDS, 3)
        self.assertTrue(inspect.getsource(clean_subtema).startswith("def clean_subtema("))
        self.assertIn("needs_fallback", inspect.getsource(clean_subtema))
        self.assertIn("_fallback_subtema", inspect.getsource(clean_subtema))
        self.assertIn("MAX_SUBTEMA_WORDS", inspect.getsource(_clip_subtema_words))
        unusable = inspect.getsource(_phrase_is_unusable)
        self.assertIn("looks_like_collage", unusable)
        self.assertIn("looks_like_title_or_lead", unusable)
        self.assertIn("looks_like_truncated_clause", unusable)
        self.assertIn("looks_like_body_extract", unusable)
        self.assertNotIn("extracto", unusable)
        self.assertNotIn("looks_like_extract", unusable)
        self.assertIn("strip_brand_mentions", inspect.getsource(_finalize_subtema))
        self.assertIn("_source_tokens", inspect.getsource(_fallback_subtema))
        self.assertIn("_nominal_from_title", inspect.getsource(_fallback_subtema))
        self.assertIn("_fallback_usable", inspect.getsource(_fallback_subtema))


class IncompleteExtractTests(unittest.TestCase):
    marca = UNINORTE
    aliases = UNINORTE_ALIASES
    eco_titulo = "Estudiantes de Uninorte participan en Eco Challenge STEM"
    eco_cuerpo = (
        "Durante el encuentro los estudiantes tienen la oportunidad de "
        "presentar prototipos de energía solar.\n"
        "El equipo de ingeniería representó a la institución en la competencia."
    )

    def test_rejects_dangling_finite_verb_extract(self):
        raw = "Durante el encuentro los estudiantes tienen"
        out = clean_subtema(
            raw,
            titulo=self.eco_titulo,
            resumen=self.eco_cuerpo,
            marca=self.marca,
            aliases=self.aliases,
        )
        self.assertTrue(out)
        self.assertNotEqual(fold_text(out.split()[-1]), "tienen")
        self.assertNotEqual(ocr_fold(out), ocr_fold(raw))
        self.assertFalse(ocr_fold(out).startswith("durante"))
        self.assertLessEqual(len(out.split()), 6)
        self.assertGreaterEqual(len(out.split()), 3)

    def test_eco_challenge_falls_back_to_nominal_not_title_dump(self):
        out = clean_subtema(
            "Durante el encuentro los estudiantes tienen",
            titulo=self.eco_titulo,
            resumen=self.eco_cuerpo,
            marca=self.marca,
            aliases=self.aliases,
        )
        folded = fold_text(out)
        self.assertIn("participacion", folded)
        self.assertIn("eco", folded)
        self.assertIn("challenge", folded)
        self.assertNotIn("tienen", folded)
        self.assertNotIn("durante", folded)
        self.assertFalse(same_folded_phrase(out, self.eco_titulo))
        self.assertNotIn("estudiantes participan", folded)
        self.assertLessEqual(len(out.split()), 6)
        self.assertGreaterEqual(len(out.split()), 3)

    def test_keeps_model_nominal_eco_challenge(self):
        good = "Participación en Eco Challenge"
        out = clean_subtema(
            good,
            titulo=self.eco_titulo,
            resumen=self.eco_cuerpo,
            marca=self.marca,
            aliases=self.aliases,
        )
        self.assertEqual(ocr_fold(out), ocr_fold(good))

    def test_subtema_prompt_only_differs_in_word_ceiling(self):
        start = SYSTEM_PROMPT.index("SUBTEMA\n")
        end = SYSTEM_PROMPT.index("Responde ÚNICAMENTE")
        block = SYSTEM_PROMPT[start:end]
        self.assertEqual(
            block.replace("3 a 6 palabras", "3 a 5 palabras"),
            FROZEN_SUBTEMA_PROMPT_18B79F6,
        )
        self.assertNotIn("no recortes a mitad de sentido", block)
        self.assertNotIn("extracto del cuerpo", block)
        self.assertNotIn(" Prefiere una frase corta completa", block)


class TemaGroupingTests(unittest.TestCase):
    def test_singleton_is_slightly_more_general(self):
        sub = "Entrega de becas de sostenimiento"
        tema = broaden_subtema(sub)
        self.assertEqual(tema, "Becas y apoyos estudiantiles")
        self.assertNotEqual(tema.lower(), sub.lower())
        self.assertGreaterEqual(len(tema.split()), 2)
        self.assertLessEqual(len(tema.split()), 4)

    def test_similar_subtemas_share_tema(self):
        subs = [
            "Entrega de becas de sostenimiento",
            "Becas de sostenimiento para estratos 1 y 2",
            "Protesta por alza de matrícula",
            "Alza de matrícula y protestas estudiantiles",
        ]
        temas = assign_temas(
            subs,
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
        )
        self.assertEqual(temas[0], temas[1])
        self.assertEqual(temas[2], temas[3])
        self.assertNotEqual(temas[0], temas[2])
        self.assertEqual(temas[0], "Becas y apoyos estudiantiles")
        self.assertLessEqual(len(temas[0].split()), 4)
        clusters = cluster_subtema_indices(subs)
        sizes = sorted(len(c) for c in clusters)
        self.assertEqual(sizes, [2, 2])

    def test_tema_does_not_rewrite_subtema_input(self):
        subs = ["Entrega de becas de sostenimiento", "Foro de periodismo en el campus"]
        original = list(subs)
        temas = assign_temas(subs, marca="Universidad de Antioquia", aliases=["UdeA"])
        self.assertEqual(subs, original)
        self.assertNotEqual(temas[0], subs[0])
        self.assertTrue(all(2 <= len(t.split()) <= 4 for t in temas))

    def test_tema_strips_brand_and_sentence_case(self):
        temas = assign_temas(
            ["Universidad de Antioquia entrega becas de sostenimiento"],
            marca="Universidad de Antioquia",
            aliases=["UdeA"],
        )
        folded = temas[0].lower()
        self.assertNotIn("antioquia", folded)
        self.assertNotIn("udea", folded)
        self.assertEqual(temas[0][:1], temas[0][:1].upper())
        self.assertLessEqual(len(temas[0].split()), 4)

    def test_excel_column_order_tono_tema_subtema(self):
        df = pd.DataFrame(
            {
                "Título": ["La Universidad entregó becas de sostenimiento"],
                "CuerpoEs": [
                    "La Universidad de Antioquia entregó becas a estudiantes de estratos 1 y 2."
                ],
            }
        )

        def fake_batch(client, items, **kwargs):
            return {
                items[0]["id"]: {
                    "id": items[0]["id"],
                    "tono": "Positivo",
                    "subtema": "Entrega de becas de sostenimiento",
                }
            }

        with patch("src.classify.OpenAI"), patch(
            "src.classify.classify_batch", side_effect=fake_batch
        ):
            out, _stats = classify_dataframe(
                df,
                "Título",
                "CuerpoEs",
                marca="Universidad de Antioquia",
                aliases=["UdeA"],
                voceros=[],
                api_key="test",
            )
        self.assertEqual(list(out.columns)[-3:], ["tono_AI", "tema_AI", "subtema_AI"])
        self.assertEqual(out["tono_AI"].iloc[0], "Positivo")
        self.assertEqual(out["tema_AI"].iloc[0], "Becas y apoyos estudiantiles")
        self.assertEqual(out["subtema_AI"].iloc[0], "Entrega de becas de sostenimiento")

    def test_rejects_xy_collage_from_first_tokens(self):
        from src.tema import is_xy_token_collage

        cocha = "Cocha Molina celebró"
        tema_cocha = broaden_subtema(cocha)
        self.assertFalse(is_xy_token_collage(tema_cocha, cocha))
        self.assertNotEqual(fold_text(tema_cocha), "cocha y molina")
        self.assertNotIn(" y ", f" {fold_text(tema_cocha)} ")
        self.assertLessEqual(len(tema_cocha.split()), 4)
        self.assertGreaterEqual(len(tema_cocha.split()), 2)
        self.assertIn("celebracion", fold_text(tema_cocha))
        self.assertIn("cocha", fold_text(tema_cocha))

        tamizaje = "Tamizaje nutricional en Soledad"
        tema_tam = broaden_subtema(tamizaje)
        self.assertFalse(is_xy_token_collage(tema_tam, tamizaje))
        self.assertNotEqual(fold_text(tema_tam), "tamizaje y nutricional")
        self.assertNotIn(" y ", f" {fold_text(tema_tam)} ")
        self.assertLessEqual(len(tema_tam.split()), 4)
        self.assertEqual(fold_text(tema_tam), "tamizaje nutricional")

        self.assertTrue(is_xy_token_collage("Cocha y molina", cocha))
        self.assertTrue(is_xy_token_collage("Tamizaje y nutricional", tamizaje))
        self.assertFalse(is_xy_token_collage("Becas y apoyos estudiantiles", "Entrega de becas"))

    def test_tema_from_sample_subtemas_sensible(self):
        subs = [
            "Cocha Molina celebró",
            "Tamizaje nutricional en Soledad",
            "Entrega de becas de sostenimiento",
        ]
        original = list(subs)
        temas = assign_temas(subs, marca="Universidad de Antioquia", aliases=["UdeA"])
        self.assertEqual(subs, original)
        self.assertEqual(fold_text(temas[0]), "celebracion cocha molina")
        self.assertEqual(fold_text(temas[1]), "tamizaje nutricional")
        self.assertEqual(temas[2], "Becas y apoyos estudiantiles")
        for tema in temas:
            self.assertLessEqual(len(tema.split()), 4)
            self.assertGreaterEqual(len(tema.split()), 2)

    def test_clean_tema_rejects_xy_collage_input(self):
        from src.tema import clean_tema

        out = clean_tema(
            "Cocha y molina",
            source_subtema="Cocha Molina celebró",
        )
        self.assertNotEqual(fold_text(out), "cocha y molina")
        self.assertIn("celebracion", fold_text(out))


class UsageLogTests(unittest.TestCase):
    def test_record_and_read_csv(self):
        import tempfile
        from pathlib import Path

        from src.usage import load_recent_runs, record_run

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uso_clientes.csv"
            row = record_run(
                marca="Universidad de Antioquia",
                aliases=["UdeA", "U. de Antioquia"],
                n_rows=12,
                tono_counts={"Positivo": 7, "Negativo": 2, "Neutro": 3},
                model="gpt-4.1-nano-2025-04-14",
                elapsed_s=4.2,
                cost_usd=0.0018,
                path=path,
            )
            self.assertEqual(row["marca"], "Universidad de Antioquia")
            self.assertIn("UdeA", row["aliases"])
            self.assertEqual(row["n_rows"], "12")
            self.assertEqual(row["positivo"], "7")
            loaded = load_recent_runs(path=path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["neutro"], "3")

    def test_skips_email_without_secrets(self):
        from src.usage import maybe_notify_email

        status = maybe_notify_email(
            {"marca": "UdeA", "n_rows": "1", "timestamp": "t"},
            secrets={},
        )
        self.assertEqual(status, "skipped_no_secrets")

    def test_default_notify_email_and_override(self):
        from src.usage import DEFAULT_USAGE_NOTIFY_EMAIL, notify_destination

        self.assertEqual(DEFAULT_USAGE_NOTIFY_EMAIL, "cortesalexander8@gmail.com")
        self.assertEqual(notify_destination({}), DEFAULT_USAGE_NOTIFY_EMAIL)
        self.assertEqual(
            notify_destination({"USAGE_NOTIFY_EMAIL": "otro@correo.com"}),
            "otro@correo.com",
        )

    def test_sends_via_resend_to_default_when_key_present(self):
        from unittest.mock import patch

        from src.usage import DEFAULT_USAGE_NOTIFY_EMAIL, maybe_notify_email

        with patch("src.usage._send_resend") as send:
            status = maybe_notify_email(
                {"marca": "UdeA", "n_rows": "3", "timestamp": "t"},
                secrets={"RESEND_API_KEY": "re_test", "SMTP_FROM": "from@ejemplo.com"},
            )
        self.assertEqual(status, "resend")
        send.assert_called_once()
        kwargs = send.call_args.kwargs
        self.assertEqual(kwargs["to_addr"], DEFAULT_USAGE_NOTIFY_EMAIL)
        self.assertEqual(kwargs["api_key"], "re_test")

    def test_sends_via_smtp_to_default_when_host_present(self):
        from unittest.mock import patch

        from src.usage import DEFAULT_USAGE_NOTIFY_EMAIL, maybe_notify_email

        with patch("src.usage._send_smtp") as send:
            status = maybe_notify_email(
                {"marca": "UdeA", "n_rows": "3", "timestamp": "t"},
                secrets={
                    "SMTP_HOST": "smtp.ejemplo.com",
                    "SMTP_USER": "usuario",
                    "SMTP_PASSWORD": "from-secrets-only",
                    "SMTP_FROM": "from@ejemplo.com",
                },
            )
        self.assertEqual(status, "smtp")
        send.assert_called_once()
        kwargs = send.call_args.kwargs
        self.assertEqual(kwargs["to_addr"], DEFAULT_USAGE_NOTIFY_EMAIL)
        self.assertEqual(kwargs["password"], "from-secrets-only")
        self.assertEqual(kwargs["host"], "smtp.ejemplo.com")

    def test_notify_email_secret_overrides_default(self):
        from unittest.mock import patch

        from src.usage import maybe_notify_email

        with patch("src.usage._send_smtp") as send:
            maybe_notify_email(
                {"marca": "UdeA", "n_rows": "1", "timestamp": "t"},
                secrets={
                    "USAGE_NOTIFY_EMAIL": "otro@correo.com",
                    "SMTP_HOST": "smtp.ejemplo.com",
                },
            )
        self.assertEqual(send.call_args.kwargs["to_addr"], "otro@correo.com")

    def test_no_hardcoded_smtp_passwords_in_source(self):
        from pathlib import Path

        from src.usage import DEFAULT_USAGE_NOTIFY_EMAIL

        blob = (
            Path("src/usage.py").read_text(encoding="utf-8")
            + Path("app.py").read_text(encoding="utf-8")
            + Path("README.md").read_text(encoding="utf-8")
        )
        self.assertIn(DEFAULT_USAGE_NOTIFY_EMAIL, blob)
        self.assertIn("SMTP_PASSWORD", blob)
        self.assertNotIn("SMTP_PASSWORD = \"", Path("src/usage.py").read_text(encoding="utf-8"))
        self.assertNotIn("SMTP_PASSWORD='", Path("src/usage.py").read_text(encoding="utf-8"))
        self.assertNotIn("sk-", Path("src/usage.py").read_text(encoding="utf-8"))
        self.assertNotIn("re_prod", Path("src/usage.py").read_text(encoding="utf-8").lower())

    def test_record_run_and_notify_writes_skipped_no_secrets(self):
        import tempfile
        from pathlib import Path

        from src.usage import load_recent_runs, record_run_and_notify

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uso_clientes.csv"
            row = record_run_and_notify(
                marca="Universidad de Antioquia",
                aliases=["UdeA"],
                n_rows=4,
                tono_counts={"Positivo": 2, "Negativo": 1, "Neutro": 1},
                model="gpt-4.1-nano-2025-04-14",
                elapsed_s=1.5,
                cost_usd=0.001,
                secrets={},
                path=path,
            )
            self.assertEqual(row["email_status"], "skipped_no_secrets")
            loaded = load_recent_runs(path=path)
            self.assertEqual(loaded[0]["email_status"], "skipped_no_secrets")
            header = path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("email_status", header)

    def test_nested_resend_secret_and_default_from(self):
        from src.usage import DEFAULT_USAGE_NOTIFY_EMAIL, DEFAULT_USAGE_NOTIFY_FROM, maybe_notify_email

        with patch("src.usage._send_resend") as send:
            status = maybe_notify_email(
                {"marca": "UdeA", "n_rows": "2", "timestamp": "t"},
                secrets={"resend": {"api_key": "re_nested"}},
            )
        self.assertEqual(status, "resend")
        send.assert_called_once()
        kwargs = send.call_args.kwargs
        self.assertEqual(kwargs["api_key"], "re_nested")
        self.assertEqual(kwargs["to_addr"], DEFAULT_USAGE_NOTIFY_EMAIL)
        self.assertEqual(kwargs["from_addr"], DEFAULT_USAGE_NOTIFY_FROM)
        self.assertEqual(DEFAULT_USAGE_NOTIFY_FROM, "onboarding@resend.dev")

    def test_prefers_resend_over_smtp(self):
        from src.usage import maybe_notify_email

        with patch("src.usage._send_resend") as send_resend, patch(
            "src.usage._send_smtp"
        ) as send_smtp:
            status = maybe_notify_email(
                {"marca": "UdeA", "n_rows": "1", "timestamp": "t"},
                secrets={
                    "RESEND_API_KEY": "re_test",
                    "SMTP_HOST": "smtp.ejemplo.com",
                    "SMTP_PASSWORD": "from-secrets-only",
                },
            )
        self.assertEqual(status, "resend")
        send_resend.assert_called_once()
        send_smtp.assert_not_called()

    def test_email_error_is_recorded_not_raised(self):
        import tempfile
        from pathlib import Path

        from src.usage import record_run_and_notify

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "uso_clientes.csv"
            with patch("src.usage._send_resend", side_effect=RuntimeError("boom resend")):
                row = record_run_and_notify(
                    marca="UdeA",
                    n_rows=1,
                    secrets={"RESEND_API_KEY": "re_test"},
                    path=path,
                )
        self.assertTrue(row["email_status"].startswith("error:"))
        self.assertIn("boom resend", row["email_status"])

    def test_smtp_error_does_not_raise_from_maybe_notify(self):
        from src.usage import maybe_notify_email

        with patch("src.usage._send_smtp", side_effect=OSError("smtp down")):
            status = maybe_notify_email(
                {"marca": "UdeA", "n_rows": "1", "timestamp": "t"},
                secrets={"SMTP_HOST": "smtp.ejemplo.com"},
            )
        self.assertTrue(status.startswith("error:"))
        self.assertIn("smtp down", status)


if __name__ == "__main__":
    unittest.main()
