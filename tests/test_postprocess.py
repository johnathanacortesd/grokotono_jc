"""Pruebas locales de postproceso (sin API)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.classify import ClassifyStats, classify_rows, estimate_cost_usd
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
from src.prompts import SYSTEM_PROMPT


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
        self.assertLessEqual(len(out.split()), 5)

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
        self.assertLessEqual(len(out.split()), 5)
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

            tonos, subs, _stats = classify_rows(
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


class PromptTests(unittest.TestCase):
    def test_prompt_biases_gestion_to_positivo(self):
        self.assertIn("Si el FOCO HACE la gestión, el tono es Positivo", SYSTEM_PROMPT)
        self.assertIn("NEUTRO — úsalo POCO", SYSTEM_PROMPT)
        self.assertIn("3 a 5 palabras", SYSTEM_PROMPT)
        self.assertIn("NO menciones la MARCA", SYSTEM_PROMPT)
        self.assertIn("saltos de línea", SYSTEM_PROMPT)
        self.assertNotIn("Ante duda entre Positivo y Neutro, o entre Negativo y Neutro, elige Neutro", SYSTEM_PROMPT)


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
        self.assertIn("1d9bf0", _APP_CSS.lower())
        self.assertIn("stprogress", _APP_CSS.lower())
        self.assertIn("gx-progress-status", _APP_CSS)
        src = Path("app.py").read_text(encoding="utf-8")
        self.assertNotIn("st.sidebar", src)
        self.assertIsNone(__import__("re").search(r"\.progress\([^)]*text\s*=", src))
        self.assertIn("CuerpoEs", src)
        self.assertIn("foco_y_ajuste", src)


if __name__ == "__main__":
    unittest.main()
