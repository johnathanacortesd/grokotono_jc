"""Pruebas locales de postproceso (sin API)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.group import cluster_indices, positivo_first, propagate_labels
from src.io_xlsx import dataframe_to_xlsx_bytes, guess_resumen_column, guess_title_column, read_xlsx
from src.normalize import (
    canonicalize_tono,
    clean_subtema,
    mentions_target,
    sentence_case,
    strip_dangling,
)


class NormalizeTests(unittest.TestCase):
    def test_dangling_endings(self):
        words = strip_dangling("Alimentación escolar de la".split())
        self.assertEqual(words[-1].lower(), "escolar")

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

            tonos, subs = classify_rows(
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


if __name__ == "__main__":
    unittest.main()
