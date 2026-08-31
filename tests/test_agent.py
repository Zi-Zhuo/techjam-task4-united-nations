from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from starter.agent import Agent


class KeywordEncoder:
    """Tiny deterministic encoder with the SentenceTransformer interface."""

    vocabulary = ("shoe", "leather", "running", "blue")

    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.encoded_batches.append(texts)
        return np.asarray(
            [[float(text.lower().count(word)) for word in self.vocabulary] for text in texts],
            dtype=np.float32,
        )


class SynonymEncoder:
    """Maps different lexical forms to the same semantic dimension."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([
                float("running" in lowered or "jogging" in lowered),
                float("leather" in lowered),
            ])
        return np.asarray(rows, dtype=np.float32)


def _write_catalog(path: Path) -> None:
    products = (
        {
            "parent_asin": "LEATHER",
            "title": "Leather formal shoe",
            "categories": ["Shoes"],
            "features": ["leather upper"],
            "description": [],
            "details": {},
            "store": "Example",
        },
        {
            "parent_asin": "RUNNING",
            "title": "Blue running shoe",
            "categories": ["Shoes"],
            "features": ["running comfort"],
            "description": [],
            "details": {},
            "store": "Example",
        },
    )
    path.write_text("".join(json.dumps(product) + "\n" for product in products), encoding="utf-8")


class AgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        _write_catalog(self.catalog_path)
        self.agent = Agent(
            self.catalog_path,
            encoder=KeywordEncoder(),
            cache_dir=None,
            candidate_count=10,
            dense_weight=1.0,
        )
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        self.agent.connection.close()
        self.temporary_directory.cleanup()

    def test_questions_adapt_to_category_and_disclosed_attributes(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 2)
        second = self.agent.respond("session", "Leather would be good.", 2, 2)
        third = self.agent.respond("session", "I want blue for running.", 3, 2)

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertNotEqual(second["ask_attribute"], "material")
        self.assertNotIn(third["ask_attribute"], {"color", "use_case", "material"})
        self.assertIn("features", first["message"].lower())
        self.assertEqual(third["recommendations"][0]["parent_asin"], "RUNNING")
        self.assertEqual(self.agent._sessions["session"].messages, [
            "I need a shoe.",
            "Leather would be good.",
            "I want blue for running.",
        ])

    def test_high_value_questions_precede_sparse_attributes(self) -> None:
        first = self.agent.respond("session", "I need a shoe for running in size 10.", 1, 2)
        second = self.agent.respond(
            "session",
            "I don't have an additional preference for feature.",
            2,
            2,
        )

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(second["ask_attribute"], "material")

    def test_boundary_answer_marks_attribute_as_answered(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 2)
        second = self.agent.respond(
            "session", f"I don't have a preference for {first['ask_attribute']}.", 2, 2
        )

        self.assertNotEqual(second["ask_attribute"], first["ask_attribute"])

    def test_no_preference_does_not_create_a_catalog_exclusion(self) -> None:
        response = self.agent.respond(
            "session",
            "I don't have an additional preference for brand.",
            1,
            2,
        )

        state = self.agent._sessions["session"]
        self.assertEqual(state.excluded_values, {})
        self.assertEqual(len(response["recommendations"]), 2)

    def test_use_your_judgment_does_not_create_a_catalog_exclusion(self) -> None:
        self.agent.respond(
            "session",
            "I don't have a preference for size; please use your judgment.",
            1,
            2,
        )

        self.assertEqual(self.agent._sessions["session"].excluded_values, {})

    def test_explicit_exclusion_filters_matching_products(self) -> None:
        response = self.agent.respond("session", "I want shoes but do not want blue.", 1, 2)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "LEATHER")
        self.assertNotIn(
            "RUNNING", {item["parent_asin"] for item in response["recommendations"]}
        )
        state = self.agent._sessions["session"]
        self.assertEqual(state.excluded_values["color"], {"blue"})
        self.assertIn("color", state.disclosed_attributes)
        self.assertNotIn("blue", self.agent._retrieval_query(state).lower())

    def test_later_acceptance_reverses_an_exclusion(self) -> None:
        self.agent.respond("session", "I need shoes but do not want blue.", 1, 2)
        response = self.agent.respond("session", "Actually, blue is fine for running.", 2, 2)

        self.assertNotIn("blue", self.agent._sessions["session"].excluded_values.get("color", set()))
        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING")

    def test_dense_retrieval_recovers_a_semantic_only_candidate(self) -> None:
        agent = Agent(
            self.catalog_path,
            encoder=SynonymEncoder(),
            cache_dir=None,
            candidate_count=1,
            dense_weight=0.7,
        )
        try:
            agent.reset("semantic", {})
            response = agent.respond("semantic", "I need something for jogging.", 1, 2)
        finally:
            agent.connection.close()

        # "jogging" is absent from the catalog, so BM25 returns no candidates;
        # the full-catalog dense retriever must recover the running product.
        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING")

    def test_rrf_unions_lexical_and_dense_candidate_pools(self) -> None:
        agent = Agent(
            self.catalog_path,
            encoder=SynonymEncoder(),
            cache_dir=None,
            candidate_count=1,
            dense_weight=0.5,
        )
        try:
            query_embedding = agent._encode(["jogging"], show_progress_bar=False)[0]
            lexical = agent._candidates("leather")
            dense = agent._dense_candidates(query_embedding)
            recommendations = agent._recommend("leather jogging", 2)
        finally:
            agent.connection.close()

        self.assertEqual(lexical[0][1], "LEATHER")
        self.assertEqual(dense[0][1], "RUNNING")
        self.assertEqual(
            {item["parent_asin"] for item in recommendations}, {"LEATHER", "RUNNING"}
        )

    def test_dense_weight_does_not_overwhelm_the_best_lexical_match(self) -> None:
        self.agent.dense_weight = 0.7
        lexical = [(0, "LEXICAL")]
        dense = [(index, f"DENSE-{index}") for index in range(1, 101)]

        scores = self.agent._fuse_rankings(lexical, dense)
        order = sorted(scores, key=lambda index: -scores[index])

        # Regression: with the former (1-weight)/weight split, 82 dense-only
        # candidates ranked ahead of lexical rank 1 when dense_weight was 0.7.
        self.assertEqual(order[0], 0)
        self.assertGreater(scores[0], scores[1])

    def test_recommendations_diversify_across_turns(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 1)
        second = self.agent.respond("session", "I am still considering shoes.", 2, 1)

        first_id = first["recommendations"][0]["parent_asin"]
        second_id = second["recommendations"][0]["parent_asin"]
        self.assertNotEqual(second_id, first_id)
        self.assertEqual(
            self.agent._sessions["session"].recommended_ids,
            {first_id, second_id},
        )

    def test_diversification_reuses_seen_results_only_as_fallback(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 2)
        second = self.agent.respond("session", "I am still considering shoes.", 2, 2)

        self.assertEqual(len(first["recommendations"]), 2)
        self.assertEqual(len(second["recommendations"]), 2)
        self.assertEqual(
            {item["parent_asin"] for item in second["recommendations"]},
            {item["parent_asin"] for item in first["recommendations"]},
        )

    def test_intent_override_resets_recommendation_history(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 1)
        first_id = first["recommendations"][0]["parent_asin"]

        self.agent.respond(
            "session",
            "Actually, ignore my earlier preference. I still need a shoe.",
            2,
            1,
        )

        state = self.agent._sessions["session"]
        self.assertIn(first_id, state.recommended_ids)
        self.assertEqual(len(state.recommended_ids), 1)

    def test_intent_override_discards_intermediate_preferences(self) -> None:
        self.agent.respond("session", "I need a shoe.", 1, 2)
        self.agent.respond("session", "Leather would be good.", 2, 2)
        response = self.agent.respond(
            "session", "Actually, ignore my earlier preference. I need a blue running shoe.", 3, 2
        )

        self.assertEqual(self.agent._sessions["session"].messages, [
            "I need a shoe.",
            "Actually, ignore my earlier preference. I need a blue running shoe.",
        ])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING")

    def test_targeted_override_replaces_only_the_affected_attribute(self) -> None:
        self.agent.respond(
            "session", "I need black running shoes under $100.", 1, 2
        )
        response = self.agent.respond(
            "session", "Actually, make them blue.", 2, 2
        )

        state = self.agent._sessions["session"]
        query = self.agent._retrieval_query(state).lower()
        self.assertEqual(state.messages, [
            "I need black running shoes under $100.",
            "Actually, make them blue.",
        ])
        self.assertEqual(state.superseded_values["color"], {"black"})
        self.assertNotIn("black", query)
        self.assertIn("blue", query)
        self.assertIn("running", query)
        self.assertIn("$100", query)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING")

    def test_respond_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            self.agent.respond("missing", "shoe", 1, 1)

    def test_device_is_auto_detected_or_can_be_overridden(self) -> None:
        self.assertIn(self.agent.device, {"cpu", "cuda"})
        forced = Agent(self.catalog_path, encoder=KeywordEncoder(), cache_dir=None, device="cpu")
        self.assertEqual(forced.device, "cpu")
        forced.connection.close()

    def test_catalog_embeddings_are_reused_from_cache(self) -> None:
        cache_dir = Path(self.temporary_directory.name) / "cache"
        first_encoder = KeywordEncoder()
        first = Agent(self.catalog_path, encoder=first_encoder, cache_dir=cache_dir)
        first.reset("first", {})
        first.respond("first", "running shoe", 1, 1)
        first.connection.close()

        second_encoder = KeywordEncoder()
        second = Agent(self.catalog_path, encoder=second_encoder, cache_dir=cache_dir)
        second.reset("second", {})
        second.respond("second", "running shoe", 1, 1)
        second.connection.close()

        self.assertEqual(len(list(cache_dir.glob("catalog-*.npy"))), 1)
        self.assertEqual(len(first_encoder.encoded_batches[0]), 2)
        self.assertEqual(second_encoder.encoded_batches, [["Customer: running shoe"]])


if __name__ == "__main__":
    unittest.main()
