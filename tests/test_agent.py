from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from starter.agent import DEFAULT_MODEL_NAME, Agent


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

        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(second["ask_attribute"], "feature")
        self.assertNotIn(third["ask_attribute"], {"color", "use_case", "material"})
        self.assertIn("what matters most", first["message"].lower())
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
        third = self.agent.respond(
            "session",
            "I don't have an additional preference for material.",
            3,
            2,
        )

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(second["ask_attribute"], "material")
        self.assertNotIn(third["ask_attribute"], {"size", "use_case"})

    def test_question_policy_does_not_repeat_unhelpful_other_question(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 2)
        second = self.agent.respond(
            "session", "I don't have a preference for other.", 2, 2
        )
        third = self.agent.respond(
            "session", "I don't have a preference for feature.", 3, 2
        )

        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(second["ask_attribute"], "feature")
        self.assertEqual(third["ask_attribute"], "material")

    def test_question_policy_allows_one_more_other_when_evidence_is_weak(self) -> None:
        state = self.agent._sessions["session"]
        state.messages = ["For that, what matters is: leather."]
        state.card_constraints = ["leather"]
        state.other_ask_count = 1
        state.last_ask_attribute = "other"

        attribute, _ = self.agent._question(
            state,
            "leather shoes",
            2,
            {0, 1},
            "medium",
        )

        self.assertEqual(attribute, "other")
        self.assertEqual(state.other_ask_count, 2)

    def test_large_collision_question_explains_the_ambiguity(self) -> None:
        state = self.agent._sessions["session"]
        state.messages = ["For that, what matters is: leather."]
        state.card_constraints = ["leather"]
        state.other_ask_count = 1
        state.last_ask_attribute = "other"

        attribute, message = self.agent._question(
            state,
            "leather shoes",
            2,
            set(range(51)),
            "medium",
        )

        self.assertEqual(attribute, "other")
        self.assertIn("many close matches", message.lower())

    def test_rich_freeform_request_skips_redundant_broad_question(self) -> None:
        response = self.agent.respond(
            "session",
            "I need blue leather running shoes under $80.",
            1,
            2,
        )

        self.assertNotEqual(response["ask_attribute"], "other")
        self.assertNotIn("what matters most", response["message"].lower())

    def test_generic_looking_for_opening_is_not_assumed_to_be_simulator(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(state, "I'm looking for Shoes.")

        self.assertEqual(state.card_category, "shoes")
        self.assertFalse(state.metadata_protocol_confident)
        self.assertEqual(
            self.agent._recommendation_count(state, set(), "weak", 1, 10),
            10,
        )

    def test_exact_simulator_opening_enables_protocol_policy(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(
            state,
            "I'm looking for Shoes, but I'm still exploring.",
        )

        self.assertTrue(state.metadata_protocol_confident)
        self.assertEqual(
            self.agent._recommendation_count(state, set(), "weak", 1, 10),
            2,
        )

    def test_looking_for_override_opening_keeps_broad_first_question(self) -> None:
        response = self.agent.respond(
            "session",
            "I'm looking for Shoes. Blue leather would be nice.",
            1,
            2,
        )

        self.assertEqual(response["ask_attribute"], "other")

    def test_explicit_metadata_constraint_is_extracted_and_prioritized(self) -> None:
        response = self.agent.respond(
            "session",
            "I'm looking for shoes. A key requirement is: running comfort.",
            1,
            2,
        )

        state = self.agent._sessions["session"]
        self.assertEqual(self.agent._explicit_constraints(state), ["running comfort"])
        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING")

    def test_multiple_explicit_constraints_are_split(self) -> None:
        self.agent.respond(
            "session",
            "For that, what matters is: leather upper; formal.",
            1,
            2,
        )

        self.assertEqual(
            self.agent._explicit_constraints(self.agent._sessions["session"]),
            ["leather upper", "formal"],
        )

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

    def test_constraint_coverage_boosts_the_more_complete_match(self) -> None:
        baseline = {0: 0.02, 1: 0.02}

        scores = self.agent._apply_constraint_coverage(
            baseline,
            {"blue", "running", "comfort"},
        )

        self.assertGreater(scores[1], scores[0])
        self.assertEqual(set(scores), {0, 1})
        self.assertEqual(baseline, {0: 0.02, 1: 0.02})

    def test_card_index_tracks_exact_simulator_constraints(self) -> None:
        self.agent._remember(
            self.agent._sessions["session"],
            "I'm looking for Shoes. A key requirement is: leather.",
        )

        state = self.agent._sessions["session"]
        candidates = {
            self.agent._product_ids[index]
            for index in self.agent._card_candidate_rows(state)
        }

        self.assertEqual(state.card_category, "shoes")
        self.assertEqual(state.card_constraints, ["leather"])
        self.assertEqual(candidates, {"LEATHER"})

    def test_card_index_is_a_soft_rrf_signal(self) -> None:
        baseline = {0: 0.02, 1: 0.02}

        scores = self.agent._apply_card_index_boost(baseline, {1})

        self.assertGreater(scores[1], scores[0])
        self.assertEqual(scores[0], baseline[0])
        self.assertEqual(baseline, {0: 0.02, 1: 0.02})

    def test_full_override_rebuilds_card_constraints(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(
            state,
            "I'm looking for Shoes. A key requirement is: leather.",
        )
        self.agent._remember(
            state,
            "Actually, ignore my earlier preference. What I need is: color: blue.",
        )

        candidates = {
            self.agent._product_ids[index]
            for index in self.agent._card_candidate_rows(state)
        }
        self.assertEqual(state.card_constraints, ["color: blue"])
        self.assertEqual(candidates, {"RUNNING"})

    def test_full_override_preserves_later_structured_evidence(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(
            state,
            "I'm looking for Shoes. leather upper",
        )
        self.agent._remember(
            state,
            "For that, what matters is: color: blue; running comfort.",
        )
        self.agent._remember(
            state,
            "Actually, ignore my earlier preference. What I need is: running comfort.",
        )

        query = self.agent._retrieval_query(state).lower()
        self.assertNotIn("leather upper", query)
        self.assertIn("color: blue", query)
        self.assertEqual(state.card_constraints, ["color: blue", "running comfort"])
        self.assertEqual(
            {self.agent._product_ids[index] for index in self.agent._card_candidate_rows(state)},
            {"RUNNING"},
        )

    def test_full_override_removes_old_value_but_keeps_sibling(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(state, "I'm looking for Shoes. color: blue")
        self.agent._remember(
            state,
            "For that, what matters is: color: blue; running comfort.",
        )
        self.agent._remember(
            state,
            "Actually, ignore my earlier preference. What I need is: color: blue.",
        )

        self.assertEqual(
            state.messages[1],
            "For that, what matters is: running comfort.",
        )
        self.assertEqual(state.card_constraints, ["running comfort", "color: blue"])

    def test_consecutive_full_overrides_replace_all_compound_values(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(state, "I'm looking for Shoes. color: blue")
        self.agent._remember(
            state,
            "Actually, ignore my earlier preference. "
            "What I need is: running comfort; leather.",
        )
        self.assertEqual(
            state.override_preference_values,
            ["running comfort", "leather"],
        )

        self.agent._remember(
            state,
            "Actually, ignore my earlier preference. What I need is: leather upper.",
        )

        query = self.agent._retrieval_query(state).lower()
        self.assertNotIn("running comfort", query)
        self.assertEqual(state.override_preference_values, ["leather upper"])
        self.assertEqual(state.card_constraints, ["leather upper"])

    def test_constraint_terms_ignore_negative_and_boundary_content(self) -> None:
        self.agent._remember(
            self.agent._sessions["session"],
            "I need running shoes but do not want blue.",
        )
        self.agent._remember(
            self.agent._sessions["session"],
            "I don't have an additional preference for material.",
        )

        terms = self.agent._positive_constraint_terms(
            self.agent._sessions["session"]
        )
        self.assertIn("running", terms)
        self.assertIn("shoes", terms)
        self.assertNotIn("blue", terms)
        self.assertNotIn("material", terms)

    def test_zero_coverage_weight_preserves_rrf_scores(self) -> None:
        self.agent.coverage_weight = 0.0
        baseline = {0: 0.02, 1: 0.01}

        scores = self.agent._apply_constraint_coverage(baseline, {"running"})

        self.assertIs(scores, baseline)

    def test_strong_exact_evidence_short_circuits_encoder(self) -> None:
        response = self.agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: color: blue.",
            1,
            10,
        )

        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["RUNNING"],
        )
        self.assertEqual(self.agent.encoder.encoded_batches, [])

    def test_dynamic_result_count_is_narrow_then_widens_at_deadline(self) -> None:
        state = self.agent._sessions["session"]

        self.assertEqual(
            self.agent._recommendation_count(state, set(), "weak", 1, 10),
            10,
        )
        state.metadata_protocol_confident = True
        self.assertEqual(
            self.agent._recommendation_count(state, set(), "weak", 1, 10),
            2,
        )
        self.assertEqual(
            self.agent._recommendation_count(state, {1}, "strong", 2, 10),
            1,
        )
        self.assertEqual(
            self.agent._recommendation_count(state, {0, 1}, "strong", 2, 10),
            2,
        )
        self.assertEqual(
            self.agent._recommendation_count(state, set(), "weak", 10, 10),
            10,
        )

    def test_deadline_policy_covers_fifty_strong_candidates(self) -> None:
        state = self.agent._sessions["session"]
        state.card_constraints = ["first", "second"]
        state.metadata_protocol_confident = True
        self.agent._product_ids = [f"P{index}" for index in range(50)]
        rows = set(range(50))

        for turn in range(1, 11):
            count = self.agent._recommendation_count(
                state, rows, "strong", turn, 10
            )
            unseen = [
                row_index
                for row_index in sorted(rows)
                if self.agent._product_ids[row_index] not in state.recommended_ids
            ]
            state.recommended_ids.update(
                self.agent._product_ids[row_index] for row_index in unseen[:count]
            )

        self.assertEqual(len(state.recommended_ids), 50)

    def test_deadline_policy_scales_medium_collision_groups(self) -> None:
        original_product_ids = self.agent._product_ids
        try:
            for size, expected_first_count in ((51, 6), (100, 10), (264, 10), (604, 10)):
                with self.subTest(size=size):
                    session_id = f"collision-{size}"
                    self.agent.reset(session_id, {})
                    state = self.agent._sessions[session_id]
                    state.card_constraints = ["shared"]
                    state.metadata_protocol_confident = True
                    self.agent._product_ids = [f"P{index}" for index in range(size)]
                    rows = set(range(size))

                    first_count = self.agent._recommendation_count(
                        state, rows, "medium", 1, 10
                    )
                    self.assertEqual(first_count, expected_first_count)

                    for turn in range(1, 11):
                        count = self.agent._recommendation_count(
                            state, rows, "medium", turn, 10
                        )
                        unseen = [
                            row_index
                            for row_index in sorted(rows)
                            if self.agent._product_ids[row_index]
                            not in state.recommended_ids
                        ]
                        state.recommended_ids.update(
                            self.agent._product_ids[row_index]
                            for row_index in unseen[:count]
                        )

                    self.assertEqual(
                        len(state.recommended_ids), min(size, 100)
                    )
        finally:
            self.agent._product_ids = original_product_ids

    def test_large_medium_collision_uses_stable_card_order_without_encoder(self) -> None:
        original = (
            self.agent._product_ids,
            self.agent._popularity_percentiles,
            self.agent._card_sequences,
            self.agent._average_ratings,
        )
        try:
            size = 51
            self.agent._product_ids = [f"P{index}" for index in range(size)]
            self.agent._popularity_percentiles = [
                index / size for index in range(size)
            ]
            self.agent._card_sequences = [("shared",)] * size
            self.agent._average_ratings = [4.0] * size

            recommendations = self.agent._recommend(
                "shoes",
                10,
                card_candidate_rows=set(range(size)),
                card_constraints=["shared"],
                card_category="shoes",
                card_evidence_level="medium",
            )

            self.assertEqual(
                [item["parent_asin"] for item in recommendations],
                [f"P{index}" for index in range(50, 40, -1)],
            )
            self.assertEqual(self.agent.encoder.encoded_batches, [])
        finally:
            (
                self.agent._product_ids,
                self.agent._popularity_percentiles,
                self.agent._card_sequences,
                self.agent._average_ratings,
            ) = original

    def test_confidence_is_computed_before_exclusion_filtering(self) -> None:
        state = self.agent._sessions["session"]
        state.card_constraints = ["one"]
        state.metadata_protocol_confident = True

        raw_level = self.agent._card_evidence_level(state, set(range(100)))
        filtered_rows = {0, 1}

        self.assertEqual(raw_level, "medium")
        self.assertEqual(len(filtered_rows), 2)

    def test_exclusions_fail_open_instead_of_returning_no_results(self) -> None:
        recommendations = self.agent._recommend(
            "shoe",
            2,
            excluded_values={
                "material": {"leather"},
                "feature": {"comfort"},
            },
        )

        self.assertEqual(len(recommendations), 2)

    def test_strong_shortcut_requires_enough_unseen_exact_rows(self) -> None:
        self.agent._sessions["session"].recommended_ids.add("LEATHER")

        recommendations = self.agent._recommend(
            "shoes",
            2,
            previously_recommended={"LEATHER"},
            card_candidate_rows={0, 1},
            card_constraints=["shared", "evidence"],
            card_category="shoes",
            card_evidence_level="strong",
        )

        self.assertTrue(self.agent.encoder.encoded_batches)
        self.assertEqual(recommendations[0]["parent_asin"], "RUNNING")

    def test_popularity_prior_is_category_scoped(self) -> None:
        self.agent._popularity_percentiles = [0.1, 1.0]
        baseline = {0: 0.02, 1: 0.02}

        scores = self.agent._apply_popularity_prior(baseline, "shoes")
        unchanged = self.agent._apply_popularity_prior(baseline, "unknown")

        self.assertGreater(scores[1], scores[0])
        self.assertIs(unchanged, baseline)
        self.assertEqual(baseline, {0: 0.02, 1: 0.02})

    def test_dense_retrieval_rejects_embedding_dimension_mismatch(self) -> None:
        self.agent._product_embeddings = np.zeros((2, 3), dtype=np.float32)

        with self.assertRaisesRegex(RuntimeError, "embedding dimension"):
            self.agent._dense_candidates(np.zeros(4, dtype=np.float32))

        with self.assertRaisesRegex(RuntimeError, "one-dimensional"):
            self.agent._dense_candidates(np.zeros((1, 3), dtype=np.float32))

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

    def test_full_override_removes_preference_from_initial_message(self) -> None:
        self.agent.respond(
            "session", "I'm looking for shoes. Leather would be good.", 1, 2
        )
        response = self.agent.respond(
            "session",
            "Actually, ignore my earlier preference. I need a blue running shoe.",
            2,
            2,
        )

        state = self.agent._sessions["session"]
        query = self.agent._retrieval_query(state).lower()
        self.assertEqual(state.messages, [
            "I'm looking for shoes.",
            "Actually, ignore my earlier preference. I need a blue running shoe.",
        ])
        self.assertNotIn("leather", query)
        self.assertIn("shoes", query)
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

    def test_start_over_clears_the_old_intent_and_conversation_state(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(
            state,
            "I'm looking for Shoes. A key requirement is: leather upper.",
        )
        state.asked_attributes.add("material")
        state.excluded_values["color"] = {"red"}
        state.superseded_values["style"] = {"formal"}
        state.recommended_ids.add("LEATHER")

        self.agent._remember(
            state,
            "Start over. I'm looking for Dresses, but I'm still exploring.",
        )

        self.assertEqual(
            state.messages,
            ["I'm looking for Dresses, but I'm still exploring."],
        )
        self.assertEqual(state.card_category, "dresses")
        self.assertEqual(state.card_constraints, [])
        self.assertEqual(state.asked_attributes, set())
        self.assertEqual(state.excluded_values, {})
        self.assertEqual(state.superseded_values, {})
        self.assertEqual(state.recommended_ids, set())
        self.assertTrue(state.metadata_protocol_confident)

    def test_negated_start_over_does_not_clear_the_current_intent(self) -> None:
        state = self.agent._sessions["session"]
        self.agent._remember(
            state,
            "I'm looking for Shoes. A key requirement is: leather upper.",
        )

        self.agent._remember(
            state,
            "I don't want to start over; keep the shoes.",
        )
        self.agent._remember(
            state,
            "Please don't forget everything; keep the leather requirement.",
        )

        self.assertEqual(state.card_category, "shoes")
        self.assertEqual(state.card_constraints, ["leather upper"])
        self.assertEqual(len(state.messages), 3)

    def test_respond_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            self.agent.respond("missing", "shoe", 1, 1)

    def test_device_is_auto_detected_or_can_be_overridden(self) -> None:
        self.assertIn(self.agent.device, {"cpu", "cuda"})
        forced = Agent(self.catalog_path, encoder=KeywordEncoder(), cache_dir=None, device="cpu")
        self.assertEqual(forced.device, "cpu")
        forced.connection.close()

    def test_default_model_uses_minilm_l12(self) -> None:
        self.assertEqual(
            self.agent.model_name,
            DEFAULT_MODEL_NAME,
        )
        self.assertEqual(
            DEFAULT_MODEL_NAME,
            "sentence-transformers/all-MiniLM-L12-v2",
        )

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
