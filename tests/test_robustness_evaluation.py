from __future__ import annotations

import unittest
from collections import Counter
import random

from scripts.evaluate_robustness import (
    MATCH_FIELDS,
    ProductProfile,
    build_product_profiles,
    choose_target,
    scenario_counts,
    score_summary,
    stratified_bootstrap_ci,
)


def _product(parent_asin: str, second_feature: str) -> dict:
    return {
        "parent_asin": parent_asin,
        "title": f"Example {parent_asin}",
        "features": ["waterproof", second_feature],
        "description": [],
        "price": None,
        "categories": ["Clothing", "Women", "Jackets"],
        "details": {"department": "womens"},
        "average_rating": 4.0,
        "rating_number": 10,
        "store": "Example",
    }


def _profile(parent_asin: str, *, full_collision: int, row_index: int = 0) -> ProductProfile:
    return ProductProfile(
        parent_asin=parent_asin,
        row_index=row_index,
        category="women jackets",
        broad_category="women",
        constraint_type="feature",
        rating_number=100.0,
        log_rating_number=4.0,
        average_rating=4.2,
        price=20.0,
        log_price=3.0,
        price_missing=0.0,
        metadata_complete=1.0,
        metadata_richness=8.0,
        row_percentile=0.1,
        first_collision=100,
        pair_collision=max(full_collision, 1),
        triple_collision=max(full_collision, 1),
        full_collision=full_collision,
        log_first_collision=4.0,
        log_pair_collision=1.0,
        log_triple_collision=1.0,
    )


class RobustnessEvaluationTest(unittest.TestCase):
    def test_scenario_counts_preserve_official_mix(self) -> None:
        self.assertEqual(
            scenario_counts(20),
            {"buying": 8, "browsing": 8, "intent_override": 3, "boundary": 1},
        )
        with self.assertRaisesRegex(ValueError, "multiple of 20"):
            scenario_counts(19)

    def test_collision_counts_use_constraint_intersections(self) -> None:
        profiles = build_product_profiles(
            [_product("A", "zip"), _product("B", "zip"), _product("C", "lace")]
        )
        by_id = {profile.parent_asin: profile for profile in profiles}
        self.assertEqual(by_id["A"].first_collision, 3)
        self.assertEqual(by_id["A"].pair_collision, 2)
        self.assertEqual(by_id["A"].full_collision, 2)
        self.assertEqual(by_id["C"].full_collision, 1)

    def test_ambiguous_mode_explicitly_selects_unseen_over_50_collision(self) -> None:
        anchor = _profile("PUBLIC", full_collision=1)
        ordinary = _profile("ORDINARY", full_collision=1, row_index=1)
        ambiguous = _profile("AMBIGUOUS", full_collision=60, row_index=2)
        selected, _, _, _ = choose_target(
            anchor,
            [ordinary, ambiguous],
            {"PUBLIC"},
            {field: 1.0 for field in MATCH_FIELDS},
            random.Random(7),
            "ambiguous",
            8,
            128,
        )
        self.assertEqual(selected.parent_asin, "AMBIGUOUS")

    def test_score_and_stratified_bootstrap_are_reproducible(self) -> None:
        scenarios = ["buying"] * 8 + ["browsing"] * 8 + ["intent_override"] * 3 + ["boundary"]
        sessions = [
            {
                "sample_id": str(index),
                "scenario_type": scenario,
                "hit": index != 19,
                "first_hit_turn": None if index == 19 else 2,
                "reciprocal_rank": 0.0 if index == 19 else 1.0,
            }
            for index, scenario in enumerate(scenarios)
        ]
        self.assertEqual(score_summary(sessions)["sample_count"], 20)
        first = stratified_bootstrap_ci(sessions, 100, 11)
        second = stratified_bootstrap_ci(sessions, 100, 11)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(scenarios),
            Counter({"buying": 8, "browsing": 8, "intent_override": 3, "boundary": 1}),
        )


if __name__ == "__main__":
    unittest.main()
