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

    def test_first_three_turns_are_conversational_and_accumulate_context(self) -> None:
        first = self.agent.respond("session", "I need a shoe.", 1, 2)
        second = self.agent.respond("session", "Leather would be good.", 2, 2)
        third = self.agent.respond("session", "I want blue for running.", 3, 2)

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(second["ask_attribute"], "material")
        self.assertEqual(third["ask_attribute"], "use_case")
        self.assertIn("what features", first["message"].lower())
        self.assertIn("material", second["message"].lower())
        self.assertIn("planning to use", third["message"].lower())
        self.assertEqual(third["recommendations"][0]["parent_asin"], "RUNNING")
        self.assertEqual(self.agent._sessions["session"].messages, [
            "I need a shoe.",
            "Leather would be good.",
            "I want blue for running.",
        ])

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

    def test_respond_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            self.agent.respond("missing", "shoe", 1, 1)

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
