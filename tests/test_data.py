from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.data import CATALOG_FIELDS, DataValidationError, validate_catalog


def _product(parent_asin: str) -> dict:
    return {
        **{field: None for field in CATALOG_FIELDS},
        "parent_asin": parent_asin,
        "title": "Test product",
        "features": [],
        "description": [],
        "categories": ["Clothing"],
        "details": {},
        "average_rating": 4.0,
        "rating_number": 1,
    }


class DataValidationTest(unittest.TestCase):
    def test_catalog_validator_accepts_expected_unique_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                "".join(json.dumps(_product(identifier)) + "\n" for identifier in ("A", "B")),
                encoding="utf-8",
            )
            with patch("scripts.data.CATALOG_ROWS", 2):
                self.assertEqual(validate_catalog(path), 2)

    def test_catalog_validator_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text((json.dumps(_product("A")) + "\n") * 2, encoding="utf-8")
            with patch("scripts.data.CATALOG_ROWS", 2):
                with self.assertRaisesRegex(DataValidationError, "duplicate parent_asin"):
                    validate_catalog(path)


if __name__ == "__main__":
    unittest.main()
