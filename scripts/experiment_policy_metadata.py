from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import (
    catalog_index,
    classify_constraint,
    coarse_category,
    evaluate,
    intent_card,
    load_jsonl,
)


QUESTION_POLICIES = ("current_like", "other_first", "repeated_other")
K_POLICIES = ("fixed_1", "fixed_2", "fixed_3", "fixed_10", "deadline_aware")
RANKING_POLICIES = ("catalog_row", "rating_number")
ATTRIBUTE_PRIORITY = (
    "feature",
    "material",
    "color",
    "style",
    "size",
    "use_case",
    "budget",
    "brand",
    "other",
)
HIGH_VALUE_ATTRIBUTES = frozenset({"feature", "material"})
QUESTION_TEXT = {
    "feature": "What features matter most to you?",
    "material": "Do you have a material preference?",
    "color": "Is there a color you would prefer?",
    "style": "What kind of style or fit would suit you best?",
    "size": "Are there any size or width requirements?",
    "use_case": "How are you planning to use it?",
    "budget": "Do you have a budget range in mind?",
    "brand": "Do you have a preferred brand?",
    "other": "What else would make one option stand out?",
}
INITIAL_CATEGORY_RE = re.compile(r"\bI'm looking for (.+?)(?:,|\.|$)", re.I)
OVERRIDE_RE = re.compile(
    r"\b(actually|instead|changed my mind|ignore (?:my )?(?:earlier|previous)|no longer)\b",
    re.I,
)
CONSTRAINT_PATTERNS = (
    re.compile(r"A key requirement is:\s*(.+)\.$", re.I),
    re.compile(r"For that, what matters is:\s*(.+)\.$", re.I),
    re.compile(r"What I need is:\s*(.+)\.$", re.I),
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n").lower()


def _numeric(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class PolicyConfig:
    question_policy: str
    k_policy: str
    ranking_policy: str

    @property
    def mode_id(self) -> str:
        return "__".join((self.question_policy, self.k_policy, self.ranking_policy))


@dataclass
class MetadataSession:
    category: str = ""
    constraints: list[str] = field(default_factory=list)
    asked_attributes: set[str] = field(default_factory=set)
    recommended_rows: set[int] = field(default_factory=set)


class MetadataPolicyAgent:
    """Catalog-only policy agent; it receives only the documented Agent inputs."""

    def __init__(self, catalog_path: str | Path, config: PolicyConfig) -> None:
        self.catalog_path = Path(catalog_path)
        self.config = config
        self._sessions: dict[str, MetadataSession] = {}
        self._product_ids: list[str] = []
        self._rating_numbers: list[float] = []
        self._card_sequences: list[tuple[str, ...]] = []
        self._card_values_by_attribute: list[dict[str, frozenset[str]]] = []
        self._category_rows: dict[str, set[int]] = {}
        self._constraint_rows: dict[str, set[int]] = {}
        self._all_rows: set[int] = set()
        self._build_index()

    def _build_index(self) -> None:
        with self.catalog_path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                product = json.loads(line)
                self._product_ids.append(str(product["parent_asin"]))
                self._rating_numbers.append(_numeric(product.get("rating_number")))
                category = _normalize(
                    coarse_category([str(value) for value in product.get("categories") or []])
                )
                self._category_rows.setdefault(category, set()).add(row_index)

                card = intent_card(product)
                sequence = tuple(
                    _normalize(str(value))
                    for value in (*card["hard_constraints"], *card["soft_preferences"])
                )
                self._card_sequences.append(sequence)
                values_by_attribute: dict[str, set[str]] = {}
                for constraint in sequence:
                    self._constraint_rows.setdefault(constraint, set()).add(row_index)
                    attribute = classify_constraint(constraint)
                    values_by_attribute.setdefault(attribute, set()).add(constraint)
                self._card_values_by_attribute.append(
                    {
                        attribute: frozenset(values)
                        for attribute, values in values_by_attribute.items()
                    }
                )
        self._all_rows = set(range(len(self._product_ids)))

    def set_config(self, config: PolicyConfig) -> None:
        self.config = config
        self._sessions.clear()

    def reset(self, session_id: str, user_profile: dict) -> None:
        del user_profile
        self._sessions[session_id] = MetadataSession()

    def _split_known_constraints(self, payload: str) -> list[str]:
        whole = _normalize(payload)
        if whole in self._constraint_rows:
            return [whole]
        split_points = [match.start() for match in re.finditer(r";\s+", payload)]
        for split_point in split_points:
            left = _normalize(payload[:split_point])
            right = _normalize(payload[split_point + 1 :])
            if left in self._constraint_rows and right in self._constraint_rows:
                return [left, right]
        return [
            normalized
            for part in re.split(r";\s+", payload)
            if (normalized := _normalize(part)) in self._constraint_rows
        ]

    def _observe(self, state: MetadataSession, message: str) -> None:
        category_match = INITIAL_CATEGORY_RE.search(message)
        if category_match:
            state.category = _normalize(category_match.group(1))

        if OVERRIDE_RE.search(message):
            state.constraints.clear()
            state.recommended_rows.clear()

        for pattern in CONSTRAINT_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            for constraint in self._split_known_constraints(match.group(1)):
                if constraint not in state.constraints:
                    state.constraints.append(constraint)
            break

    def _evidence_rows(self, state: MetadataSession) -> tuple[set[int], bool]:
        category_rows = set(self._category_rows.get(state.category, self._all_rows))
        if not state.constraints:
            return category_rows, False
        exact_rows = set(category_rows)
        for constraint in state.constraints:
            exact_rows.intersection_update(self._constraint_rows.get(constraint, set()))
            if not exact_rows:
                break
        return (exact_rows, True) if exact_rows else (category_rows, False)

    def _current_like_question(self, state: MetadataSession, rows: set[int]) -> str:
        available = [
            attribute
            for attribute in ATTRIBUTE_PRIORITY
            if attribute not in state.asked_attributes
        ]
        if not available:
            return "other"

        def utility(attribute: str) -> tuple[float, int]:
            values = [
                self._card_values_by_attribute[row_index].get(attribute, frozenset())
                for row_index in rows
            ]
            covered = sum(bool(value) for value in values)
            distinct_variants = len(set(values))
            information = (
                (covered / len(values)) * max(0, distinct_variants - 1)
                if values
                else 0.0
            )
            return information, -ATTRIBUTE_PRIORITY.index(attribute)

        high_value = [
            attribute for attribute in available if attribute in HIGH_VALUE_ATTRIBUTES
        ]
        return max(high_value or available, key=utility)

    def _question(
        self,
        state: MetadataSession,
        rows: set[int],
        turn: int,
    ) -> tuple[str, str]:
        if self.config.question_policy == "repeated_other":
            attribute = "other"
        elif self.config.question_policy == "other_first" and turn == 1:
            attribute = "other"
        else:
            attribute = self._current_like_question(state, rows)
        state.asked_attributes.add(attribute)
        return attribute, QUESTION_TEXT[attribute]

    def _rank(self, rows: set[int]) -> list[int]:
        if self.config.ranking_policy == "catalog_row":
            return sorted(rows)
        return sorted(rows, key=lambda index: (-self._rating_numbers[index], index))

    def _result_count(
        self,
        state: MetadataSession,
        ranked_rows: list[int],
        turn: int,
    ) -> int:
        if self.config.k_policy.startswith("fixed_"):
            return int(self.config.k_policy.removeprefix("fixed_"))
        unseen_count = sum(row_index not in state.recommended_rows for row_index in ranked_rows)
        candidate_count = unseen_count or len(ranked_rows)
        remaining_turns = max(1, 11 - turn)
        return max(1, min(10, math.ceil(candidate_count / remaining_turns)))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._observe(state, user_message)
        rows, _ = self._evidence_rows(state)
        ranked_rows = self._rank(rows)
        attribute, message = self._question(state, rows, turn)
        count = min(top_k, self._result_count(state, ranked_rows, turn))
        unseen = [row_index for row_index in ranked_rows if row_index not in state.recommended_rows]
        repeated = [row_index for row_index in ranked_rows if row_index in state.recommended_rows]
        selected = (unseen + repeated)[:count]
        state.recommended_rows.update(selected)
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": self._product_ids[row_index]}
                for row_index in selected
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def _source_isolation_audit() -> dict:
    source = inspect.getsource(MetadataPolicyAgent).lower()
    forbidden = ("ground_truth", "sample_id", "public_set")
    references = {name: name in source for name in forbidden}
    return {
        "passed": not any(references.values()),
        "forbidden_references_in_agent_class": references,
        "constructor_parameters": list(inspect.signature(MetadataPolicyAgent).parameters),
        "reset_parameters": list(inspect.signature(MetadataPolicyAgent.reset).parameters),
        "respond_parameters": list(inspect.signature(MetadataPolicyAgent.respond).parameters),
    }


def _loaded_model_modules() -> list[str]:
    prefixes = ("sentence_transformers", "torch", "transformers")
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def _configs() -> list[PolicyConfig]:
    return [
        PolicyConfig(question, k_policy, ranking)
        for question in QUESTION_POLICIES
        for k_policy in K_POLICIES
        for ranking in RANKING_POLICIES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata-only policy ablation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="experiment_policy_metadata_results.json")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    configs = _configs()
    isolation_audit = _source_isolation_audit()
    if not isolation_audit["passed"]:
        raise RuntimeError("Agent source isolation audit failed")

    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    index_started = time.perf_counter()
    agent = MetadataPolicyAgent(catalog_path, configs[0])
    index_elapsed = time.perf_counter() - index_started

    results: dict[str, dict] = {}
    experiment_started = time.perf_counter()
    for config in configs:
        agent.set_config(config)
        started = time.perf_counter()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        results[config.mode_id] = {
            "config": {
                "question_policy": config.question_policy,
                "k_policy": config.k_policy,
                "ranking_policy": config.ranking_policy,
            },
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            **result,
        }
        summary = {
            key: value
            for key, value in results[config.mode_id].items()
            if key not in {"sessions", "scenario_metrics", "reported_token_usage"}
        }
        print(json.dumps({config.mode_id: summary}), flush=True)

    loaded_model_modules = _loaded_model_modules()
    output = {
        "experiment": "metadata_policy_ablation",
        "sample_count": len(samples),
        "mode_count": len(configs),
        "index_elapsed_seconds": round(index_elapsed, 6),
        "evaluation_elapsed_seconds": round(
            time.perf_counter() - experiment_started,
            6,
        ),
        "policy_definitions": {
            "current_like": (
                "Choose between feature/material first using catalog-card coverage/diversity, "
                "then use the remaining attribute priority."
            ),
            "other_first": "Ask other on turn 1, then use current_like.",
            "repeated_other": "Ask other on every turn.",
            "deadline_aware": (
                "Return ceil(unseen candidates / remaining turns), clipped to [1, 10]."
            ),
            "catalog_row": "Order candidate rows by frozen catalog position.",
            "rating_number": "Order by descending rating_number, then catalog position.",
        },
        "isolation_audit": {
            **isolation_audit,
            "agent_received_catalog_only": True,
            "external_api_calls": 0,
            "loaded_model_modules": loaded_model_modules,
            "no_embedding_model_loaded": not loaded_model_modules,
        },
        "artifacts": {
            "catalog": str(catalog_path),
            "catalog_sha256": _sha256(catalog_path),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "script": str(Path(__file__)),
            "script_sha256": _sha256(Path(__file__)),
        },
        "git": _git_metadata(),
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "mode_count": len(configs),
                "index_elapsed_seconds": output["index_elapsed_seconds"],
                "evaluation_elapsed_seconds": output["evaluation_elapsed_seconds"],
                "isolation_audit": output["isolation_audit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
