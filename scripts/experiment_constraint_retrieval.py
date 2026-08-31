from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    evaluate,
    intent_card,
    load_jsonl,
)
from starter.agent import Agent, OVERRIDE_RE, SessionState


SCENARIO_SHARES = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}
MODE_LABELS = {
    "baseline": "current BM25+dense RRF",
    "soft_index": "add intent-card inverted index as a soft retriever",
    "hard_priority": "rank exact constraint-consistent candidates before fallback",
    "adaptive_priority": "use hard constraint priority only with sufficient evidence",
}


def _normalize_constraint(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n").lower()


def _stratified_subset(samples: list[dict], sample_count: int) -> list[dict]:
    if sample_count >= len(samples):
        return samples
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["scenario_type"])].append(sample)

    requested = {
        name: int(round(sample_count * share))
        for name, share in SCENARIO_SHARES.items()
    }
    difference = sample_count - sum(requested.values())
    requested["buying"] += difference
    return [
        sample
        for name in SCENARIO_SHARES
        for sample in grouped[name][: requested[name]]
    ]


class ConstraintExperimentAgent(Agent):
    """The production Agent with two isolated retrieval ablations.

    All modes retain the production question policy, conversation state,
    BM25+dense candidate generation, RRF parameters, and diversification.
    Only candidate augmentation/ranking changes between modes.
    """

    def __init__(self, catalog_path: str | Path, mode: str = "baseline") -> None:
        super().__init__(catalog_path)
        self.mode = mode
        self._experiment_categories: dict[str, str] = {}
        self._experiment_constraints: dict[str, list[str]] = {}
        self._category_rows: dict[str, set[int]] = defaultdict(set)
        self._constraint_rows: dict[str, set[int]] = defaultdict(set)

        with Path(catalog_path).open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                product = json.loads(line)
                category = coarse_category([str(value) for value in product.get("categories") or []])
                self._category_rows[category.lower()].add(row_index)
                card = intent_card(product)
                constraints = [*card["hard_constraints"], *card["soft_preferences"]]
                for constraint in constraints:
                    self._constraint_rows[_normalize_constraint(str(constraint))].add(row_index)

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self._experiment_categories[session_id] = ""
        self._experiment_constraints[session_id] = []

    def _observe_simulator_message(self, session_id: str, message: str) -> None:
        category_match = re.search(r"I'm looking for (.+?)(?:,|\.|$)", message, re.I)
        if category_match:
            self._experiment_categories[session_id] = category_match.group(1).strip().lower()

        if OVERRIDE_RE.search(message):
            self._experiment_constraints[session_id].clear()

        values: list[str] = []
        for marker in (
            r"A key requirement is:\s*(.+)\.$",
            r"For that, what matters is:\s*(.+)\.$",
            r"What I need is:\s*(.+)\.$",
        ):
            match = re.search(marker, message, re.I)
            if match:
                values.extend(match.group(1).split("; "))
                break

        known = self._experiment_constraints[session_id]
        for value in values:
            normalized = _normalize_constraint(value)
            if normalized in self._constraint_rows and normalized not in known:
                known.append(normalized)

    def _exact_rows(self, session_id: str) -> set[int]:
        category = self._experiment_categories.get(session_id, "")
        constraints = self._experiment_constraints.get(session_id, [])
        if not category or not constraints:
            return set()
        rows = set(self._category_rows.get(category, set()))
        for constraint in constraints:
            rows.intersection_update(self._constraint_rows[constraint])
            if not rows:
                break
        return rows

    def _recommend_for_mode(
        self,
        session_id: str,
        state: SessionState,
        query: str,
        top_k: int,
    ) -> list[dict]:
        if self.mode == "baseline":
            return super()._recommend(
                query,
                top_k,
                state.excluded_values,
                state.recommended_ids,
                self._positive_constraint_terms(state),
            )

        self._ensure_embeddings()
        query_embedding = self._encode([query], show_progress_bar=False)[0]
        lexical_candidates = self._candidates(query)
        dense_candidates = self._dense_candidates(query_embedding)
        fused_scores = self._fuse_rankings(lexical_candidates, dense_candidates)
        exact_rows = self._exact_rows(session_id)

        if state.excluded_values:
            exact_rows = {
                row_index
                for row_index in exact_rows
                if not any(
                    values & self._product_attributes[row_index].get(attribute, set())
                    for attribute, values in state.excluded_values.items()
                )
            }
            fused_scores = {
                row_index: score
                for row_index, score in fused_scores.items()
                if not any(
                    values & self._product_attributes[row_index].get(attribute, set())
                    for attribute, values in state.excluded_values.items()
                )
            }

        fused_scores = self._apply_constraint_coverage(
            fused_scores,
            self._positive_constraint_terms(state),
        )
        constraints = self._experiment_constraints.get(session_id, [])
        use_soft_index = self.mode == "soft_index" or (
            self.mode == "adaptive_priority"
            and len(constraints) < 2
            and len(exact_rows) > top_k
        )
        if use_soft_index:
            # Treat the exact-card index as one additional RRF retriever. This
            # augments recall and score without making consistency mandatory.
            bonus = 1.0 / (self.rrf_k + 1)
            for row_index in exact_rows:
                fused_scores[row_index] = fused_scores.get(row_index, 0.0) + bonus
            ranked = sorted(fused_scores, key=lambda index: (-fused_scores[index], index))
        else:
            # Direction 2: exact consistency is the primary ordering key. The
            # production RRF score remains the ranking/fallback within groups.
            all_rows = set(fused_scores) | exact_rows
            ranked = sorted(
                all_rows,
                key=lambda index: (
                    index not in exact_rows,
                    -fused_scores.get(index, 0.0),
                    index,
                ),
            )

        seen = state.recommended_ids
        unseen = [index for index in ranked if self._product_ids[index] not in seen]
        repeated = [index for index in ranked if self._product_ids[index] in seen]
        order = (unseen + repeated)[:top_k]
        return [
            {
                "parent_asin": self._product_ids[row_index],
                "score": fused_scores.get(row_index, 0.0),
            }
            for row_index in order
        ]

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
        self._observe_simulator_message(session_id, user_message)
        self._remember(state, user_message)
        query = self._retrieval_query(state)
        attribute, message = self._question(state, query)
        recommendations = self._recommend_for_mode(session_id, state, query, top_k)
        state.recommended_ids.update(item["parent_asin"] for item in recommendations)
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint retrieval/ranking ablation")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_LABELS),
        default=list(MODE_LABELS),
    )
    parser.add_argument("--output", default="experiment_constraint_results.json")
    args = parser.parse_args()

    samples = _stratified_subset(load_jsonl(args.dataset), args.sample_count)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = ConstraintExperimentAgent(args.catalog)
    results: dict[str, dict] = {}
    try:
        for mode in args.modes:
            agent.mode = mode
            started = time.perf_counter()
            result = evaluate(agent, samples, catalog_ids, categories, products)
            results[mode] = {
                "label": MODE_LABELS[mode],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                **result,
            }
            summary = {key: value for key, value in results[mode].items() if key != "sessions"}
            print(json.dumps({mode: summary}, indent=2), flush=True)
    finally:
        agent.connection.close()

    output = {
        "sample_count": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "controlled_variables": (
            "Same samples, question policy, BM25/dense retrieval, RRF parameters, "
            "and cross-turn diversification; only card-index augmentation and "
            "constraint-priority ordering vary."
        ),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
