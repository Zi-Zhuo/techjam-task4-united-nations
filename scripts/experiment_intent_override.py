from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import (
    FULL_RESET_RE,
    INITIAL_CONTEXT_RE,
    OVERRIDE_RE,
    QUESTION_TEXT,
    SIMULATOR_CONSTRAINT_PATTERNS,
    Agent,
    SessionState,
    _card_constraints,
    _normalize_card_value,
)


RESET_POLICIES = ("current_reset", "preserve_observed")
QUESTION_POLICIES = ("current_question", "turn1_other")
METRIC_NAMES = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)


class _ForbiddenEncoder:
    """Fail loudly if this metadata-only experiment accidentally loads BERT."""

    def encode(self, texts: list[str], **kwargs: object) -> object:
        raise AssertionError("the intent-override metadata experiment must not encode text")


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


class IntentOverrideMetadataAgent(Agent):
    """Production state/question logic with a local card-only recommendation path.

    The evaluator calls ``reset`` with only a profile and ``respond`` with only a
    conversational message. This class never receives or reads a sample id or
    ground-truth id. It reads the same participant-visible catalog as production.
    """

    def __init__(self, catalog_path: str | Path) -> None:
        started = time.perf_counter()
        super().__init__(
            catalog_path,
            encoder=_ForbiddenEncoder(),
            cache_dir=None,
            device="cpu",
        )
        self.index_build_seconds = time.perf_counter() - started
        self.reset_policy = RESET_POLICIES[0]
        self.question_policy = QUESTION_POLICIES[0]
        self._card_sequences: list[list[str]] = []
        self._popularity: list[float] = []
        self._ratings: list[float] = []
        self._run_session_ids: list[str] = []
        self._traces: dict[str, list[dict]] = {}
        self._initial_old_values: dict[str, str] = {}

        with Path(catalog_path).open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                product = json.loads(line)
                if str(product["parent_asin"]) != self._product_ids[row_index]:
                    raise RuntimeError("catalog order changed while building card metadata")
                self._card_sequences.append(
                    [_normalize_card_value(value) for value in _card_constraints(product)]
                )
                rating_number = product.get("rating_number")
                average_rating = product.get("average_rating")
                self._popularity.append(
                    float(rating_number) if isinstance(rating_number, (int, float)) else 0.0
                )
                self._ratings.append(
                    float(average_rating) if isinstance(average_rating, (int, float)) else 0.0
                )

    def begin_mode(self, reset_policy: str, question_policy: str) -> None:
        if reset_policy not in RESET_POLICIES:
            raise ValueError(f"unknown reset policy: {reset_policy}")
        if question_policy not in QUESTION_POLICIES:
            raise ValueError(f"unknown question policy: {question_policy}")
        self.reset_policy = reset_policy
        self.question_policy = question_policy
        self._run_session_ids = []
        self._traces = {}
        self._initial_old_values = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self._run_session_ids.append(session_id)
        self._traces[session_id] = []
        self._initial_old_values[session_id] = ""

    def _capture_initial_old_value(
        self, session_id: str, state: SessionState, user_message: str
    ) -> None:
        """Record the exact post-category preference without sample metadata."""
        if state.messages or self._initial_old_values.get(session_id):
            return
        initial_match = INITIAL_CONTEXT_RE.match(user_message.strip())
        if not initial_match:
            return
        tail = user_message.strip()[initial_match.end() :].strip()
        normalized = _normalize_card_value(tail)
        if normalized in self._constraint_rows:
            self._initial_old_values[session_id] = normalized

    def _without_old_structured_value(
        self, message: str, old_value: str
    ) -> str | None:
        """Remove one exact card value while retaining its sibling reply value."""
        if not old_value:
            return message
        for pattern in SIMULATOR_CONSTRAINT_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            values = self._split_known_card_constraints(match.group(1))
            if old_value not in values:
                return message
            retained = [value for value in values if value != old_value]
            if not retained:
                return None
            return (
                message[: match.start(1)]
                + "; ".join(retained)
                + message[match.end(1) :]
            )
        return message

    def _remember_preserving_observed(
        self, session_id: str, state: SessionState, user_message: str
    ) -> None:
        is_override = bool(OVERRIDE_RE.search(user_message))
        is_full_reset = is_override and bool(FULL_RESET_RE.search(user_message))
        if not is_full_reset:
            super()._remember(state, user_message)
            return

        # Match production behavior that recommendations made before an intent
        # change are eligible again, but retain later structured answers. The
        # evaluator changes only the initial soft preference; its already
        # disclosed hard/card constraints remain valid and are not sent again.
        state.recommended_ids.clear()
        retained_messages: list[str] = []
        if state.messages:
            initial_match = INITIAL_CONTEXT_RE.match(state.messages[0].strip())
            if initial_match:
                retained_messages.append(initial_match.group(0).strip())
            old_value = self._initial_old_values.get(session_id, "")
            for message in state.messages[1:]:
                retained = self._without_old_structured_value(message, old_value)
                if retained:
                    retained_messages.append(retained)
        state.messages = retained_messages
        state.superseded_values.clear()
        state.messages.append(user_message)
        self._refresh_preferences(state)
        self._refresh_card_index_state(state)

    def _ask(self, state: SessionState, query: str, turn: int) -> tuple[str, str]:
        if self.question_policy == "turn1_other" and turn == 1:
            state.asked_attributes.add("other")
            return "other", QUESTION_TEXT["other"]
        return self._question(state, query)

    def _metadata_recommend(
        self, state: SessionState, query: str, top_k: int
    ) -> tuple[list[dict], int]:
        exact_rows = self._card_candidate_rows(state)
        lexical = self._candidates(query)
        lexical_rank = {
            row_index: rank for rank, (row_index, _) in enumerate(lexical, start=1)
        }
        category_rows = self._category_rows.get(state.card_category, set())
        candidate_rows = set(exact_rows) | set(lexical_rank)
        if len(candidate_rows) < top_k:
            candidate_rows.update(category_rows)

        observed = state.card_constraints

        def rank_key(row_index: int) -> tuple:
            sequence = self._card_sequences[row_index]
            exact_slots = sum(
                index < len(sequence) and sequence[index] == constraint
                for index, constraint in enumerate(observed)
            )
            lcs = _lcs_length(observed, sequence)
            return (
                bool(exact_rows) and row_index not in exact_rows,
                -exact_slots,
                -lcs,
                lexical_rank.get(row_index, self.candidate_count + 1),
                -math.log1p(self._popularity[row_index]),
                -self._ratings[row_index],
                row_index,
            )

        ranked = sorted(candidate_rows, key=rank_key)
        unseen = [
            row_index
            for row_index in ranked
            if self._product_ids[row_index] not in state.recommended_ids
        ]
        repeated = [
            row_index
            for row_index in ranked
            if self._product_ids[row_index] in state.recommended_ids
        ]
        chosen = (unseen + repeated)[:top_k]
        return (
            [{"parent_asin": self._product_ids[row_index]} for row_index in chosen],
            len(exact_rows),
        )

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

        before_messages = list(state.messages)
        is_override = bool(OVERRIDE_RE.search(user_message))
        self._capture_initial_old_value(session_id, state, user_message)
        if self.reset_policy == "preserve_observed":
            self._remember_preserving_observed(session_id, state, user_message)
        else:
            self._remember(state, user_message)

        query = self._retrieval_query(state)
        attribute, message = self._ask(state, query, turn)
        recommendations, exact_candidate_count = self._metadata_recommend(
            state, query, top_k
        )
        state.recommended_ids.update(
            item["parent_asin"] for item in recommendations
        )
        self._traces[session_id].append(
            {
                "turn": turn,
                "user_message": user_message,
                "is_override": is_override,
                "initial_old_value": self._initial_old_values.get(session_id, ""),
                "messages_before": before_messages,
                "messages_after": list(state.messages),
                "observed_card_constraints": list(state.card_constraints),
                "exact_card_candidate_count": exact_candidate_count,
                "ask_attribute": attribute,
                "recommendations": [
                    item["parent_asin"] for item in recommendations
                ],
            }
        )
        return {
            "message": message,
            "ask_attribute": attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def ordered_traces(self) -> list[list[dict]]:
        return [self._traces[session_id] for session_id in self._run_session_ids]


def _mode_name(reset_policy: str, question_policy: str) -> str:
    return f"{reset_policy}__{question_policy}"


def _delta(left: dict, right: dict) -> dict[str, float]:
    """Return right-minus-left metric deltas (preserve minus current)."""
    return {
        metric: round(float(right[metric]) - float(left[metric]), 6)
        for metric in METRIC_NAMES
    }


def _paired_session_deltas(current: dict, preserved: dict) -> list[dict]:
    current_sessions = {
        item["sample_id"]: item for item in current["sessions"]
    }
    preserved_sessions = {
        item["sample_id"]: item for item in preserved["sessions"]
    }
    return [
        {
            "sample_id": sample_id,
            "current_first_hit_turn": current_sessions[sample_id]["first_hit_turn"],
            "preserved_first_hit_turn": preserved_sessions[sample_id]["first_hit_turn"],
            "turn_delta": (
                (preserved_sessions[sample_id]["first_hit_turn"] or 11)
                - (current_sessions[sample_id]["first_hit_turn"] or 11)
            ),
            "current_best_rank": current_sessions[sample_id]["best_rank"],
            "preserved_best_rank": preserved_sessions[sample_id]["best_rank"],
            "reciprocal_rank_delta": round(
                preserved_sessions[sample_id]["reciprocal_rank"]
                - current_sessions[sample_id]["reciprocal_rank"],
                6,
            ),
        }
        for sample_id in current_sessions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intent-override reset-policy/card-evidence 2x2 ablation"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--output", default="experiment_intent_override_results.json"
    )
    args = parser.parse_args()

    samples = [
        sample
        for sample in load_jsonl(args.dataset)
        if sample.get("scenario_type") == "intent_override"
    ]
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = IntentOverrideMetadataAgent(args.catalog)
    results: dict[str, dict] = {}
    try:
        for question_policy in QUESTION_POLICIES:
            for reset_policy in RESET_POLICIES:
                name = _mode_name(reset_policy, question_policy)
                agent.begin_mode(reset_policy, question_policy)
                started = time.perf_counter()
                evaluation = evaluate(
                    agent, samples, catalog_ids, categories, products
                )
                elapsed_seconds = time.perf_counter() - started
                traces = agent.ordered_traces()
                if len(traces) != len(evaluation["sessions"]):
                    raise RuntimeError("trace/session count mismatch")
                sessions = [
                    {**session, "trajectory": trajectory}
                    for session, trajectory in zip(
                        evaluation["sessions"], traces, strict=True
                    )
                ]
                results[name] = {
                    "config": {
                        "reset_policy": reset_policy,
                        "question_policy": question_policy,
                        "ranking": (
                            "exact-card hard priority, slot/LCS evidence, BM25, "
                            "then log popularity/rating"
                        ),
                        "encoder_loaded": False,
                    },
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    **{
                        key: value
                        for key, value in evaluation.items()
                        if key != "sessions"
                    },
                    "sessions": sessions,
                }
                summary = {
                    key: value
                    for key, value in results[name].items()
                    if key not in {"sessions", "reported_token_usage"}
                }
                print(json.dumps({name: summary}, indent=2), flush=True)
    finally:
        agent.connection.close()

    comparisons: dict[str, dict] = {}
    for question_policy in QUESTION_POLICIES:
        current = results[_mode_name("current_reset", question_policy)]
        preserved = results[_mode_name("preserve_observed", question_policy)]
        comparisons[question_policy] = {
            "preserve_minus_current": _delta(current, preserved),
            "paired_sessions": _paired_session_deltas(current, preserved),
        }

    current_effect = comparisons["current_question"]["preserve_minus_current"]
    other_effect = comparisons["turn1_other"]["preserve_minus_current"]
    interaction = {
        metric: round(other_effect[metric] - current_effect[metric], 6)
        for metric in METRIC_NAMES
    }

    output = {
        "experiment": "intent override observed-card evidence 2x2 ablation",
        "sample_count": len(samples),
        "scenario_filter": "intent_override",
        "sample_ids": [sample["sample_id"] for sample in samples],
        "agent_input_guard": (
            "Agent receives only user_profile and user_message through the official "
            "interface; sample_id and ground_truth remain evaluator-only."
        ),
        "model": "none (metadata/BM25 runner; encoder calls are forbidden)",
        "index_build_seconds": round(agent.index_build_seconds, 3),
        "controlled_variables": (
            "Same 30 sessions, catalog, exact-card/BM25 ranking, popularity tie-break, "
            "and diversification. Only full-reset history retention and the first "
            "question vary."
        ),
        "results": results,
        "paired_comparisons": comparisons,
        "preserve_x_turn1_other_interaction": interaction,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
