from __future__ import annotations

import math
import uuid
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)


class EvaluationSession:
    """Replay one official-simulator session for the local browser demo.

    This adapter uses the official evaluator's public helper functions but lives
    outside the evaluator so the submitted scoring implementation stays unchanged.
    Target-related fields remain private until the demo explicitly reveals them
    after the session has ended.
    """

    def __init__(
        self,
        agent: Any,
        sample: dict,
        catalog_ids: set[str],
        categories: dict[str, list[str]],
        products: dict[str, dict],
        *,
        session_id: str | None = None,
    ) -> None:
        self.agent = agent
        self.sample = sample
        self.catalog_ids = catalog_ids
        self.categories = categories
        self.products = products
        self.session_id = session_id or f"demo_{uuid.uuid4().hex}"

        self.agent.reset(self.session_id, sample["user_profile"])
        self.target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(
            sample, products
        )
        self.effective_sample = {
            **sample,
            "intent_card": effective_intent_card,
            "behavior": effective_behavior,
        }
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = sample["scenario_type"] != "intent_override"
        self.user_message = initial_message(
            self.effective_sample,
            coarse_category(categories.get(self.target, [])),
            self.disclosed,
        )
        self.user_message_source = "initial"
        self.turn = 1
        self.hit_turn: int | None = None
        self.best_rank: int | None = None
        self.done = False
        self.termination_reason: str | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.turns: list[dict] = []

    @staticmethod
    def _display_recommendations(response: dict, ranked: list[str]) -> list[dict]:
        """Retain optional Agent scores without changing normalization."""
        raw = response.get("recommendations")
        scores: dict[str, int | float] = {}
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                parent_asin = str(item.get("parent_asin", "")).strip()
                score = item.get("score")
                if (
                    parent_asin not in scores
                    and isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                ):
                    scores[parent_asin] = score
        return [
            {
                "parent_asin": parent_asin,
                **({"score": scores[parent_asin]} if parent_asin in scores else {}),
            }
            for parent_asin in ranked
        ]

    def step(self) -> dict:
        if self.done:
            raise RuntimeError("evaluation session has already ended")

        turn = self.turn
        user_message = self.user_message
        user_message_source = self.user_message_source
        boundary_used_before = self.boundary_used
        override_applied_before = self.override_applied
        disclosed_before = sorted(self.disclosed)
        degraded_reason: str | None = None

        try:
            response = self.agent.respond(
                self.session_id, user_message, turn, TOP_K
            )
        except Exception as exc:
            degraded_reason = (
                f"Agent raised {type(exc).__name__}; demo used an empty response."
            )
            response = {"message": "", "ask_attribute": None, "recommendations": []}
        if not isinstance(response, dict) or not isinstance(
            response.get("message"), str
        ):
            degraded_reason = (
                "Agent returned an invalid top-level response; "
                "demo used an empty response."
            )
            response = {"message": "", "ask_attribute": None, "recommendations": []}

        usage = response.get("usage")
        turn_prompt_tokens = 0
        turn_completion_tokens = 0
        if isinstance(usage, dict):
            if (
                isinstance(usage.get("prompt_tokens"), int)
                and usage["prompt_tokens"] >= 0
            ):
                turn_prompt_tokens = usage["prompt_tokens"]
                self.prompt_tokens += turn_prompt_tokens
            if (
                isinstance(usage.get("completion_tokens"), int)
                and usage["completion_tokens"] >= 0
            ):
                turn_completion_tokens = usage["completion_tokens"]
                self.completion_tokens += turn_completion_tokens

        ranked = normalize_recommendations(
            response.get("recommendations"), self.catalog_ids
        )
        target_rank = ranked.index(self.target) + 1 if self.target in ranked else None
        hit_rank = target_rank if self.override_applied else None
        transition: str
        next_user_message: str | None = None
        next_user_message_source: str | None = None

        if hit_rank is not None:
            self.best_rank = hit_rank
            self.hit_turn = turn
            self.done = True
            self.termination_reason = "hit"
            transition = "hit"
        elif turn == MAX_TURNS:
            self.done = True
            self.termination_reason = "max_turns"
            transition = "max_turns"
        else:
            override = self.effective_sample.get("behavior", {}).get("override") or {}
            if not self.override_applied and turn + 1 == int(override.get("turn", 3)):
                self.override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    self.disclosed.add(new_value)
                next_user_message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
                next_user_message_source = "intent_override"
                transition = "override"
            else:
                next_user_message, self.boundary_used = customer_reply(
                    self.effective_sample,
                    response.get("ask_attribute"),
                    self.disclosed,
                    self.boundary_used,
                )
                next_user_message_source = "simulator_reply"
                transition = "customer_reply"
            self.turn += 1
            self.user_message = next_user_message
            self.user_message_source = next_user_message_source

        event = {
            "turn": turn,
            "user_message": user_message,
            "user_message_source": user_message_source,
            "request_payload": {
                "session_id": self.session_id,
                "user_message": user_message,
                "turn": turn,
                "top_k": TOP_K,
            },
            "message": response["message"],
            "ask_attribute": (
                response.get("ask_attribute")
                if isinstance(response.get("ask_attribute"), str)
                else None
            ),
            "recommendations": self._display_recommendations(response, ranked),
            "ranked_ids": ranked,
            "target_was_eligible": override_applied_before,
            "target_rank": target_rank,
            "hit_rank": hit_rank,
            "transition": transition,
            "next_user_message": next_user_message,
            "next_user_message_source": next_user_message_source,
            "boundary_used_before": boundary_used_before,
            "boundary_used_after": self.boundary_used,
            "override_applied_before": override_applied_before,
            "override_applied_after": self.override_applied,
            "disclosed_before": disclosed_before,
            "disclosed_after": sorted(self.disclosed),
            "usage": {
                "prompt_tokens": turn_prompt_tokens,
                "completion_tokens": turn_completion_tokens,
                "total_tokens": turn_prompt_tokens + turn_completion_tokens,
            },
            "degraded": degraded_reason is not None,
            "degraded_reason": degraded_reason,
        }
        self.turns.append(event)
        return event

    def result(self) -> dict:
        return {
            "sample_id": self.sample["sample_id"],
            "scenario_type": self.sample["scenario_type"],
            "hit": self.hit_turn is not None,
            "first_hit_turn": self.hit_turn,
            "best_rank": self.best_rank,
            "reciprocal_rank": (
                0.0 if self.best_rank is None else 1.0 / self.best_rank
            ),
        }
