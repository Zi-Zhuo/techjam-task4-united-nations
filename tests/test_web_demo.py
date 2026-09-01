from __future__ import annotations

import json
import unittest

from scripts.demo_evaluation_session import EvaluationSession
from scripts.web_demo import DemoAPIError, DemoApplication


def sample_for(scenario: str, *, override_turn: int = 3) -> dict:
    behavior: dict = {"scenario_type": scenario}
    if scenario == "intent_override":
        behavior["override"] = {
            "turn": override_turn,
            "old_value": "I used to prefer red.",
            "new_value": "cotton",
            "message": "Actually, ignore red. I need cotton.",
        }
    return {
        "sample_id": f"sample_{scenario}",
        "scenario_type": scenario,
        "category_bucket": "clothing",
        "difficulty_bucket": "hard",
        "user_profile": {
            "summary": "Prefers durable, comfortable products.",
            "preference_tags": ["durability", "comfort"],
            "purchase_frequency": "3-4 prior purchases",
            "rating_style": "usually positive",
        },
        "ground_truth": {"parent_asin": "A"},
        "intent_card": {
            "target_category": "Cotton shirt",
            "hard_constraints": ["cotton", "color: blue"],
            "soft_preferences": ["durable"],
        },
        "behavior": behavior,
    }


PRODUCTS = {
    "A": {
        "parent_asin": "A",
        "title": "Target cotton shirt",
        "features": ["cotton"],
        "categories": ["Clothing", "Shirts"],
        "store": "Target Store",
        "price": 25.0,
        "average_rating": 4.5,
        "rating_number": 20,
    },
    "B": {
        "parent_asin": "B",
        "title": "Other shirt",
        "features": ["polyester"],
        "categories": ["Clothing", "Shirts"],
        "store": "Other Store",
        "price": 20.0,
        "average_rating": 4.0,
        "rating_number": 10,
    },
}
CATEGORIES = {"A": ["Clothing", "Shirts"], "B": ["Clothing", "Shirts"]}


class ScriptedAgent:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, int, int]] = []
        self.resets: list[tuple[str, dict]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.resets.append((session_id, user_profile))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.calls.append((session_id, user_message, turn, top_k))
        response = self.responses[turn - 1]
        if isinstance(response, BaseException):
            raise response
        return response  # type: ignore[return-value]


def response(
    recommendations: list[object] | None = None,
    *,
    ask_attribute: object = None,
    message: object = "agent reply",
    usage: object = None,
) -> dict:
    result = {
        "message": message,
        "ask_attribute": ask_attribute,
        "recommendations": recommendations or [],
    }
    if usage is not None:
        result["usage"] = usage
    return result


class EvaluationSessionTest(unittest.TestCase):
    def make_session(self, agent: ScriptedAgent, scenario: str, *, override_turn: int = 3) -> EvaluationSession:
        return EvaluationSession(
            agent,
            sample_for(scenario, override_turn=override_turn),
            {"A", "B"},
            CATEGORIES,
            PRODUCTS,
            session_id="test_session",
        )

    def test_intent_override_requires_target_to_reappear_after_override(self) -> None:
        agent = ScriptedAgent([
            response([{"parent_asin": "A", "score": 0.9}], ask_attribute="feature"),
            response(["B"], ask_attribute="material"),
            response(["A"]),
        ])
        session = self.make_session(agent, "intent_override", override_turn=3)

        self.assertIn("I used to prefer red.", session.user_message)
        self.assertEqual(session.disclosed, set())
        first = session.step()
        self.assertFalse(session.done)
        self.assertEqual(first["target_rank"], 1)
        self.assertIsNone(first["hit_rank"])

        second = session.step()
        self.assertEqual(second["transition"], "override")
        self.assertEqual(session.user_message, "Actually, ignore red. I need cotton.")
        self.assertIn("cotton", session.disclosed)

        third = session.step()
        self.assertEqual(third["transition"], "hit")
        self.assertEqual(session.result()["first_hit_turn"], 3)
        self.assertEqual(session.result()["best_rank"], 1)

    def test_boundary_is_consumed_only_by_first_string_attribute(self) -> None:
        agent = ScriptedAgent([
            response(["B"], ask_attribute=None),
            response(["B"], ask_attribute="mystery"),
            response(["A"], ask_attribute="material"),
        ])
        session = self.make_session(agent, "boundary")

        first = session.step()
        self.assertFalse(first["boundary_used_after"])
        self.assertIn("Ask me about one specific attribute", session.user_message)

        second = session.step()
        self.assertFalse(second["boundary_used_before"])
        self.assertTrue(second["boundary_used_after"])
        self.assertEqual(
            session.user_message,
            "I don't have a preference for mystery; please use your judgment.",
        )

        third = session.step()
        self.assertEqual(third["transition"], "hit")

    def test_invalid_or_exception_response_uses_empty_fallback_and_turn_ten_can_hit(self) -> None:
        responses: list[object] = [
            RuntimeError("boom"),
            response(["A"], message=None, usage={"prompt_tokens": 99}),
        ]
        responses.extend(response(["B"]) for _ in range(7))
        responses.append(response(["A"], usage={"prompt_tokens": 2, "completion_tokens": 3}))
        session = self.make_session(ScriptedAgent(responses), "browsing")

        first = session.step()
        second = session.step()
        self.assertTrue(first["degraded"])
        self.assertTrue(second["degraded"])
        self.assertEqual(second["ranked_ids"], [])
        while not session.done:
            session.step()

        self.assertEqual(session.hit_turn, 10)
        self.assertEqual(session.prompt_tokens, 2)
        self.assertEqual(session.completion_tokens, 3)


class DemoApplicationTest(unittest.TestCase):
    def test_target_labels_and_hidden_state_are_sealed_until_terminal_reveal(self) -> None:
        agent = ScriptedAgent([response([{"parent_asin": "A", "score": 1.0}])])
        application = DemoApplication(
            agent,
            [sample_for("buying")],
            {"A", "B"},
            CATEGORIES,
            PRODUCTS,
        )
        created = application.create_session("sample_buying")
        token = created["session_token"]

        self.assertNotIn("target", created)
        self.assertNotIn("hidden_simulator_state", created)
        with self.assertRaises(DemoAPIError):
            application.reveal_session(token)

        completed = application.step_session(token)
        self.assertTrue(completed["status"]["hit"])
        self.assertNotIn("target", completed)
        self.assertNotIn("is_target", json.dumps(completed))

        revealed = application.reveal_session(token)
        self.assertEqual(revealed["target"]["parent_asin"], "A")
        self.assertTrue(revealed["turns"][0]["recommendations"][0]["is_target"])
        self.assertIn("hidden_simulator_state", revealed)

    def test_sample_listing_does_not_expose_ground_truth(self) -> None:
        application = DemoApplication(
            ScriptedAgent([]),
            [sample_for("buying")],
            {"A", "B"},
            CATEGORIES,
            PRODUCTS,
        )
        serialized = json.dumps(application.sample_options())
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn('"A"', serialized)


if __name__ == "__main__":
    unittest.main()
