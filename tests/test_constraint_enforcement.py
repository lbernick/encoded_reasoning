import asyncio
import unittest

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from src.encoded_reasoning.evals.datasets import wrap_with_constraint, arithmetic_scorer
from src.encoded_reasoning.evals.constraints import get_constraint


def make_state(reasoning: str, answer: str = "2") -> TaskState:
    """Build a minimal TaskState for scorer tests."""
    return TaskState(
        model=ModelName("openai/gpt-4o-mini"),
        sample_id=1,
        epoch=0,
        input="1+1",
        messages=[ChatMessageSystem(content=""), ChatMessageUser(content="1+1")],
        target=Target(answer),
        output=ModelOutput(completion=f"<reasoning>{reasoning}</reasoning><answer>{answer}</answer>"),
    )


class ConstraintEnforcementTests(unittest.TestCase):
    def test_enforcement_passes_when_compliant(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "max_len_3", enforce=True)()
        state = make_state("two add")  # all words <=3 letters, compliance=1.0
        result = asyncio.run(scorer(state, Target("2")))
        self.assertTrue(result.value)
        self.assertIn("reasoning_compliance=1.000", result.explanation)

    def test_enforcement_fails_on_violation(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "max_len_3", enforce=True)()
        state = make_state("longwordexceeds")  # violates max_len_3, compliance=0.0
        result = asyncio.run(scorer(state, Target("2")))
        self.assertFalse(result.value)
        self.assertIn("reasoning_compliance=0.000", result.explanation)

    def test_enforcement_disabled_returns_base_score(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "max_len_3", enforce=False)()
        state = make_state("longwordexceeds")  # would fail if enforced
        result = asyncio.run(scorer(state, Target("2")))
        self.assertTrue(result.value)  # should ignore constraint and accept correct answer
        self.assertNotIn("reasoning_compliance", result.explanation)

    def test_missing_reasoning_block_fails_when_enforced(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "max_len_3", enforce=True)()
        state = TaskState(
            model=ModelName("openai/gpt-4o-mini"),
            sample_id=1,
            epoch=0,
            input="1+1",
            messages=[ChatMessageSystem(content=""), ChatMessageUser(content="1+1")],
            target=Target("2"),
            output=ModelOutput(completion="<answer>2</answer>"),
        )
        result = asyncio.run(scorer(state, Target("2")))
        self.assertFalse(result.value)
        self.assertIn("missing <reasoning> block", result.explanation)

    def test_no_reward_function_skips_enforcement(self):
        # no_cot has expects_reasoning=False and no reward function
        scorer = wrap_with_constraint(arithmetic_scorer, "no_cot", enforce=True)()
        state = make_state("any text")
        result = asyncio.run(scorer(state, Target("2")))
        self.assertTrue(result.value)
        self.assertNotIn("reasoning_compliance", result.explanation)

    def test_partial_compliance_passes_with_lower_threshold(self):
        constraint = get_constraint("common_100")
        original_threshold = constraint.pass_threshold
        constraint.pass_threshold = 0.6  # allow partial compliance
        try:
            scorer = wrap_with_constraint(arithmetic_scorer, "common_100", enforce=True)()
            # 2 of 3 words are common -> compliance ≈ 0.667
            state = make_state("the the uncommon")
            result = asyncio.run(scorer(state, Target("2")))
            self.assertTrue(result.value)
            self.assertIn("reasoning_compliance=0.667", result.explanation)
        finally:
            constraint.pass_threshold = original_threshold

    def test_partial_compliance_fails_with_higher_threshold(self):
        constraint = get_constraint("common_100")
        original_threshold = constraint.pass_threshold
        constraint.pass_threshold = 0.9
        try:
            scorer = wrap_with_constraint(arithmetic_scorer, "common_100", enforce=True)()
            state = make_state("the the uncommon")  # compliance ≈ 0.667
            result = asyncio.run(scorer(state, Target("2")))
            self.assertFalse(result.value)
            self.assertIn("reasoning_compliance=0.667", result.explanation)
        finally:
            constraint.pass_threshold = original_threshold

    def test_base_score_failure_stays_false(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "max_len_3", enforce=True)()
        # wrong numeric answer -> base scorer false even though compliance is fine
        state = make_state("two add", answer="3")
        result = asyncio.run(scorer(state, Target("2")))
        self.assertFalse(result.value)
        self.assertIn("Predicted: 3, Expected: 2", result.explanation)

    def test_common_100_zero_alphabetic_words_fails(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "common_100", enforce=True)()
        state = make_state("!!!")  # no alphabetic words => compliance 0.0
        result = asyncio.run(scorer(state, Target("2")))
        self.assertFalse(result.value)
        self.assertIn("reasoning_compliance=0.000", result.explanation)

    def test_max_len_3_digits_count_as_violations(self):
        scorer = wrap_with_constraint(arithmetic_scorer, "max_len_3", enforce=True)()
        # alpha words=1, numeric sequences=1 -> compliance=0.5
        state = make_state("one 123")
        result = asyncio.run(scorer(state, Target("2")))
        self.assertFalse(result.value)
        self.assertIn("reasoning_compliance=0.500", result.explanation)


if __name__ == "__main__":
    unittest.main()
