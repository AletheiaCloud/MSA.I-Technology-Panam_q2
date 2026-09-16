import pytest

from panam_q.decision.decision_policy import (
    ActionCandidate,
    DecisionPolicy,
)
from panam_q.selfmodel.self_model import SelfModel


def make_candidates():
    """
    easy:
        wartość 0.6, trudność 0

    hard:
        wartość 1.0, trudność 1
    """
    return [
        ActionCandidate(
            action="easy",
            base_value=0.6,
            cost=0.0,
            difficulty=0.0,
        ),
        ActionCandidate(
            action="hard",
            base_value=1.0,
            cost=0.0,
            difficulty=1.0,
        ),
    ]


def test_empty_candidates_raise():
    policy = DecisionPolicy()

    with pytest.raises(ValueError):
        policy.choose([])


def test_no_self_model_chooses_best_base_value():
    candidates = [
        ActionCandidate(
            action="a",
            base_value=0.7,
            cost=0.0,
            difficulty=0.0,
        ),
        ActionCandidate(
            action="b",
            base_value=1.0,
            cost=0.4,
            difficulty=0.0,
        ),
    ]

    policy = DecisionPolicy(
        self_model=None,
        use_self_model=True,
        default_success_factor=0.5,
    )

    result = policy.choose(candidates)

    # a: 0.7
    # b: 1.0 - 0.4 = 0.6
    assert result.chosen_action == "a"
    assert result.self_model_used is False


def test_high_self_model_prefers_difficult_high_value():
    self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
    )

    policy = DecisionPolicy(
        self_model=self_model,
        use_self_model=True,
    )

    result = policy.choose(make_candidates())

    # easy: 0.6
    # hard: 1.0 * 0.9 = 0.9
    assert result.chosen_action == "hard"
    assert result.self_model_used is True


def test_low_self_model_prefers_easy_safe():
    self_model = SelfModel(
        initial_capability=0.1,
        initial_uncertainty=0.0,
    )

    policy = DecisionPolicy(
        self_model=self_model,
        use_self_model=True,
    )

    result = policy.choose(make_candidates())

    # easy: 0.6
    # hard: 1.0 * 0.1 = 0.1
    assert result.chosen_action == "easy"
    assert result.self_model_used is True


def test_self_model_disabled_ignores_self_model():
    self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
    )

    policy = DecisionPolicy(
        self_model=self_model,
        use_self_model=False,
        default_success_factor=0.5,
    )

    result = policy.choose(make_candidates())

    # easy: 0.6
    # hard: 1.0 * 0.5 = 0.5
    assert result.chosen_action == "easy"
    assert result.self_model_used is False


def test_no_self_model_uses_default_success_factor():
    policy = DecisionPolicy(
        self_model=None,
        use_self_model=True,
        default_success_factor=0.8,
    )

    result = policy.choose(make_candidates())

    # easy: 0.6
    # hard: 1.0 * 0.8 = 0.8
    assert result.chosen_action == "hard"
    assert result.success_factor == pytest.approx(0.8)


def test_difficulty_zero_unaffected_by_self_model():
    high_self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
    )

    low_self_model = SelfModel(
        initial_capability=0.1,
        initial_uncertainty=0.0,
    )

    candidate = ActionCandidate(
        action="easy",
        base_value=0.6,
        cost=0.0,
        difficulty=0.0,
    )

    high_policy = DecisionPolicy(
        self_model=high_self_model,
        use_self_model=True,
    )

    low_policy = DecisionPolicy(
        self_model=low_self_model,
        use_self_model=True,
    )

    high_score = high_policy.score_candidate(candidate)
    low_score = low_policy.score_candidate(candidate)

    assert high_score == pytest.approx(0.6)
    assert low_score == pytest.approx(0.6)


def test_result_metadata():
    self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
    )

    policy = DecisionPolicy(
        self_model=self_model,
        use_self_model=True,
    )

    result = policy.choose(make_candidates())

    assert result.self_model_used is True
    assert result.success_factor == pytest.approx(0.9)
    assert "easy" in result.scores
    assert "hard" in result.scores


def test_deterministic_tie_breaking():
    candidates = [
        ActionCandidate(
            action="b",
            base_value=1.0,
            cost=0.0,
            difficulty=0.0,
        ),
        ActionCandidate(
            action="a",
            base_value=1.0,
            cost=0.0,
            difficulty=0.0,
        ),
    ]

    policy = DecisionPolicy()

    result = policy.choose(candidates)

    assert result.chosen_action == "a"


def test_ablation_changes_choice():
    self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
    )

    candidates = make_candidates()

    full_policy = DecisionPolicy(
        self_model=self_model,
        use_self_model=True,
    )

    ablated_policy = DecisionPolicy(
        self_model=None,
        use_self_model=True,
        default_success_factor=0.5,
    )

    full_result = full_policy.choose(candidates)
    ablated_result = ablated_policy.choose(candidates)

    assert full_result.chosen_action == "hard"
    assert ablated_result.chosen_action == "easy"
    assert full_result.chosen_action != ablated_result.chosen_action