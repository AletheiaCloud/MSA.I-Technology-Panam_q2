import pytest

from panam_q.decision.candidate_factory import task_candidates
from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.self_referential_loop import SelfReferentialLoop
from panam_q.selfmodel.self_model import SelfModel


def make_hard_fail_environment(hard_cost=0.2):
    return TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
        easy_cost=0.0,
        hard_cost=hard_cost,
    )


def test_task_candidates_without_environment_zero_costs():
    candidates = task_candidates(None)

    by_action = {candidate.action: candidate for candidate in candidates}

    assert "easy" in by_action
    assert "hard" in by_action
    assert by_action["easy"].cost == 0.0
    assert by_action["hard"].cost == 0.0


def test_task_candidates_reads_environment_costs():
    environment = TaskEnvironment(
        easy_cost=0.1,
        hard_cost=0.4,
    )

    candidates = task_candidates(environment)

    by_action = {candidate.action: candidate for candidate in candidates}

    assert by_action["easy"].cost == pytest.approx(0.1)
    assert by_action["hard"].cost == pytest.approx(0.4)


def test_loop_uses_environment_costs():
    environment = make_hard_fail_environment(hard_cost=0.2)

    self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
        learning_rate=0.1,
        uncertainty_rate=0.0,
    )

    loop = SelfReferentialLoop(
        environment=environment,
        self_model=self_model,
        use_self_model=True,
    )

    record = loop.step()

    # easy score = 0.6
    # hard score = 1.0 * 0.9 - 0.2 = 0.7
    assert record["chosen_action"] == "hard"


def test_cost_makes_hard_less_attractive():
    environment = make_hard_fail_environment(hard_cost=0.2)

    self_model = SelfModel(
        initial_capability=0.6,
        initial_uncertainty=0.0,
        learning_rate=0.1,
        uncertainty_rate=0.0,
    )

    loop = SelfReferentialLoop(
        environment=environment,
        self_model=self_model,
        use_self_model=True,
    )

    record = loop.step()

    # easy score = 0.6
    # hard score = 1.0 * 0.6 - 0.2 = 0.4
    assert record["chosen_action"] == "easy"


def test_high_cost_disables_hard_even_high_capability():
    environment = make_hard_fail_environment(hard_cost=0.4)

    self_model = SelfModel(
        initial_capability=0.9,
        initial_uncertainty=0.0,
        learning_rate=0.1,
        uncertainty_rate=0.0,
    )

    loop = SelfReferentialLoop(
        environment=environment,
        self_model=self_model,
        use_self_model=True,
    )

    record = loop.step()

    # easy score = 0.6
    # hard score = 1.0 * 0.9 - 0.4 = 0.5
    assert record["chosen_action"] == "easy"


def test_ablation_default_with_cost():
    environment = make_hard_fail_environment(hard_cost=0.2)

    loop = SelfReferentialLoop(
        environment=environment,
        self_model=None,
        use_self_model=True,
        default_success_factor=0.5,
    )

    record = loop.step()

    # easy score = 0.6
    # hard score = 1.0 * 0.5 - 0.2 = 0.3
    assert record["chosen_action"] == "easy"
    assert record["self_model_used"] is False


def test_candidate_factory_custom_values():
    candidates = task_candidates(
        None,
        easy_base_value=0.2,
        hard_base_value=0.9,
        easy_difficulty=0.1,
        hard_difficulty=0.7,
    )

    by_action = {candidate.action: candidate for candidate in candidates}

    assert by_action["easy"].base_value == pytest.approx(0.2)
    assert by_action["hard"].base_value == pytest.approx(0.9)
    assert by_action["easy"].difficulty == pytest.approx(0.1)
    assert by_action["hard"].difficulty == pytest.approx(0.7)


def test_candidate_factory_rejects_negative_cost():
    class FakeBadEnvironment:
        easy_cost = -0.1
        hard_cost = 0.0

    with pytest.raises(ValueError):
        task_candidates(FakeBadEnvironment())