import pytest

from panam_q.decision.decision_policy import ActionCandidate, DecisionPolicy
from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop


def make_simple_candidates():
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


def test_policy_explore_bonus_changes_choice():
    policy_continue = DecisionPolicy(
        self_model=None,
        world_model=None,
        use_self_model=False,
        default_success_factor=0.5,
        strategy="continue",
        explore_bonus=0.2,
    )

    policy_explore = DecisionPolicy(
        self_model=None,
        world_model=None,
        use_self_model=False,
        default_success_factor=0.5,
        strategy="explore",
        explore_bonus=0.2,
    )

    candidates = make_simple_candidates()

    result_continue = policy_continue.choose(candidates)
    result_explore = policy_explore.choose(candidates)

    # Bez explore:
    # easy score = 0.6
    # hard score = 1.0 * 0.5 = 0.5
    assert result_continue.chosen_action == "easy"

    # Z explore:
    # easy score = 0.6
    # hard score = 0.5 + 0.2 = 0.7
    assert result_explore.chosen_action == "hard"
    assert result_explore.strategy == "explore"


def test_policy_recalibrate_penalty_changes_choice():
    policy = DecisionPolicy(
        self_model=None,
        world_model=None,
        use_self_model=False,
        default_success_factor=0.9,
        strategy="recalibrate",
        recalibrate_penalty=0.5,
    )

    candidates = make_simple_candidates()

    result = policy.choose(candidates)

    # Bez recalibrate:
    # easy = 0.6
    # hard = 1.0 * 0.9 = 0.9
    #
    # Z recalibrate:
    # hard = 0.9 - 0.5 = 0.4
    # easy = 0.6
    assert result.chosen_action == "easy"
    assert result.strategy == "recalibrate"


def test_policy_default_continue_no_bonus():
    policy = DecisionPolicy(
        self_model=None,
        world_model=None,
        use_self_model=False,
        default_success_factor=0.5,
    )

    candidates = make_simple_candidates()

    result = policy.choose(candidates)

    assert result.chosen_action == "easy"
    assert result.strategy == "continue"


def test_loop_meta_records_strategy():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="none",
        use_self_model=False,
        use_world_model=True,
        use_meta_observer=True,
    )

    record = loop.step()

    assert "meta_strategy" in record
    assert "meta_model_quality" in record
    assert "meta_confidence" in record
    assert loop.policy.strategy == record["meta_strategy"]


def test_loop_meta_disabled():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="none",
        use_self_model=False,
        use_world_model=True,
        use_meta_observer=False,
    )

    record = loop.step()

    assert "meta_strategy" not in record


def test_loop_meta_explore_after_world_error():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="none",
        use_self_model=False,
        use_world_model=True,
        use_meta_observer=True,
        meta_error_threshold=0.3,
    )

    record = loop.step()

    # World model startowo przewiduje P(success)=0.5,
    # ale easy zawsze succeeds -> performance=1.0.
    # Błąd world model = 0.5 > 0.3, więc strategia explore.
    assert record["meta_strategy"] == "explore"


def test_loop_meta_recalibrate_after_self_error():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=0.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="none",
        use_self_model=True,
        use_world_model=False,
        use_meta_observer=True,
        meta_error_threshold=0.3,
    )

    record = loop.step()

    # Self model startowo przewiduje capability=0.5,
    # ale performance=0.0, więc self_model_error=0.5.
    # Wysoki self error -> recalibrate.
    assert record["meta_strategy"] == "recalibrate"


def test_loop_meta_exploit_after_low_errors():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="none",
        use_self_model=False,
        use_world_model=True,
        use_meta_observer=True,
        learning_rate=0.5,
        meta_window_size=5,
        meta_error_threshold=0.3,
    )

    records = loop.run(20)

    # Po wielu sukcesach world model nauczy się, że easy succeeds.
    # Błędy predykcji spadną, model quality wzrośnie,
    # więc meta observer powinien przejść do exploit.
    assert records[-1]["meta_strategy"] == "exploit"