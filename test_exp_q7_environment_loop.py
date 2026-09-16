import pytest

from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.self_referential_loop import SelfReferentialLoop
from panam_q.selfmodel.self_model import SelfModel


def make_hard_fail_environment():
    """
    Środowisko:
    - easy zawsze succeeds
    - hard zawsze fails
    """
    return TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
        easy_cost=0.0,
        hard_cost=0.2,
    )


def test_environment_invalid_params():
    with pytest.raises(ValueError):
        TaskEnvironment(easy_success_probability=1.2)

    with pytest.raises(ValueError):
        TaskEnvironment(hard_success_probability=-0.1)

    with pytest.raises(ValueError):
        TaskEnvironment(easy_cost=-1.0)

    with pytest.raises(ValueError):
        TaskEnvironment(hard_cost=-0.1)


def test_easy_action_deterministic_success():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
    )

    for i in range(3):
        outcome = environment.step("easy", timestamp=float(i))

        assert outcome.success is True
        assert outcome.performance == 1.0


def test_hard_action_deterministic_failure():
    environment = TaskEnvironment(
        seed=1,
        hard_success_probability=0.0,
        hard_cost=0.2,
    )

    outcome = environment.step("hard", timestamp=0.0)

    assert outcome.success is False
    assert outcome.performance == 0.0
    assert outcome.net_reward == pytest.approx(-0.2)


def test_environment_seed_reproducibility():
    env1 = TaskEnvironment(
        seed=123,
        hard_success_probability=0.5,
    )

    env2 = TaskEnvironment(
        seed=123,
        hard_success_probability=0.5,
    )

    sequence1 = [
        env1.step("hard", timestamp=float(i)).success
        for i in range(10)
    ]

    sequence2 = [
        env2.step("hard", timestamp=float(i)).success
        for i in range(10)
    ]

    assert sequence1 == sequence2


def test_unknown_action_raises():
    environment = TaskEnvironment()

    with pytest.raises(ValueError):
        environment.step("fly", timestamp=0.0)


def test_loop_high_self_model_initially_chooses_hard():
    environment = make_hard_fail_environment()

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

    assert record["chosen_action"] == "hard"
    assert record["self_model_used"] is True


def test_loop_failure_updates_self_model_down():
    environment = make_hard_fail_environment()

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

    assert record["success"] is False
    assert record["post_capability"] < 0.9
    assert self_model.samples == 1


def test_loop_low_self_model_prefers_easy_and_improves():
    environment = make_hard_fail_environment()

    self_model = SelfModel(
        initial_capability=0.1,
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

    assert record["chosen_action"] == "easy"
    assert record["success"] is True
    assert record["post_capability"] > 0.1


def test_loop_adapts_from_hard_to_easy_after_failures():
    environment = make_hard_fail_environment()

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

    records = loop.run(5)

    actions = [record["chosen_action"] for record in records]

    # Po podłączeniu kosztów środowiska adaptacja jest szybsza.
    # hard cost = 0.2, więc hard wymaga capability > 0.8.
    #
    # start capability = 0.9
    # step 1: hard, fail -> 0.81
    # step 2: hard, fail -> 0.729
    # step 3: easy, success
    assert actions[:2] == ["hard", "hard"]
    assert actions[2:] == ["easy", "easy", "easy"]

def test_loop_without_self_model_ablation():
    environment = make_hard_fail_environment()

    loop = SelfReferentialLoop(
        environment=environment,
        self_model=None,
        use_self_model=True,
        default_success_factor=0.5,
    )

    record = loop.step()

    # easy score = 0.6
    # hard score = 1.0 * 0.5 = 0.5
    assert record["chosen_action"] == "easy"
    assert record["self_model_used"] is False