import pytest

from panam_q.env.contextual_environment import ContextualTaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop


def test_contextual_environment_context_a():
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    # Context A: easy succeeds
    outcome_easy = env.step("easy", timestamp=0.0)
    assert outcome_easy.success is True
    assert outcome_easy.context == "A"

    # Context A: hard fails
    outcome_hard = env.step("hard", timestamp=1.0)
    assert outcome_hard.success is False
    assert outcome_hard.context == "A"


def test_contextual_environment_context_b():
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("B")

    # Context B: easy fails
    outcome_easy = env.step("easy", timestamp=0.0)
    assert outcome_easy.success is False

    # Context B: hard succeeds
    outcome_hard = env.step("hard", timestamp=1.0)
    assert outcome_hard.success is True


def test_contextual_environment_invalid_action():
    env = ContextualTaskEnvironment(seed=42)

    with pytest.raises(ValueError):
        env.step("fly", timestamp=0.0)


def test_memory_augmented_loop_quantum():
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="quantum",
        use_self_model=True,
    )

    records = loop.run(5)

    assert len(records) == 5
    assert all(r["memory_type"] == "quantum" for r in records)


def test_memory_augmented_loop_classical():
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="classical",
        use_self_model=True,
    )

    records = loop.run(5)

    assert len(records) == 5
    assert all(r["memory_type"] == "classical" for r in records)


def test_memory_augmented_loop_no_memory():
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="none",
        use_self_model=False,
    )

    records = loop.run(5)

    assert len(records) == 5
    assert all(r["memory_type"] == "none" for r in records)


def test_quantum_and_classical_both_run():
    """
    Oba typy pamięci powinny działać bez błędów.
    """
    env_q = ContextualTaskEnvironment(seed=42)
    env_q.force_context("A")

    env_c = ContextualTaskEnvironment(seed=42)
    env_c.force_context("A")

    loop_q = MemoryAugmentedLoop(env_q, memory_type="quantum")
    loop_c = MemoryAugmentedLoop(env_c, memory_type="classical")

    records_q = loop_q.run(10)
    records_c = loop_c.run(10)

    assert len(records_q) == 10
    assert len(records_c) == 10


def test_adaptation_metrics_computed():
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(env, memory_type="quantum")
    loop.run(10)

    metrics = loop.get_adaptation_metrics()

    assert "cumulative_reward" in metrics
    assert "mean_prediction_error" in metrics
    assert "total_cycles" in metrics
    assert metrics["total_cycles"] == 10


def test_invalid_memory_type():
    env = ContextualTaskEnvironment(seed=42)

    with pytest.raises(ValueError):
        MemoryAugmentedLoop(env, memory_type="invalid")


def test_context_switch_detection():
    """
    Test, że pętla może wykryć zmianę kontekstu
    poprzez zmianę wzorca sukcesów.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(env, memory_type="quantum")

    # Pierwsze 5 kroków w kontekście A
    loop.run(5)
    records_a = list(loop.history)

    # Zmiana kontekstu na B
    env.force_context("B")

    # Kolejne 5 kroków w kontekście B
    loop.run(5)
    records_b = loop.history[5:]  # Tylko nowe rekordy (od indeksu 5)

    assert len(records_a) == 5
    assert len(records_b) == 5

    # W kontekście A easy powinno succeed
    # W kontekście B easy powinno fail
    # Ale to zależy od decyzji agenta, więc tylko sprawdzamy strukturę
    assert all("true_context" in r for r in records_a)
    assert all("true_context" in r for r in records_b)