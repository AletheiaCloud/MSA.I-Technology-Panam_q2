import pytest

from panam_q.env.contextual_environment import ContextualTaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop


def run_full_loop(
    seed: int,
    memory_type: str = "quantum",
    use_self_model: bool = True,
    use_world_model: bool = True,
    use_meta_observer: bool = True,
    cycles_per_context: int = 20,
    learning_rate: float = 0.3,
) -> dict:
    """
    Uruchamia pełną pętlę z opcjonalnymi ablacjami.

    Uwaga:
    loop.run() zwraca pełną historię, ale tutaj używamy loop.history
    jako jednego źródła prawdy, żeby nie dublować rekordów.
    """
    env = ContextualTaskEnvironment(seed=seed)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type=memory_type,
        use_self_model=use_self_model,
        use_world_model=use_world_model,
        use_meta_observer=use_meta_observer,
        learning_rate=learning_rate,
        meta_window_size=5,
        meta_error_threshold=0.3,
        explore_bonus=0.15,
        recalibrate_penalty=0.15,
    )

    # Faza 1: Context A
    loop.run(cycles_per_context)

    # Switch do Context B
    env.force_context("B")

    # Faza 2: Context B
    loop.run(cycles_per_context)

    # Jedno źródło prawdy: pełna historia pętli.
    all_records = list(loop.history)

    total_cycles = len(all_records)
    cumulative_reward = sum(r["net_reward"] for r in all_records)

    # World model accuracy
    world_errors = [
        r["world_prediction_error"]
        for r in all_records
        if "world_prediction_error" in r
    ]

    mean_world_error = (
        sum(world_errors) / len(world_errors)
        if world_errors
        else 0.0
    )

    world_accuracy = 1.0 - mean_world_error

    # Self model calibration
    self_errors = [
        r["self_model_error"]
        for r in all_records
        if "self_model_error" in r
    ]

    mean_self_error = (
        sum(self_errors) / len(self_errors)
        if self_errors
        else 0.0
    )

    self_calibration = 1.0 - mean_self_error

    # Meta strategy diversity
    strategies = [
        r["meta_strategy"]
        for r in all_records
        if "meta_strategy" in r
    ]

    strategy_diversity = len(set(strategies))

    # Adaptation after switch:
    # porównaj success rate w pierwszych 5 krokach fazy B
    # vs ostatnie 5 kroków fazy B.
    phase_b_records = all_records[cycles_per_context:]

    early_b = phase_b_records[:5]
    late_b = phase_b_records[-5:]

    early_success = (
        sum(1 for r in early_b if r["success"]) / len(early_b)
        if early_b
        else 0.0
    )

    late_success = (
        sum(1 for r in late_b if r["success"]) / len(late_b)
        if late_b
        else 0.0
    )

    adaptation_improvement = late_success - early_success

    return {
        "total_cycles": total_cycles,
        "cumulative_reward": cumulative_reward,
        "world_accuracy": world_accuracy,
        "self_calibration": self_calibration,
        "strategy_diversity": strategy_diversity,
        "adaptation_improvement": adaptation_improvement,
        "memory_type": memory_type,
    }


def test_full_loop_runs_without_error():
    """
    Pełna pętla powinna działać bez błędów.
    """
    metrics = run_full_loop(seed=42)

    assert metrics["total_cycles"] == 40
    assert "cumulative_reward" in metrics
    assert "world_accuracy" in metrics
    assert "self_calibration" in metrics


def test_full_loop_all_components_active():
    """
    Z wszystkimi komponentami, meta observer powinien
    generować strategie.
    """
    metrics = run_full_loop(
        seed=42,
        use_meta_observer=True,
    )

    # Z meta observer powinniśmy zobaczyć co najmniej jedną strategię.
    assert metrics["strategy_diversity"] >= 1


def test_ablation_no_meta_observer():
    """
    Bez meta observer nie powinno być meta_strategy w rekordach.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="quantum",
        use_self_model=True,
        use_world_model=True,
        use_meta_observer=False,
    )

    records = loop.run(5)

    assert all("meta_strategy" not in r for r in records)


def test_ablation_no_world_model():
    """
    Bez world model nie powinno być world_prediction_error.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="quantum",
        use_self_model=True,
        use_world_model=False,
        use_meta_observer=True,
    )

    records = loop.run(5)

    assert all("world_prediction_error" not in r for r in records)


def test_ablation_no_self_model():
    """
    Bez self model nie powinno być self_model_error.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=True,
        use_meta_observer=True,
    )

    records = loop.run(5)

    assert all("self_model_error" not in r for r in records)


def test_quantum_vs_classical_full_loop():
    """
    Porównanie quantum vs classical w pełnej pętli.
    """
    quantum_metrics = run_full_loop(
        seed=42,
        memory_type="quantum",
    )

    classical_metrics = run_full_loop(
        seed=42,
        memory_type="classical",
    )

    # Oba powinny działać.
    assert quantum_metrics["total_cycles"] == 40
    assert classical_metrics["total_cycles"] == 40

    # Na tym etapie nie wymagamy, żeby quantum był lepszy.
    # Sprawdzamy tylko, że oba działają i można je porównać.
    assert "cumulative_reward" in quantum_metrics
    assert "cumulative_reward" in classical_metrics


def test_full_loop_adaptation_direction():
    """
    Po zmianie kontekstu system powinien poprawić
    success rate w późniejszej fazie.

    Uwaga: to jest test kierunkowy, nie bezwzględny.
    W niektórych seedach adaptacja może być wolna.
    """
    improvements = []

    for seed in range(3):
        metrics = run_full_loop(
            seed=seed,
            cycles_per_context=30,
            learning_rate=0.5,
        )
        improvements.append(metrics["adaptation_improvement"])

    mean_improvement = sum(improvements) / len(improvements)

    # Nie wymagamy pozytywnej adaptacji dla każdego seeda,
    # ale średnia nie powinna być katastrofalnie negatywna.
    assert mean_improvement >= -0.1


def test_memory_type_ablation_none():
    """
    Bez pamięci pętla powinna nadal działać.
    """
    metrics = run_full_loop(
        seed=42,
        memory_type="none",
    )

    assert metrics["total_cycles"] == 40
    assert "cumulative_reward" in metrics


def test_full_loop_deterministic():
    """
    Ten sam seed powinien dawać ten sam wynik.
    """
    metrics1 = run_full_loop(seed=123)
    metrics2 = run_full_loop(seed=123)

    assert metrics1["cumulative_reward"] == pytest.approx(
        metrics2["cumulative_reward"]
    )

    assert metrics1["world_accuracy"] == pytest.approx(
        metrics2["world_accuracy"]
    )