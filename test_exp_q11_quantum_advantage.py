import pytest
from typing import Dict, List

from panam_q.env.contextual_environment import ContextualTaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop


def run_condition(
    memory_type: str,
    use_self_model: bool,
    use_world_model: bool,
    seed: int,
    cycles_per_context: int = 20,
) -> Dict[str, float]:
    """
    Uruchamia pojedynczy warunek eksperymentalny.
    """
    env = ContextualTaskEnvironment(seed=seed)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type=memory_type,
        use_self_model=use_self_model,
        use_world_model=use_world_model,
    )

    # Faza 1: Context A
    loop.run(cycles_per_context)

    # Switch do Context B
    env.force_context("B")

    # Faza 2: Context B
    loop.run(cycles_per_context)

    # Oblicz metryki
    metrics = loop.get_adaptation_metrics()

    # Dodaj dodatkowe metryki
    total_success = sum(1 for r in loop.history if r["success"])
    total_cycles = len(loop.history)
    metrics["success_rate"] = total_success / total_cycles if total_cycles > 0 else 0.0

    return metrics


def test_quantum_vs_classical_memory():
    """
    Główny test porównawczy: Quantum vs Classical memory.
    """
    seeds = list(range(5))  # 5 seedów dla szybkiego testu

    quantum_results = []
    classical_results = []

    for seed in seeds:
        q_metrics = run_condition(
            memory_type="quantum",
            use_self_model=True,
            use_world_model=True,
            seed=seed,
        )
        quantum_results.append(q_metrics)

        c_metrics = run_condition(
            memory_type="classical",
            use_self_model=True,
            use_world_model=True,
            seed=seed,
        )
        classical_results.append(c_metrics)

    # Oblicz średnie
    q_avg_reward = sum(r["cumulative_reward"] for r in quantum_results) / len(quantum_results)
    c_avg_reward = sum(r["cumulative_reward"] for r in classical_results) / len(classical_results)

    q_avg_error = sum(r["mean_prediction_error"] for r in quantum_results) / len(quantum_results)
    c_avg_error = sum(r["mean_prediction_error"] for r in classical_results) / len(classical_results)

    # Log wyniki (do późniejszej analizy)
    print(f"\nQuantum avg reward: {q_avg_reward:.3f}, avg error: {q_avg_error:.3f}")
    print(f"Classical avg reward: {c_avg_reward:.3f}, avg error: {c_avg_error:.3f}")

    # Na tym etapie nie wymagamy, żeby quantum był lepszy.
    # Sprawdzamy tylko, że oba działają i można je porównać.
    assert len(quantum_results) == len(seeds)
    assert len(classical_results) == len(seeds)


def test_ablation_no_memory():
    """
    Ablacja: brak pamięci.
    """
    metrics = run_condition(
        memory_type="none",
        use_self_model=True,
        use_world_model=True,
        seed=42,
    )

    assert "cumulative_reward" in metrics
    assert "mean_prediction_error" in metrics


def test_ablation_no_self_model():
    """
    Ablacja: brak self-modelu.
    """
    metrics = run_condition(
        memory_type="quantum",
        use_self_model=False,
        use_world_model=True,
        seed=42,
    )

    assert "cumulative_reward" in metrics
    assert "mean_prediction_error" in metrics


def test_ablation_no_world_model():
    """
    Ablacja: brak world-modelu.
    """
    metrics = run_condition(
        memory_type="quantum",
        use_self_model=True,
        use_world_model=False,
        seed=42,
    )

    assert "cumulative_reward" in metrics
    # Bez world model, mean_prediction_error powinien być inny
    assert "mean_prediction_error" in metrics


def test_all_conditions_run_without_error():
    """
    Wszystkie warunki powinny działać bez błędów.
    """
    conditions = [
        ("quantum", True, True),
        ("classical", True, True),
        ("none", True, True),
        ("quantum", False, True),
        ("quantum", True, False),
    ]

    for memory_type, use_self, use_world in conditions:
        metrics = run_condition(
            memory_type=memory_type,
            use_self_model=use_self,
            use_world_model=use_world,
            seed=42,
        )
        assert "cumulative_reward" in metrics