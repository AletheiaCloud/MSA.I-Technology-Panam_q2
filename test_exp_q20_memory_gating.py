import pytest

from panam_q.env.contextual_environment import ContextualTaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop


def calculate_recovery_time(records, switch_index, required_consecutive=3):
    phase_b = records[switch_index:]

    consecutive_successes = 0

    for i, record in enumerate(phase_b):
        if record["success"]:
            consecutive_successes += 1

            if consecutive_successes >= required_consecutive:
                return i + 1
        else:
            consecutive_successes = 0

    return len(phase_b)


def test_memory_gating_disables_bias_on_high_error():
    """
    Jeśli w poprzednim kroku wystąpił wysoki błąd predykcji,
    waga pamięci w obecnym kroku powinna spaść do 0.0.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=True,
        use_memory_readout=True,
        memory_readout_weight=0.5,
    )

    # Uczymy model, że easy zawsze wygrywa.
    loop.run(5)

    # Zmieniamy kontekst na B.
    env.force_context("B")

    # Krok, w którym występuje szok.
    loop.step()

    # W następnym kroku pamięć powinna być odcięta.
    record = loop.step()

    for bias_value in record.get("memory_bias", {}).values():
        assert bias_value == pytest.approx(0.0)


def test_causal_benchmark_with_gating_matches_no_memory():
    """
    GŁÓWNY TEST:

    Z włączonym regime shift gate, pętla z pamięcią kwantową
    powinna adaptować się równie szybko co baseline bez pamięci.
    """
    seeds = list(range(5))
    quantum_gated_recovery = []

    for seed in seeds:
        env = ContextualTaskEnvironment(seed=seed, switch_probability=0.0)
        env.force_context("A")

        loop = MemoryAugmentedLoop(
            environment=env,
            memory_type="quantum",
            use_self_model=False,
            use_world_model=True,
            use_meta_observer=False,
            use_memory_readout=True,
            use_quantum_memory_readout=True,
            use_epistemic_coherence=True,
            learning_rate=0.2,
            quantum_coherence_threshold=0.3,
            quantum_explore_bonus=0.4,
            regime_shift_threshold=0.6,
            regime_shift_duration=3,
            regime_shift_explore_bonus=0.4,
        )

        loop.run(15)
        env.force_context("B")
        loop.run(15)

        all_records = list(loop.history)
        recovery_time = calculate_recovery_time(
            all_records,
            switch_index=15,
            required_consecutive=2,
        )

        quantum_gated_recovery.append(recovery_time)

    mean_gated = sum(quantum_gated_recovery) / len(quantum_gated_recovery)

    print(f"\n--- EXP-Q20 GATING RESULTS ---")
    print(f"Quantum (Gated) Mean Recovery Time: {mean_gated:.2f} steps")
    print(f"Previous No Memory Baseline:        6.00 steps")
    print(f"Previous Quantum (No Gating):       7.00 steps")
    print(f"--------------------------------")

    assert mean_gated <= 6.5, (
        f"GATING FAILED: Quantum with gating ({mean_gated}) "
        f"is still slower than No Memory baseline (~6.0)."
    )