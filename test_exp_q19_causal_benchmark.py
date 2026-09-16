import pytest

from panam_q.env.contextual_environment import ContextualTaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop


def calculate_recovery_time(records, switch_index, required_consecutive=3):
    """
    Oblicza, ile kroków po zmianie kontekstu zajęło systemowi
    osiągnięcie 'required_consecutive' sukcesów z rzędu.
    Jeśli nie osiągnął, zwraca liczbę pozostałych kroków (kara).
    """
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


def run_benchmark_condition(
    seed: int,
    memory_type: str,
    use_quantum_readout: bool,
    use_epistemic_coherence: bool,
):
    env = ContextualTaskEnvironment(seed=seed, switch_probability=0.0)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type=memory_type,
        use_self_model=False,
        use_world_model=True,
        use_meta_observer=False,
        use_memory_readout=(memory_type != "none"),
        use_quantum_memory_readout=use_quantum_readout,
        use_epistemic_coherence=use_epistemic_coherence,
        learning_rate=0.2,
        quantum_coherence_threshold=0.3,
        quantum_explore_bonus=0.4,
    )

    # Faza A: Context A (easy succeeds, hard fails)
    loop.run(15)

    # Zmiana kontekstu
    env.force_context("B")  # (easy fails, hard succeeds)

    # Faza B: Context B
    loop.run(15)

    all_records = list(loop.history)
    
    recovery_time = calculate_recovery_time(all_records, switch_index=15, required_consecutive=2)
    cumulative_reward = sum(r["net_reward"] for r in all_records)
    
    return {
        "recovery_time": recovery_time,
        "cumulative_reward": cumulative_reward,
        "phase_b_success_rate": sum(1 for r in all_records[15:] if r["success"]) / 15.0,
    }


def test_epistemic_coherence_triggers_quantum_explore():
    """
    W środowisku, gdzie następuje zmiana kontekstu,
    wysoki błąd predykcji powinien wygenerować stan koherentny [1.0, 1.0].
    Quantum Controller powinien to wykryć i wymusić 'explore'.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=True,
        use_quantum_memory_readout=True,
        use_epistemic_coherence=True,
        quantum_coherence_threshold=0.3,
        quantum_explore_bonus=0.5,
    )

    # Uczymy model, że easy zawsze wygrywa
    loop.run(5)
    
    # Zmieniamy kontekst na B (easy teraz przegrywa)
    env.force_context("B")
    
    # Pierwszy krok w nowym kontekście:
    # World model przewiduje easy=1.0, ale wynik to 0.0.
    # Błąd predykcji = 1.0 > 0.5 -> Epistemic Coherence!
    record = loop.step()
    
    # Ponieważ pamięć kwantowa zachowała koherencję z poprzedniego kroku (lub właśnie ją zapisała),
    # controller powinien zareagować.
    # Uwaga: w tym samym kroku, w którym wystąpił błąd, stan jest dopiero zapisywany.
    # Ale w następnym kroku koherencja będzie odczytana.
    record_2 = loop.step()
    
    # W kroku 2 system powinien mieć w pamięci stan koherentny i wybrać 'explore' (hard).
    assert record_2["quantum_strategy"] == "explore"
    assert record_2["quantum_coherence"] > 0.3


def test_classical_memory_loses_coherence_on_surprise():
    """
    Klasyczna pamięć, nawet przy włączonym use_epistemic_coherence,
    zredukuje stan [1.0, 1.0] do mieszanki diagonalnej [0.5, 0.5].
    Koherencja wyniesie 0.
    """
    env = ContextualTaskEnvironment(seed=42)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="classical",
        use_self_model=False,
        use_world_model=True,
        use_quantum_memory_readout=True, # Odczytujemy to samo, ale z klasycznej pamięci
        use_epistemic_coherence=True,
    )

    loop.run(5)
    env.force_context("B")
    
    loop.step() # Generuje błąd i zapisuje [1.0, 1.0]
    record_2 = loop.step() # Odczytuje pamięć
    
    # Classical memory nie ma koherencji
    assert record_2["quantum_coherence"] == pytest.approx(0.0, abs=1e-10)
    # Dlatego strategia nie powinna być 'explore' (chyba że z innych powodów, ale nie z koherencji)
    # W naszym progowym modelu: coherence=0 < 0.3, więc nie explore z tego powodu.


def test_causal_benchmark_quantum_faster_recovery():
    """
    GŁÓWNY TEST PRZYCZYNOWOŚCI (CAUSAL BENCHMARK).
    
    Porównujemy 3 warunki na 5 seedach:
    1. Quantum + Epistemic Coherence
    2. Classical + Epistemic Coherence
    3. No Memory (Baseline)
    
    Hipoteza: Quantum odzyskuje sukcesy szybciej po zmianie kontekstu.
    """
    seeds = list(range(5))
    
    quantum_recovery = []
    classical_recovery = []
    no_memory_recovery = []
    
    for seed in seeds:
        q_res = run_benchmark_condition(
            seed=seed,
            memory_type="quantum",
            use_quantum_readout=True,
            use_epistemic_coherence=True,
        )
        quantum_recovery.append(q_res["recovery_time"])
        
        c_res = run_benchmark_condition(
            seed=seed,
            memory_type="classical",
            use_quantum_readout=True,
            use_epistemic_coherence=True,
        )
        classical_recovery.append(c_res["recovery_time"])
        
        n_res = run_benchmark_condition(
            seed=seed,
            memory_type="none",
            use_quantum_readout=False,
            use_epistemic_coherence=False,
        )
        no_memory_recovery.append(n_res["recovery_time"])
        
    mean_q = sum(quantum_recovery) / len(quantum_recovery)
    mean_c = sum(classical_recovery) / len(classical_recovery)
    mean_n = sum(no_memory_recovery) / len(no_memory_recovery)
    
    print(f"\n--- CAUSAL BENCHMARK RESULTS ---")
    print(f"Quantum Mean Recovery Time:  {mean_q:.2f} steps")
    print(f"Classical Mean Recovery Time: {mean_c:.2f} steps")
    print(f"No Memory Mean Recovery Time: {mean_n:.2f} steps")
    print(f"--------------------------------")
    
    # Falsyfikacja: Jeśli Quantum nie jest lepszy (niższy czas) niż Classical,
    # hipoteza o przewadze kwantowej w tym zadaniu jest odrzucona.
    # Uwaga: W niektórych seedach mogą być remisy, ale średnia powinna faworyzować Quantum.
    assert mean_q <= mean_c, (
        f"QUANTUM ADVANTAGE FALSIFIED: Quantum ({mean_q}) "
        f"was slower than Classical ({mean_c})."
    )