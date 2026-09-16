from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Union

from ..env.contextual_environment import ContextualTaskEnvironment
from ..loop.memory_augmented_loop import MemoryAugmentedLoop


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0

    return float(sum(values) / float(len(values)))


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0

    mean_value = _mean(values)

    variance = sum(
        (value - mean_value) ** 2
        for value in values
    ) / float(len(values) - 1)

    return float(math.sqrt(variance))


@dataclass(frozen=True)
class ConditionSummary:
    """
    Podsumowanie warunku eksperymentalnego.
    """

    condition: str
    seeds: int
    mean_cumulative_reward: float
    std_cumulative_reward: float
    mean_world_accuracy: float
    mean_self_calibration: float
    mean_adaptation_improvement: float
    mean_strategy_diversity: float


def run_condition(
    seed: int,
    memory_type: str,
    cycles_per_context: int = 20,
    learning_rate: float = 0.4,
) -> dict:
    """
    Uruchamia pojedynczy warunek:

    Context A -> cycles_per_context
    switch
    Context B -> cycles_per_context
    """
    env = ContextualTaskEnvironment(seed=seed)
    env.force_context("A")

    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type=memory_type,
        use_self_model=True,
        use_world_model=True,
        use_meta_observer=True,
        learning_rate=learning_rate,
        meta_window_size=5,
        meta_error_threshold=0.3,
        explore_bonus=0.15,
        recalibrate_penalty=0.15,
    )

    # Faza A
    loop.run(cycles_per_context)

    # Switch
    env.force_context("B")

    # Faza B
    loop.run(cycles_per_context)

    all_records = list(loop.history)

    total_cycles = len(all_records)
    cumulative_reward = sum(r["net_reward"] for r in all_records)

    actions = [
        r["chosen_action"]
        for r in all_records
    ]

    # World model accuracy
    world_errors = [
        r["world_prediction_error"]
        for r in all_records
        if "world_prediction_error" in r
    ]

    mean_world_error = _mean(world_errors)
    world_accuracy = 1.0 - mean_world_error

    # Self model calibration
    self_errors = [
        r["self_model_error"]
        for r in all_records
        if "self_model_error" in r
    ]

    mean_self_error = _mean(self_errors)
    self_calibration = 1.0 - mean_self_error

    # Meta strategy diversity
    strategies = [
        r["meta_strategy"]
        for r in all_records
        if "meta_strategy" in r
    ]

    strategy_diversity = float(len(set(strategies)))

    # Adaptation after switch
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
        "seed": seed,
        "memory_type": memory_type,
        "total_cycles": total_cycles,
        "cumulative_reward": cumulative_reward,
        "actions": actions,
        "world_accuracy": world_accuracy,
        "self_calibration": self_calibration,
        "strategy_diversity": strategy_diversity,
        "adaptation_improvement": adaptation_improvement,
    }


def run_quantum_advantage_experiment(
    seeds: Union[int, List[int]] = 5,
    cycles_per_context: int = 20,
    learning_rate: float = 0.4,
) -> Dict[str, ConditionSummary]:
    """
    Uruchamia porównanie:

    quantum vs classical vs none

    dla wielu seedów.
    """
    if isinstance(seeds, int):
        if seeds <= 0:
            raise ValueError("seeds must be positive.")

        seed_list = list(range(seeds))

    else:
        seed_list = list(seeds)

        if not seed_list:
            raise ValueError("seed list cannot be empty.")

    conditions = ("quantum", "classical", "none")

    raw_results: Dict[str, List[dict]] = {
        condition: []
        for condition in conditions
    }

    for seed in seed_list:
        for condition in conditions:
            result = run_condition(
                seed=seed,
                memory_type=condition,
                cycles_per_context=cycles_per_context,
                learning_rate=learning_rate,
            )

            raw_results[condition].append(result)

    summaries: Dict[str, ConditionSummary] = {}

    for condition in conditions:
        results = raw_results[condition]

        rewards = [
            r["cumulative_reward"]
            for r in results
        ]

        world_accuracies = [
            r["world_accuracy"]
            for r in results
        ]

        self_calibrations = [
            r["self_calibration"]
            for r in results
        ]

        adaptations = [
            r["adaptation_improvement"]
            for r in results
        ]

        strategy_diversities = [
            r["strategy_diversity"]
            for r in results
        ]

        summaries[condition] = ConditionSummary(
            condition=condition,
            seeds=len(seed_list),
            mean_cumulative_reward=_mean(rewards),
            std_cumulative_reward=_std(rewards),
            mean_world_accuracy=_mean(world_accuracies),
            mean_self_calibration=_mean(self_calibrations),
            mean_adaptation_improvement=_mean(adaptations),
            mean_strategy_diversity=_mean(strategy_diversities),
        )

    return summaries


def _cohens_d_from_summaries(
    condition_a: ConditionSummary,
    condition_b: ConditionSummary,
) -> float:
    """
    Uproszczony Cohen's d dla porównania dwóch warunków.
    """
    if condition_a.seeds < 2 or condition_b.seeds < 2:
        return 0.0

    n1 = float(condition_a.seeds)
    n2 = float(condition_b.seeds)

    var1 = condition_a.std_cumulative_reward ** 2
    var2 = condition_b.std_cumulative_reward ** 2

    denominator = n1 + n2 - 2.0

    if denominator <= 0.0:
        return 0.0

    pooled_variance = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / denominator

    if pooled_variance <= 0.0:
        return 0.0

    mean_difference = (
        condition_a.mean_cumulative_reward
        - condition_b.mean_cumulative_reward
    )

    return float(mean_difference / math.sqrt(pooled_variance))


def format_advantage_report(
    summaries: Dict[str, ConditionSummary],
) -> str:
    """
    Tworzy tekstowy raport z eksperymentu.
    """
    lines = [
        "PANAM_Q EXP-Q15 QUANTUM ADVANTAGE REPORT",
        "",
    ]

    for condition, summary in summaries.items():
        lines.append(f"{condition}:")
        lines.append(f"  seeds: {summary.seeds}")
        lines.append(
            "  mean_reward: "
            f"{summary.mean_cumulative_reward:.4f} "
            f"± {summary.std_cumulative_reward:.4f}"
        )
        lines.append(
            "  world_accuracy: "
            f"{summary.mean_world_accuracy:.4f}"
        )
        lines.append(
            "  self_calibration: "
            f"{summary.mean_self_calibration:.4f}"
        )
        lines.append(
            "  adaptation_improvement: "
            f"{summary.mean_adaptation_improvement:.4f}"
        )
        lines.append(
            "  strategy_diversity: "
            f"{summary.mean_strategy_diversity:.2f}"
        )
        lines.append("")

    quantum = summaries.get("quantum")
    classical = summaries.get("classical")

    if quantum is not None and classical is not None:
        delta = (
            quantum.mean_cumulative_reward
            - classical.mean_cumulative_reward
        )

        effect_size = _cohens_d_from_summaries(quantum, classical)

        lines.append(
            "reward_delta_quantum_minus_classical: "
            f"{delta:.4f}"
        )

        lines.append(
            "effect_size_cohens_d: "
            f"{effect_size:.4f}"
        )

        if abs(delta) <= 1e-9:
            verdict = "NO_MEAN_DIFFERENCE"
        elif delta > 0.0:
            verdict = "QUANTUM_MEAN_GREATER"
        else:
            verdict = "CLASSICAL_MEAN_GREATER"

        lines.append(f"verdict: {verdict}")
        lines.append("")
        lines.append(
            "INTERPRETATION: "
            "mean reward difference alone is not sufficient proof "
            "of quantum advantage."
        )

    return "\n".join(lines)