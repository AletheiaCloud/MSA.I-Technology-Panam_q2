import math

import pytest

from panam_q.experiments.quantum_advantage import (
    format_advantage_report,
    run_condition,
    run_quantum_advantage_experiment,
)


def test_run_condition_returns_metrics():
    metrics = run_condition(
        seed=42,
        memory_type="quantum",
        cycles_per_context=5,
    )

    assert metrics["total_cycles"] == 10
    assert metrics["memory_type"] == "quantum"

    required_keys = {
        "seed",
        "memory_type",
        "total_cycles",
        "cumulative_reward",
        "actions",
        "world_accuracy",
        "self_calibration",
        "strategy_diversity",
        "adaptation_improvement",
    }

    for key in required_keys:
        assert key in metrics


def test_run_condition_invalid_memory_type():
    with pytest.raises(ValueError):
        run_condition(
            seed=42,
            memory_type="invalid",
            cycles_per_context=2,
        )


def test_run_condition_none_memory():
    metrics = run_condition(
        seed=42,
        memory_type="none",
        cycles_per_context=4,
    )

    assert metrics["total_cycles"] == 8
    assert metrics["memory_type"] == "none"


def test_experiment_summaries_all_conditions():
    summaries = run_quantum_advantage_experiment(
        seeds=2,
        cycles_per_context=5,
    )

    assert set(summaries.keys()) == {
        "quantum",
        "classical",
        "none",
    }

    for summary in summaries.values():
        assert summary.seeds == 2


def test_experiment_deterministic():
    summaries1 = run_quantum_advantage_experiment(
        seeds=2,
        cycles_per_context=5,
    )

    summaries2 = run_quantum_advantage_experiment(
        seeds=2,
        cycles_per_context=5,
    )

    for condition in summaries1:
        assert summaries1[condition].mean_cumulative_reward == pytest.approx(
            summaries2[condition].mean_cumulative_reward
        )

        assert summaries1[condition].mean_world_accuracy == pytest.approx(
            summaries2[condition].mean_world_accuracy
        )


def test_metrics_are_finite():
    summaries = run_quantum_advantage_experiment(
        seeds=2,
        cycles_per_context=5,
    )

    for summary in summaries.values():
        assert math.isfinite(summary.mean_cumulative_reward)
        assert math.isfinite(summary.std_cumulative_reward)
        assert math.isfinite(summary.mean_world_accuracy)
        assert math.isfinite(summary.mean_self_calibration)
        assert math.isfinite(summary.mean_adaptation_improvement)
        assert math.isfinite(summary.mean_strategy_diversity)


def test_adaptation_metric_bounded():
    metrics = run_condition(
        seed=42,
        memory_type="quantum",
        cycles_per_context=8,
    )

    assert -1.0 <= metrics["adaptation_improvement"] <= 1.0


def test_report_contains_conditions():
    summaries = run_quantum_advantage_experiment(
        seeds=2,
        cycles_per_context=5,
    )

    report = format_advantage_report(summaries)

    assert "quantum" in report
    assert "classical" in report
    assert "none" in report
    assert "verdict" in report


def test_current_memory_is_not_causally_active():
    """
    Ten test dokumentuje obecny stan architektury.

    Jeśli pamięć nie jest jeszcze odczytywana przez decyzję,
    quantum / classical / none powinny dawać takie samo zachowanie.

    Jeśli ten test zacznie failować po przyszłych zmianach,
    to znaczy, że pamięć zaczęła wpływać na pętlę decyzyjną.
    """
    quantum_metrics = run_condition(
        seed=11,
        memory_type="quantum",
        cycles_per_context=6,
    )

    classical_metrics = run_condition(
        seed=11,
        memory_type="classical",
        cycles_per_context=6,
    )

    none_metrics = run_condition(
        seed=11,
        memory_type="none",
        cycles_per_context=6,
    )

    assert quantum_metrics["actions"] == classical_metrics["actions"]
    assert quantum_metrics["actions"] == none_metrics["actions"]

    assert quantum_metrics["cumulative_reward"] == pytest.approx(
        classical_metrics["cumulative_reward"]
    )

    assert quantum_metrics["cumulative_reward"] == pytest.approx(
        none_metrics["cumulative_reward"]
    )