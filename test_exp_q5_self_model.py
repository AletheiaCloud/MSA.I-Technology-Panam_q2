import numpy as np
import pytest

from panam_q.selfmodel.self_model import SelfModel, SelfModelEstimate


def test_self_model_initial_state_valid():
    model = SelfModel()

    estimate = model.get_estimate()

    assert isinstance(estimate, SelfModelEstimate)
    assert 0.0 <= estimate.capability_estimate <= 1.0
    assert 0.0 <= estimate.uncertainty <= 1.0
    assert estimate.samples == 0
    assert estimate.mean_error == 0.0


def test_self_model_invalid_params():
    with pytest.raises(ValueError):
        SelfModel(initial_capability=1.2)

    with pytest.raises(ValueError):
        SelfModel(initial_uncertainty=-0.1)

    with pytest.raises(ValueError):
        SelfModel(learning_rate=-0.1)

    with pytest.raises(ValueError):
        SelfModel(learning_rate=1.1)

    with pytest.raises(ValueError):
        SelfModel(uncertainty_rate=1.5)

    with pytest.raises(ValueError):
        SelfModel(error_threshold=2.0)

    with pytest.raises(ValueError):
        SelfModel(history_limit=0)


def test_record_success_increases_capability():
    model = SelfModel(
        initial_capability=0.5,
        initial_uncertainty=0.5,
        learning_rate=0.1,
        uncertainty_rate=0.0,
    )

    result = model.record_success(timestamp=1.0)

    assert np.isclose(result["capability"], 0.55, atol=1e-12)
    assert result["actual"] == 1.0


def test_record_failure_decreases_capability():
    model = SelfModel(
        initial_capability=0.5,
        initial_uncertainty=0.5,
        learning_rate=0.1,
        uncertainty_rate=0.0,
    )

    result = model.record_failure(timestamp=1.0)

    assert np.isclose(result["capability"], 0.45, atol=1e-12)
    assert result["actual"] == 0.0


def test_large_error_increases_uncertainty():
    model = SelfModel(
        initial_capability=0.5,
        initial_uncertainty=0.5,
        learning_rate=0.0,
        uncertainty_rate=0.1,
        error_threshold=0.2,
    )

    result = model.record_outcome(1.0, timestamp=1.0)

    assert result["error"] == 0.5
    assert np.isclose(model.uncertainty, 0.55, atol=1e-12)


def test_small_error_decreases_uncertainty():
    model = SelfModel(
        initial_capability=0.5,
        initial_uncertainty=0.5,
        learning_rate=0.0,
        uncertainty_rate=0.1,
        error_threshold=0.2,
    )

    result = model.record_outcome(0.5, timestamp=1.0)

    assert result["error"] == 0.0
    assert np.isclose(model.uncertainty, 0.4, atol=1e-12)


def test_consistent_outcomes_reduce_uncertainty():
    model = SelfModel(
        initial_capability=0.5,
        initial_uncertainty=0.5,
        learning_rate=0.0,
        uncertainty_rate=0.1,
        error_threshold=0.2,
    )

    for i in range(3):
        model.record_outcome(0.5, timestamp=float(i))

    assert model.uncertainty < 0.5


def test_history_limit():
    model = SelfModel(
        initial_capability=0.5,
        history_limit=3,
    )

    for i in range(5):
        model.record_outcome(0.5, timestamp=float(i))

    assert len(model.history) == 3


def test_conservative_estimate_bounded():
    model = SelfModel(
        initial_capability=0.8,
        initial_uncertainty=0.5,
    )

    conservative = model.conservative_estimate()

    assert conservative <= model.capability
    assert np.isclose(conservative, 0.4, atol=1e-12)

    model_zero_uncertainty = SelfModel(
        initial_capability=0.8,
        initial_uncertainty=0.0,
    )

    assert np.isclose(
        model_zero_uncertainty.conservative_estimate(),
        model_zero_uncertainty.capability,
        atol=1e-12,
    )


def test_estimate_and_mean_error():
    model = SelfModel(
        initial_capability=0.5,
        learning_rate=0.0,
        uncertainty_rate=0.0,
    )

    result = model.record_outcome(1.0, timestamp=1.0)
    estimate = model.get_estimate()

    assert isinstance(estimate, SelfModelEstimate)
    assert estimate.samples == 1
    assert estimate.last_update_time == 1.0
    assert np.isclose(estimate.mean_error, result["error"], atol=1e-12)