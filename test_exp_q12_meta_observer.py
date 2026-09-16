import pytest

from panam_q.meta.meta_observer import MetaObserver


def test_meta_observer_invalid_params():
    with pytest.raises(ValueError):
        MetaObserver(window_size=0)

    with pytest.raises(ValueError):
        MetaObserver(error_threshold=1.2)

    with pytest.raises(ValueError):
        MetaObserver(error_threshold=-0.1)

    with pytest.raises(ValueError):
        MetaObserver(confidence_k=0.0)

    with pytest.raises(ValueError):
        MetaObserver(confidence_k=-1.0)


def test_meta_observer_initial_state():
    observer = MetaObserver()

    state = observer.get_state()

    assert state["samples"] == 0
    assert state["window_size"] == 0
    assert 0.0 <= state["model_quality"] <= 1.0
    assert 0.0 <= state["confidence"] <= 1.0
    assert state["strategy"] == "continue"


def test_low_errors_lead_to_exploit():
    observer = MetaObserver(
        window_size=10,
        error_threshold=0.3,
        confidence_k=5.0,
    )

    for i in range(20):
        observer.update(
            world_prediction_error=0.05,
            self_model_error=0.05,
            success=True,
            timestamp=float(i),
        )

    state = observer.get_state()

    assert state["strategy"] == "exploit"
    assert state["model_quality"] > 0.8
    assert state["confidence"] > 0.5


def test_high_world_error_leads_to_explore():
    observer = MetaObserver(
        window_size=5,
        error_threshold=0.3,
        confidence_k=1.0,
    )

    for i in range(5):
        observer.update(
            world_prediction_error=0.8,
            self_model_error=0.05,
            success=False,
            timestamp=float(i),
        )

    state = observer.get_state()

    assert state["strategy"] == "explore"
    assert state["mean_world_error"] > 0.3


def test_high_self_error_leads_to_recalibrate():
    observer = MetaObserver(
        window_size=5,
        error_threshold=0.3,
        confidence_k=1.0,
    )

    for i in range(5):
        observer.update(
            world_prediction_error=0.05,
            self_model_error=0.8,
            success=False,
            timestamp=float(i),
        )

    state = observer.get_state()

    assert state["strategy"] == "recalibrate"
    assert state["mean_self_error"] > 0.3


def test_window_size_limits_history():
    observer = MetaObserver(window_size=3)

    for i in range(5):
        observer.update(
            world_prediction_error=0.1,
            timestamp=float(i),
        )

    state = observer.get_state()

    assert state["samples"] == 5
    assert state["window_size"] == 3


def test_confidence_increases_with_samples():
    observer = MetaObserver(
        window_size=20,
        confidence_k=10.0,
    )

    confidence_before = observer.get_state()["confidence"]

    observer.update(
        world_prediction_error=0.1,
        timestamp=0.0,
    )

    confidence_after_one = observer.get_state()["confidence"]

    observer.update(
        world_prediction_error=0.1,
        timestamp=1.0,
    )

    confidence_after_two = observer.get_state()["confidence"]

    assert confidence_after_one > confidence_before
    assert confidence_after_two > confidence_after_one


def test_update_returns_state_keys():
    observer = MetaObserver()

    state = observer.update(
        world_prediction_error=0.2,
        self_model_error=0.1,
        success=True,
        timestamp=0.0,
    )

    required_keys = {
        "samples",
        "window_size",
        "model_quality",
        "confidence",
        "strategy",
        "mean_world_error",
        "mean_self_error",
    }

    for key in required_keys:
        assert key in state