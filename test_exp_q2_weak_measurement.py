import numpy as np
import pytest

from panam_q.math.density_matrix import DensityMatrix
from panam_q.math.measurements import (
    projective_measurement,
    weak_measurement_qubit,
)


def test_projective_measurement_probabilities():
    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])

    result = projective_measurement(plus_state)

    assert np.isclose(result.probabilities[0], 0.5, atol=1e-10)
    assert np.isclose(result.probabilities[1], 0.5, atol=1e-10)
    assert result.post_state.is_valid()


def test_projective_nonselective_decoheres_superposition():
    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])

    result = projective_measurement(plus_state)

    off_diagonal = result.post_state.data[0, 1]

    assert np.isclose(abs(off_diagonal), 0.0, atol=1e-10)
    assert np.isclose(result.probability, 1.0, atol=1e-10)


def test_projective_selective_outcome():
    diagonal_state = DensityMatrix.from_probabilities([0.8, 0.2])

    result = projective_measurement(diagonal_state, outcome=0)

    assert np.isclose(result.probability, 0.8, atol=1e-10)
    assert result.post_state.is_valid()
    assert np.isclose(result.post_state.data[0, 0].real, 1.0, atol=1e-10)


def test_weak_measurement_strength_zero_is_identity():
    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])

    result = weak_measurement_qubit(plus_state, strength=0.0)

    assert np.allclose(result.post_state.data, plus_state.data, atol=1e-10)
    
    # Bures distance involves matrix square roots and eigenvalues,
    # so numerical noise around 1e-8 is expected for exact identity.
    assert np.isclose(result.disturbance, 0.0, atol=1e-7)


def test_weak_measurement_strength_one_matches_projective():
    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])

    weak_result = weak_measurement_qubit(plus_state, strength=1.0)
    strong_result = projective_measurement(plus_state)

    assert np.allclose(
        weak_result.post_state.data,
        strong_result.post_state.data,
        atol=1e-10,
    )


def test_weak_measurement_probabilities_sum_and_validity():
    diagonal_state = DensityMatrix.from_probabilities([0.7, 0.3])

    result = weak_measurement_qubit(diagonal_state, strength=0.4)

    total_probability = sum(result.probabilities.values())

    assert np.isclose(total_probability, 1.0, atol=1e-10)
    assert result.post_state.is_valid()


def test_weak_measurement_selective_outcome_updates_state():
    diagonal_state = DensityMatrix.from_probabilities([0.7, 0.3])

    result = weak_measurement_qubit(
        diagonal_state,
        strength=0.8,
        outcome=0,
    )

    assert result.probability > 0.0
    assert result.post_state.is_valid()

    # Wynik 0 powinien zwiększyć prawdopodobieństwo stanu bazowego |0>.
    assert result.post_state.data[0, 0].real >= 0.7 - 1e-10


def test_weak_measurement_disturbance_monotonic():
    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]

    disturbances = []

    for strength in strengths:
        result = weak_measurement_qubit(plus_state, strength=strength)
        disturbances.append(result.disturbance)

    for lower, higher in zip(disturbances, disturbances[1:]):
        assert higher >= lower - 1e-10


def test_weak_measurement_information_gain_increases_for_diagonal_state():
    diagonal_state = DensityMatrix.from_probabilities([0.8, 0.2])

    gain_zero = weak_measurement_qubit(
        diagonal_state,
        strength=0.0,
    ).information_gain

    gain_full = weak_measurement_qubit(
        diagonal_state,
        strength=1.0,
    ).information_gain

    assert gain_full > gain_zero + 1e-6


def test_weak_measurement_invalid_strength():
    state = DensityMatrix.from_state_vector([1.0, 0.0])

    with pytest.raises(ValueError):
        weak_measurement_qubit(state, strength=1.2)