import numpy as np
import pytest

from panam_q.math.density_matrix import DensityMatrix
from panam_q.memory.classical_memory import ClassicalExponentialMemory
from panam_q.memory.quantum_memory import QuantumMemory


def test_classical_memory_capacity_limit():
    memory = ClassicalExponentialMemory(capacity=2)

    memory.store(0.0, {"id": "a"})
    memory.store(1.0, {"id": "b"})
    memory.store(2.0, {"id": "c"})

    assert memory.size() == 2

    payloads = [record.payload["id"] for record in memory.records()]

    assert payloads == ["b", "c"]


def test_classical_memory_invalid_params():
    with pytest.raises(ValueError):
        ClassicalExponentialMemory(capacity=0)

    with pytest.raises(ValueError):
        ClassicalExponentialMemory(capacity=4, forgetting_rate=-0.1)

    memory = ClassicalExponentialMemory(capacity=4)

    with pytest.raises(ValueError):
        memory.store(-1.0)


def test_classical_memory_state_without_states_is_none():
    memory = ClassicalExponentialMemory(capacity=4)

    memory.store(0.0, {"info": "no state"})

    assert memory.memory_state(0.0) is None


def test_classical_memory_single_record():
    memory = ClassicalExponentialMemory(capacity=4)

    state = DensityMatrix.from_probabilities([0.7, 0.3])
    memory.store(0.0, state=state)

    rho_memory = memory.memory_state(0.0)

    assert rho_memory is not None
    assert rho_memory.is_valid()
    assert np.allclose(rho_memory.data, state.data, atol=1e-10)


def test_classical_memory_mixture_equal_weights():
    memory = ClassicalExponentialMemory(capacity=4)

    memory.store(0.0, state=DensityMatrix.from_probabilities([1.0, 0.0]))
    memory.store(0.0, state=DensityMatrix.from_probabilities([0.0, 1.0]))

    rho_memory = memory.memory_state(0.0)
    expected = DensityMatrix.from_probabilities([0.5, 0.5])

    assert rho_memory is not None
    assert rho_memory.is_valid()
    assert np.allclose(rho_memory.data, expected.data, atol=1e-10)


def test_classical_memory_forgetting_weights_newer_more():
    memory = ClassicalExponentialMemory(
        capacity=4,
        forgetting_rate=1.0,
    )

    memory.store(0.0, state=DensityMatrix.from_probabilities([1.0, 0.0]))
    memory.store(1.0, state=DensityMatrix.from_probabilities([0.0, 1.0]))

    rho_memory = memory.memory_state(1.0)

    assert rho_memory is not None
    assert rho_memory.is_valid()

    probability_old = rho_memory.data[0, 0].real
    probability_new = rho_memory.data[1, 1].real

    assert probability_new > probability_old


def test_classical_memory_plus_state_has_no_coherence():
    memory = ClassicalExponentialMemory(capacity=4)

    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])
    memory.store(0.0, state=plus_state)

    rho_memory = memory.memory_state(0.0)
    expected = DensityMatrix.from_probabilities([0.5, 0.5])

    assert rho_memory is not None
    assert rho_memory.is_valid()
    assert abs(rho_memory.data[0, 1]) <= 1e-12
    assert np.allclose(rho_memory.data, expected.data, atol=1e-10)


def test_classical_memory_matches_quantum_for_diagonal_input():
    classical_memory = ClassicalExponentialMemory(
        capacity=4,
        forgetting_rate=0.0,
    )

    quantum_memory = QuantumMemory(
        capacity=4,
        forgetting_rate=0.0,
        decoherence_rate=0.0,
    )

    diagonal_state = DensityMatrix.from_probabilities([0.7, 0.3])

    classical_memory.store(0.0, state=diagonal_state)
    quantum_memory.store(0.0, state=diagonal_state)

    classical_state = classical_memory.memory_state(0.0)
    quantum_state = quantum_memory.memory_state(0.0)

    assert classical_state is not None
    assert quantum_state is not None

    assert np.allclose(classical_state.data, quantum_state.data, atol=1e-10)


def test_classical_memory_state_always_diagonal():
    memory = ClassicalExponentialMemory(capacity=4)

    memory.store(0.0, state=DensityMatrix.from_state_vector([1.0, 1.0]))
    memory.store(1.0, state=DensityMatrix.from_state_vector([1.0, 1.0j]))

    rho_memory = memory.memory_state(1.0)

    assert rho_memory is not None
    assert rho_memory.is_valid()
    assert abs(rho_memory.data[0, 1]) <= 1e-12
    assert abs(rho_memory.data[1, 0]) <= 1e-12