import numpy as np
import pytest

from panam_q.math.density_matrix import DensityMatrix
from panam_q.memory.decoherence import dephase_density_matrix
from panam_q.memory.quantum_memory import QuantumMemory


def test_memory_capacity_limit():
    memory = QuantumMemory(capacity=2)

    memory.store(0.0, {"id": "a"})
    memory.store(1.0, {"id": "b"})
    memory.store(2.0, {"id": "c"})

    assert memory.size() == 2

    payloads = [record.payload["id"] for record in memory.records()]

    assert payloads == ["b", "c"]


def test_memory_invalid_params():
    with pytest.raises(ValueError):
        QuantumMemory(capacity=0)

    with pytest.raises(ValueError):
        QuantumMemory(capacity=4, forgetting_rate=-0.1)

    with pytest.raises(ValueError):
        QuantumMemory(capacity=4, decoherence_rate=-0.1)

    memory = QuantumMemory(capacity=4)

    with pytest.raises(ValueError):
        memory.store(-1.0)


def test_memory_state_single_record():
    memory = QuantumMemory(capacity=4)

    state = DensityMatrix.from_state_vector([1.0, 0.0])
    memory.store(0.0, state=state)

    rho_memory = memory.memory_state(0.0)

    assert rho_memory is not None
    assert rho_memory.is_valid()
    assert np.allclose(rho_memory.data, state.data, atol=1e-10)


def test_memory_state_mixture_equal_weights():
    memory = QuantumMemory(capacity=4)

    memory.store(0.0, state=DensityMatrix.from_state_vector([1.0, 0.0]))
    memory.store(0.0, state=DensityMatrix.from_state_vector([0.0, 1.0]))

    rho_memory = memory.memory_state(0.0)
    expected = DensityMatrix.from_probabilities([0.5, 0.5])

    assert rho_memory is not None
    assert rho_memory.is_valid()
    assert np.allclose(rho_memory.data, expected.data, atol=1e-10)


def test_forgetting_rate_weights_newer_more():
    memory = QuantumMemory(
        capacity=4,
        forgetting_rate=1.0,
        decoherence_rate=0.0,
    )

    memory.store(0.0, state=DensityMatrix.from_state_vector([1.0, 0.0]))
    memory.store(1.0, state=DensityMatrix.from_state_vector([0.0, 1.0]))

    rho_memory = memory.memory_state(1.0)

    assert rho_memory is not None
    assert rho_memory.is_valid()

    probability_old = rho_memory.data[0, 0].real
    probability_new = rho_memory.data[1, 1].real

    assert probability_new > probability_old


def test_decoherence_damps_off_diagonal():
    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])

    dephased = dephase_density_matrix(
        plus_state,
        dt=1.0,
        rate=1.0,
    )

    assert abs(dephased.data[0, 1]) < abs(plus_state.data[0, 1])
    assert np.isclose(dephased.trace().real, 1.0, atol=1e-10)
    assert dephased.is_valid()


def test_memory_decoherence_over_time():
    memory = QuantumMemory(
        capacity=4,
        forgetting_rate=0.0,
        decoherence_rate=1.0,
    )

    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])
    memory.store(0.0, state=plus_state)

    rho_t0 = memory.memory_state(0.0)
    rho_t1 = memory.memory_state(1.0)

    assert rho_t0 is not None
    assert rho_t1 is not None

    assert abs(rho_t1.data[0, 1]) < abs(rho_t0.data[0, 1])
    assert rho_t1.is_valid()


def test_memory_continuity_high_for_small_change():
    memory = QuantumMemory(
        capacity=4,
        forgetting_rate=0.0,
        decoherence_rate=0.1,
    )

    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])
    memory.store(0.0, state=plus_state)

    previous_state = memory.memory_state(0.0)
    continuity = memory.continuity(previous_state, 0.001)

    assert continuity > 0.99


def test_memory_continuity_lower_for_large_decoherence():
    memory = QuantumMemory(
        capacity=4,
        forgetting_rate=0.0,
        decoherence_rate=1.0,
    )

    plus_state = DensityMatrix.from_state_vector([1.0, 1.0])
    memory.store(0.0, state=plus_state)

    previous_state = memory.memory_state(0.0)
    continuity = memory.continuity(previous_state, 10.0)

    assert continuity < 0.99


def test_memory_state_without_states_is_none():
    memory = QuantumMemory(capacity=4)

    memory.store(0.0, {"info": "no quantum state"})

    assert memory.memory_state(0.0) is None