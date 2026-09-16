import pytest

from panam_q.memory.classical_memory import ClassicalExponentialMemory
from panam_q.memory.quantum_memory import QuantumMemory
from panam_q.memory.quantum_memory_readout import (
    QuantumMemoryEvidence,
    QuantumMemoryReadout,
)


def test_readout_empty_memory():
    memory = QuantumMemory(capacity=10)
    readout = QuantumMemoryReadout()

    evidence = readout.read(memory, current_time=0.0)

    assert isinstance(evidence, QuantumMemoryEvidence)
    assert evidence.available is False
    assert evidence.success_probability == 0.0
    assert evidence.coherence == 0.0


def test_readout_none_memory():
    readout = QuantumMemoryReadout()

    evidence = readout.read(None, current_time=0.0)

    assert evidence.available is False


def test_readout_invalid_indices():
    with pytest.raises(ValueError):
        QuantumMemoryReadout(success_index=-1, failure_index=1)

    with pytest.raises(ValueError):
        QuantumMemoryReadout(success_index=0, failure_index=-1)

    with pytest.raises(ValueError):
        QuantumMemoryReadout(success_index=0, failure_index=0)


def test_readout_success_basis_state():
    memory = QuantumMemory(capacity=10)
    readout = QuantumMemoryReadout()

    memory.store(
        timestamp=0.0,
        payload={"action": "easy", "success": True},
        state=[1.0, 0.0],
    )

    evidence = readout.read(memory, current_time=0.0)

    assert evidence.available is True
    assert evidence.success_probability == pytest.approx(1.0)
    assert evidence.failure_probability == pytest.approx(0.0)
    assert evidence.coherence == pytest.approx(0.0, abs=1e-10)


def test_readout_coherent_state_quantum_vs_classical():
    quantum_memory = QuantumMemory(
        capacity=10,
        forgetting_rate=0.0,
        decoherence_rate=0.0,
    )

    classical_memory = ClassicalExponentialMemory(
        capacity=10,
        forgetting_rate=0.0,
    )

    # Stan koherentny: superpozycja success/failure.
    coherent_state = [1.0, 1.0]

    quantum_memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=coherent_state,
    )

    classical_memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=coherent_state,
    )

    readout = QuantumMemoryReadout()

    quantum_evidence = readout.read(quantum_memory, current_time=0.0)
    classical_evidence = readout.read(classical_memory, current_time=0.0)

    assert quantum_evidence.available is True
    assert classical_evidence.available is True

    # Quantum memory preserves coherence.
    assert quantum_evidence.coherence == pytest.approx(1.0, abs=1e-10)

    # Classical memory is diagonal, so coherence is zero.
    assert classical_evidence.coherence == pytest.approx(0.0, abs=1e-10)

    # Quantum pure state has higher purity than classical mixture.
    assert quantum_evidence.purity > classical_evidence.purity


def test_readout_decoherence_reduces_coherence():
    memory = QuantumMemory(
        capacity=10,
        forgetting_rate=0.0,
        decoherence_rate=1.0,
    )

    memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=[1.0, 1.0],
    )

    readout = QuantumMemoryReadout()

    evidence_t0 = readout.read(memory, current_time=0.0)
    evidence_t1 = readout.read(memory, current_time=1.0)

    assert evidence_t0.coherence == pytest.approx(1.0, abs=1e-10)
    assert evidence_t1.coherence < evidence_t0.coherence


def test_readout_forgetting_weights_recent_success_more():
    memory = QuantumMemory(
        capacity=10,
        forgetting_rate=1.0,
        decoherence_rate=0.0,
    )

    # Stara porażka.
    memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": False},
        state=[0.0, 1.0],
    )

    # Nowy sukces.
    memory.store(
        timestamp=1.0,
        payload={"action": "hard", "success": True},
        state=[1.0, 0.0],
    )

    readout = QuantumMemoryReadout()

    evidence = readout.read(memory, current_time=1.0)

    # Nowy sukces powinien mieć większą wagę niż stara porażka.
    assert evidence.success_probability > evidence.failure_probability
    assert evidence.success_probability > 0.5


def test_readout_memory_without_states_is_unavailable():
    memory = QuantumMemory(capacity=10)

    memory.store(
        timestamp=0.0,
        payload={"action": "easy", "success": True},
        state=None,
    )

    readout = QuantumMemoryReadout()

    evidence = readout.read(memory, current_time=0.0)

    assert evidence.available is False