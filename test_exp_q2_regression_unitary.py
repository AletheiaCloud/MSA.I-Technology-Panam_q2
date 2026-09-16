import numpy as np
import pytest

from panam_q.math.measurements import _ancilla_unitary_qubit


def test_ancilla_unitary_is_unitary():
    strengths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    identity = np.eye(4, dtype=np.complex128)

    for strength in strengths:
        unitary = _ancilla_unitary_qubit(strength)

        assert np.allclose(
            unitary @ unitary.conj().T,
            identity,
            atol=1e-10,
        )

        assert np.allclose(
            unitary.conj().T @ unitary,
            identity,
            atol=1e-10,
        )


def test_ancilla_unitary_invalid_strength():
    with pytest.raises(ValueError):
        _ancilla_unitary_qubit(-0.1)

    with pytest.raises(ValueError):
        _ancilla_unitary_qubit(1.1)