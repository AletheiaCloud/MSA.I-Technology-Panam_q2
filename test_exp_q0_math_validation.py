import numpy as np

from panam_q.math.density_matrix import DensityMatrix
from panam_q.math.channels import QuantumChannel
from panam_q.math.entropy import (
    quantum_relative_entropy,
    von_neumann_entropy,
)
from panam_q.math.distances import (
    bures_distance,
    fidelity,
)


def test_density_matrix_from_state_vector_is_valid():
    rho = DensityMatrix.from_state_vector([1.0, 0.0])

    assert rho.is_valid()
    assert np.isclose(rho.purity(), 1.0, atol=1e-10)


def test_density_matrix_from_probabilities_is_valid():
    rho = DensityMatrix.from_probabilities([0.5, 0.5])

    assert rho.is_valid()
    assert np.isclose(rho.purity(), 0.5, atol=1e-10)


def test_density_matrix_maximally_mixed():
    rho = DensityMatrix.maximally_mixed(3)

    assert rho.is_valid()
    assert np.isclose(rho.purity(), 1.0 / 3.0, atol=1e-10)


def test_invalid_density_matrix_trace_not_one():
    rho = DensityMatrix(np.eye(2, dtype=np.complex128))

    assert not rho.is_trace_one()
    assert not rho.is_valid()


def test_invalid_density_matrix_negative_eigenvalue():
    data = np.array(
        [
            [1.2, 0.0],
            [0.0, -0.2],
        ],
        dtype=np.complex128,
    )

    rho = DensityMatrix(data)

    assert not rho.is_positive_semidefinite()
    assert not rho.is_valid()


def test_identity_channel_preserves_state():
    rho = DensityMatrix.from_probabilities([0.7, 0.3])
    channel = QuantumChannel.identity(2)

    output = channel.apply(rho)

    assert output.is_valid()
    assert np.allclose(output.data, rho.data, atol=1e-10)


def test_depolarizing_channel_is_cptp():
    channel = QuantumChannel.depolarizing_qubit(0.2)

    assert channel.is_trace_preserving()
    assert channel.is_completely_positive()
    assert channel.is_cptp()

    rho = DensityMatrix.from_state_vector([1.0, 0.0])
    output = channel.apply(rho)

    assert output.is_valid()


def test_invalid_channel_not_trace_preserving():
    operator = np.eye(2, dtype=np.complex128) * 0.5
    channel = QuantumChannel([operator])

    assert not channel.is_trace_preserving()
    assert not channel.is_cptp()


def test_entropy_bounds():
    pure = DensityMatrix.from_state_vector([1.0, 0.0])
    mixed = DensityMatrix.maximally_mixed(2)

    assert np.isclose(von_neumann_entropy(pure), 0.0, atol=1e-10)
    assert np.isclose(von_neumann_entropy(mixed), np.log(2.0), atol=1e-10)


def test_relative_entropy_nonnegative_and_zero():
    sigma = DensityMatrix.maximally_mixed(2)
    rho_same = DensityMatrix.maximally_mixed(2)

    assert np.isclose(
        quantum_relative_entropy(rho_same, sigma),
        0.0,
        atol=1e-10,
    )

    pure = DensityMatrix.from_state_vector([1.0, 0.0])

    assert quantum_relative_entropy(pure, sigma) > 0.0
    assert np.isclose(
        quantum_relative_entropy(pure, sigma),
        np.log(2.0),
        atol=1e-10,
    )


def test_fidelity_and_bures_identity():
    rho = DensityMatrix.from_state_vector([1.0, 0.0])

    assert np.isclose(fidelity(rho, rho), 1.0, atol=1e-10)
    assert np.isclose(bures_distance(rho, rho), 0.0, atol=1e-10)


def test_fidelity_orthogonal_pure_states():
    rho0 = DensityMatrix.from_state_vector([1.0, 0.0])
    rho1 = DensityMatrix.from_state_vector([0.0, 1.0])

    assert np.isclose(fidelity(rho0, rho1), 0.0, atol=1e-10)
    assert np.isclose(bures_distance(rho0, rho1), np.sqrt(2.0), atol=1e-10)