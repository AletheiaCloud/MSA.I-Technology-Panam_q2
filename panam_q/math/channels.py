from __future__ import annotations

import numpy as np

from .density_matrix import DensityMatrix


class QuantumChannel:
    """
    Kanał kwantowy reprezentowany przez operatory Krausa.

    Wymagania:
    E(rho) = Σ_i K_i rho K_i†
    Trace preservation:
    Σ_i K_i† K_i = I
    """

    def __init__(self, kraus_operators):
        if not kraus_operators:
            raise ValueError("Kraus operator list cannot be empty.")

        first = np.asarray(kraus_operators[0], dtype=np.complex128)

        if first.ndim != 2 or first.shape[0] != first.shape[1]:
            raise ValueError("Kraus operators must be square matrices.")

        dim = first.shape[0]
        self.kraus = []

        for operator in kraus_operators:
            arr = np.asarray(operator, dtype=np.complex128)

            if arr.shape != (dim, dim):
                raise ValueError("All Kraus operators must have the same shape.")

            self.kraus.append(arr)

        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @classmethod
    def identity(cls, dim: int) -> "QuantumChannel":
        if dim <= 0:
            raise ValueError("Dimension must be positive.")

        return cls([np.eye(dim, dtype=np.complex128)])

    @classmethod
    def depolarizing_qubit(cls, p: float) -> "QuantumChannel":
        """
        Standardowy kanał depolaryzujący dla qubitu:

        E(rho) = (1 - p) rho + p I/2
        """
        if p < 0.0 or p > 1.0:
            raise ValueError("Depolarizing probability must be in [0, 1].")

        identity = np.eye(2, dtype=np.complex128)

        sigma_x = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.complex128,
        )

        sigma_y = np.array(
            [
                [0.0, -1.0j],
                [1.0j, 0.0],
            ],
            dtype=np.complex128,
        )

        sigma_z = np.array(
            [
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.complex128,
        )

        k0 = np.sqrt(1.0 - 3.0 * p / 4.0) * identity
        k1 = np.sqrt(p / 4.0) * sigma_x
        k2 = np.sqrt(p / 4.0) * sigma_y
        k3 = np.sqrt(p / 4.0) * sigma_z

        return cls([k0, k1, k2, k3])

    def _sum_k_dagger_k(self) -> np.ndarray:
        result = np.zeros((self._dim, self._dim), dtype=np.complex128)

        for operator in self.kraus:
            result += operator.conj().T @ operator

        return result

    def is_trace_preserving(self, tol: float = 1e-10) -> bool:
        identity = np.eye(self._dim, dtype=np.complex128)
        return bool(
            np.allclose(
                self._sum_k_dagger_k(),
                identity,
                atol=tol,
                rtol=0.0,
            )
        )

    def apply(self, state) -> DensityMatrix:
        if isinstance(state, DensityMatrix):
            rho = state.data
        else:
            rho = np.asarray(state, dtype=np.complex128)

        if rho.shape != (self._dim, self._dim):
            raise ValueError("State dimension does not match channel dimension.")

        output = np.zeros((self._dim, self._dim), dtype=np.complex128)

        for operator in self.kraus:
            output += operator @ rho @ operator.conj().T

        output = (output + output.conj().T) / 2.0

        return DensityMatrix(output)

    def choi_matrix(self) -> np.ndarray:
        dim_squared = self._dim * self._dim
        choi = np.zeros((dim_squared, dim_squared), dtype=np.complex128)

        for operator in self.kraus:
            vector = operator.reshape(dim_squared, order="F")
            choi += np.outer(vector, vector.conj())

        return (choi + choi.conj().T) / 2.0

    def is_completely_positive(self, tol: float = 1e-10) -> bool:
        choi = self.choi_matrix()
        eigenvalues = np.linalg.eigvalsh(choi)
        return bool(np.all(eigenvalues >= -tol))

    def is_cptp(self, tol: float = 1e-10) -> bool:
        return self.is_trace_preserving(tol) and self.is_completely_positive(tol)