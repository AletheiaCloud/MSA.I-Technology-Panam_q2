from __future__ import annotations

import numpy as np


class DensityMatrix:
    """
    Minimalna reprezentacja macierzy gęstości.

    Wymagania matematyczne:
    - rho = rho†
    - rho >= 0
    - Tr(rho) = 1
    """

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.complex128)

        if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Density matrix must be a square 2D array.")

    @property
    def dim(self) -> int:
        return self.data.shape[0]

    def copy(self) -> "DensityMatrix":
        return DensityMatrix(self.data.copy())

    def trace(self) -> complex:
        return complex(np.trace(self.data))

    def _hermitian_form(self) -> np.ndarray:
        return (self.data + self.data.conj().T) / 2.0

    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self._hermitian_form())

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        return bool(
            np.allclose(
                self.data,
                self.data.conj().T,
                atol=tol,
                rtol=0.0,
            )
        )

    def is_positive_semidefinite(self, tol: float = 1e-10) -> bool:
        eigenvalues = self.eigenvalues()
        return bool(np.all(eigenvalues >= -tol))

    def is_trace_one(self, tol: float = 1e-10) -> bool:
        return bool(abs(np.trace(self.data) - 1.0) <= tol)

    def is_valid(self, tol: float = 1e-10) -> bool:
        return (
            self.is_hermitian(tol)
            and self.is_positive_semidefinite(tol)
            and self.is_trace_one(tol)
        )

    def normalized(self) -> "DensityMatrix":
        tr = np.trace(self.data)

        if abs(tr) <= 1e-15:
            raise ValueError("Cannot normalize a matrix with zero trace.")

        return DensityMatrix(self.data / tr)

    def purity(self) -> float:
        return float(np.real(np.trace(self.data @ self.data)))

    @classmethod
    def from_state_vector(cls, vector) -> "DensityMatrix":
        vec = np.asarray(vector, dtype=np.complex128).reshape(-1)

        norm_sq = float(np.real(np.vdot(vec, vec)))
        if norm_sq <= 0.0:
            raise ValueError("State vector must have positive norm.")

        psi = vec / np.sqrt(norm_sq)
        rho = np.outer(psi, psi.conj())

        return cls(rho)

    @classmethod
    def from_probabilities(cls, probabilities) -> "DensityMatrix":
        probs = np.asarray(probabilities, dtype=float)

        if probs.ndim != 1:
            raise ValueError("Probabilities must be a 1D array.")

        if np.any(probs < -1e-12):
            raise ValueError("Probabilities cannot be negative.")

        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))

        if total <= 0.0:
            raise ValueError("Probabilities must sum to a positive value.")

        probs = probs / total

        return cls(np.diag(probs).astype(np.complex128))

    @classmethod
    def maximally_mixed(cls, dim: int) -> "DensityMatrix":
        if dim <= 0:
            raise ValueError("Dimension must be positive.")

        return cls(np.eye(dim, dtype=np.complex128) / float(dim))