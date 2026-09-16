from __future__ import annotations

import numpy as np


def _as_complex_array(state) -> np.ndarray:
    data = getattr(state, "data", state)
    return np.asarray(data, dtype=np.complex128)


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def matrix_sqrt_psd(state, tol: float = 1e-12) -> np.ndarray:
    """
    Pierwiastek macierzy dla macierzy Hermitowskiej, dodatnio półokreślonej.
    """
    matrix = _hermitian_part(_as_complex_array(state))

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    sqrt_eigenvalues = np.sqrt(eigenvalues)

    return (eigenvectors * sqrt_eigenvalues) @ eigenvectors.conj().T


def fidelity(rho, sigma, tol: float = 1e-12) -> float:
    """
    Fidelity w konwencji kwadratowej:

    F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2

    Zakres:
    0 <= F <= 1
    """
    sqrt_rho = matrix_sqrt_psd(rho, tol)
    sigma_matrix = _hermitian_part(_as_complex_array(sigma))

    inner = sqrt_rho @ sigma_matrix @ sqrt_rho
    inner = _hermitian_part(inner)

    eigenvalues = np.linalg.eigvalsh(inner)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    sqrt_trace = float(np.sum(np.sqrt(eigenvalues)))

    result = sqrt_trace * sqrt_trace

    if result < 0.0 and result > -tol:
        result = 0.0

    if result > 1.0 and result < 1.0 + tol:
        result = 1.0

    return float(np.clip(result, 0.0, 1.0))


def bures_distance(rho, sigma, tol: float = 1e-12) -> float:
    """
    D_B(rho, sigma) = sqrt(2(1 - sqrt(F(rho, sigma))))
    """
    f = fidelity(rho, sigma, tol)

    value = 2.0 * (1.0 - np.sqrt(f))

    if value < 0.0 and value > -tol:
        value = 0.0

    return float(np.sqrt(value))