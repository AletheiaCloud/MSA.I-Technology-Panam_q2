from __future__ import annotations

import numpy as np


def _as_complex_array(state) -> np.ndarray:
    data = getattr(state, "data", state)
    return np.asarray(data, dtype=np.complex128)


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def von_neumann_entropy(state, tol: float = 1e-12) -> float:
    """
    S(rho) = -Tr(rho log rho)

    Uwaga:
    używamy logarytmu naturalnego.
    """
    matrix = _hermitian_part(_as_complex_array(state))

    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    positive = eigenvalues > tol
    if not np.any(positive):
        return 0.0

    values = eigenvalues[positive]

    return float(-np.sum(values * np.log(values)))


def quantum_relative_entropy(rho, sigma, tol: float = 1e-12) -> float:
    """
    S(rho || sigma) = Tr(rho log rho - rho log sigma)

    Implementacja:
    - obsługuje stany pełnorangowe,
    - obsługuje podstawowy warunek supportu rho ⊆ support sigma,
    - zwraca inf, jeśli support jest naruszony.
    """
    rho_matrix = _hermitian_part(_as_complex_array(rho))
    sigma_matrix = _hermitian_part(_as_complex_array(sigma))

    eig_rho, vec_rho = np.linalg.eigh(rho_matrix)
    eig_sigma, vec_sigma = np.linalg.eigh(sigma_matrix)

    eig_rho = np.clip(eig_rho, 0.0, None)
    eig_sigma = np.clip(eig_sigma, 0.0, None)

    positive_rho = eig_rho > tol
    if not np.any(positive_rho):
        return 0.0

    tr_rho_log_rho = float(
        np.sum(eig_rho[positive_rho] * np.log(eig_rho[positive_rho]))
    )

    null_sigma = eig_sigma <= tol

    if np.any(null_sigma):
        null_vectors = vec_sigma[:, null_sigma]
        projector_null = null_vectors @ null_vectors.conj().T
        overlap = np.trace(rho_matrix @ projector_null)

        if float(np.real(overlap)) > tol:
            return float("inf")

    positive_sigma = eig_sigma > tol

    if not np.any(positive_sigma):
        return float("inf")

    log_sigma = (
        vec_sigma[:, positive_sigma]
        * np.log(eig_sigma[positive_sigma])
    ) @ vec_sigma[:, positive_sigma].conj().T

    tr_rho_log_sigma = float(np.real(np.trace(rho_matrix @ log_sigma)))

    result = tr_rho_log_rho - tr_rho_log_sigma

    if result < 0.0:
        if result > -1e-10:
            result = 0.0

    return float(result)