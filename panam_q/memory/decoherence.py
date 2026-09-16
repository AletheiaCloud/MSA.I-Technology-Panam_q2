from __future__ import annotations

import numpy as np

from ..math.density_matrix import DensityMatrix


def dephase_density_matrix(
    state,
    dt: float,
    rate: float,
    tol: float = 1e-12,
) -> DensityMatrix:
    """
    Kontrolowana dekoherencja / dephasing.

    Dla dt > 0 i rate > 0 elementy poza przekątną są tłumione:

    rho_ij(t) = rho_ij(0) * exp(-rate * dt), i != j

    Elementy diagonalne pozostają bez zmian.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative.")

    if rate < 0.0:
        raise ValueError("Decoherence rate must be non-negative.")

    if isinstance(state, DensityMatrix):
        rho = state.copy()
    else:
        rho = DensityMatrix(state)

    if dt == 0.0 or rate == 0.0 or rho.dim <= 1:
        return rho

    decay = float(np.exp(-rate * dt))

    data = rho.data.copy()
    dim = rho.dim

    off_diagonal_mask = ~np.eye(dim, dtype=bool)
    data[off_diagonal_mask] *= decay

    data = (data + data.conj().T) / 2.0

    result = DensityMatrix(data)

    trace = np.trace(result.data)

    if abs(trace) > tol and abs(trace - 1.0) > tol:
        result = result.normalized()

    return result