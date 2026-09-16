from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..math.density_matrix import DensityMatrix


@dataclass
class ClassicalMemoryRecord:
    """
    Klasyczny rekord pamięci.

    Zamiast macierzy gęstości przechowuje wektor prawdopodobieństwa.
    """

    record_id: int
    timestamp: float
    payload: Dict[str, Any]
    probabilities: Optional[np.ndarray] = None


def _to_probability_vector(state) -> np.ndarray:
    """
    Konwersja stanu do klasycznego wektora prawdopodobieństwa.

    Jeśli wejściem jest DensityMatrix:
        bierzemy elementy diagonalne.

    Jeśli wejściem jest wektor stanu:
        bierzemy |psi_i|^2.

    Jeśli wejściem jest macierz 2D:
        traktujemy ją jako macierz gęstości i bierzemy diagonalę.
    """
    if isinstance(state, DensityMatrix):
        probabilities = np.real(np.diag(state.data)).astype(float)

    else:
        array = np.asarray(state)

        if array.ndim == 1:
            vector = array.astype(np.complex128)
            norm_sq = float(np.real(np.vdot(vector, vector)))

            if norm_sq <= 0.0:
                raise ValueError("State vector has zero norm.")

            probabilities = np.abs(vector) ** 2 / norm_sq

        elif array.ndim == 2:
            density_matrix = DensityMatrix(array)
            probabilities = np.real(np.diag(density_matrix.data)).astype(float)

        else:
            raise ValueError("State must be a vector, matrix, or DensityMatrix.")

    probabilities = np.clip(probabilities, 0.0, None)
    total = float(np.sum(probabilities))

    if total <= 1e-15:
        raise ValueError("Probability vector has zero mass.")

    return probabilities / total


class ClassicalExponentialMemory:
    """
    Klasyczna pamięć z wygaszaniem wag rekordów.

    forgetting_rate:
        kontroluje wygaszanie wag rekordów w czasie:

        weight = exp(-forgetting_rate * dt)

    Ta pamięć celowo nie przechowuje koherencji.
    """

    def __init__(
        self,
        capacity: int = 64,
        forgetting_rate: float = 0.0,
    ):
        if capacity <= 0:
            raise ValueError("Memory capacity must be positive.")

        if forgetting_rate < 0.0:
            raise ValueError("Forgetting rate must be non-negative.")

        self.capacity = int(capacity)
        self.forgetting_rate = float(forgetting_rate)

        self._records: List[ClassicalMemoryRecord] = []
        self._next_id = 0
        self._dim: Optional[int] = None

    def store(
        self,
        timestamp: float,
        payload: Optional[Dict[str, Any]] = None,
        state=None,
    ) -> ClassicalMemoryRecord:
        """
        Zapisz rekord do pamięci klasycznej.
        """
        if timestamp < 0.0:
            raise ValueError("Timestamp must be non-negative.")

        probabilities = None

        if state is not None:
            probabilities = _to_probability_vector(state)

            if self._dim is None:
                self._dim = len(probabilities)
            elif len(probabilities) != self._dim:
                raise ValueError("All stored states must have the same dimension.")

        record = ClassicalMemoryRecord(
            record_id=self._next_id,
            timestamp=float(timestamp),
            payload=dict(payload or {}),
            probabilities=probabilities,
        )

        self._records.append(record)
        self._next_id += 1

        if len(self._records) > self.capacity:
            self._records.pop(0)

        return record

    def size(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    def records(self) -> List[ClassicalMemoryRecord]:
        return list(self._records)

    def memory_state(self, current_time: float) -> Optional[DensityMatrix]:
        """
        Zwróć zagregowany stan pamięci jako diagonalną DensityMatrix.
        """
        if current_time < 0.0:
            raise ValueError("Current time must be non-negative.")

        vectors = []
        weights = []

        for record in self._records:
            if record.probabilities is None:
                continue

            dt = max(0.0, current_time - record.timestamp)

            if self.forgetting_rate == 0.0:
                weight = 1.0
            else:
                weight = float(np.exp(-self.forgetting_rate * dt))

            if weight <= 1e-15:
                continue

            vectors.append(record.probabilities)
            weights.append(weight)

        if not vectors:
            return None

        weights_array = np.asarray(weights, dtype=float)
        total_weight = float(np.sum(weights_array))

        if total_weight <= 1e-15:
            return None

        weights_array = weights_array / total_weight

        dimension = len(vectors[0])
        aggregate = np.zeros(dimension, dtype=float)

        for weight, vector in zip(weights_array, vectors):
            aggregate += weight * vector

        return DensityMatrix.from_probabilities(aggregate)