from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..math.density_matrix import DensityMatrix
from ..math.distances import bures_distance
from .decoherence import dephase_density_matrix


@dataclass
class MemoryRecord:
    """
    Pojedynczy rekord pamięci.

    record_id:
        unikalny identyfikator rekordu

    timestamp:
        czas zapisu

    payload:
        dane opisowe / kontekstowe

    state:
        opcjonalny stan kwantowy związany z rekordem
    """

    record_id: int
    timestamp: float
    payload: Dict[str, Any]
    state: Optional[DensityMatrix] = None


def _to_density_matrix(state) -> DensityMatrix:
    """
    Konwersja wejścia do DensityMatrix.

    Obsługuje:
    - DensityMatrix
    - wektor stanu 1D
    - macierz 2D
    """
    if isinstance(state, DensityMatrix):
        return state.copy()

    array = np.asarray(state)

    if array.ndim == 1:
        return DensityMatrix.from_state_vector(array)

    return DensityMatrix(array)


class QuantumMemory:
    """
    Minimalna pamięć kwantowa z kontrolowaną dekoherencją.

    forgetting_rate:
        kontroluje wygaszanie wag rekordów w czasie

    decoherence_rate:
        kontroluje tłumienie elementów poza przekątną w stanach rekordów

    Memory state:
        rho_memory = Σ_i w_i * dephase(rho_i, t - t_i)
    """

    def __init__(
        self,
        capacity: int = 64,
        forgetting_rate: float = 0.0,
        decoherence_rate: float = 0.0,
    ):
        if capacity <= 0:
            raise ValueError("Memory capacity must be positive.")

        if forgetting_rate < 0.0:
            raise ValueError("Forgetting rate must be non-negative.")

        if decoherence_rate < 0.0:
            raise ValueError("Decoherence rate must be non-negative.")

        self.capacity = int(capacity)
        self.forgetting_rate = float(forgetting_rate)
        self.decoherence_rate = float(decoherence_rate)

        self._records: List[MemoryRecord] = []
        self._next_id = 0

    def store(
        self,
        timestamp: float,
        payload: Optional[Dict[str, Any]] = None,
        state=None,
    ) -> MemoryRecord:
        """
        Zapisz rekord do pamięci.
        """
        if timestamp < 0.0:
            raise ValueError("Timestamp must be non-negative.")

        record_state = None

        if state is not None:
            record_state = _to_density_matrix(state)

        record = MemoryRecord(
            record_id=self._next_id,
            timestamp=float(timestamp),
            payload=dict(payload or {}),
            state=record_state,
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

    def records(self) -> List[MemoryRecord]:
        return list(self._records)

    def _effective_weight(
        self,
        record: MemoryRecord,
        current_time: float,
    ) -> float:
        dt = max(0.0, current_time - record.timestamp)

        if self.forgetting_rate == 0.0:
            return 1.0

        return float(np.exp(-self.forgetting_rate * dt))

    def memory_state(self, current_time: float) -> Optional[DensityMatrix]:
        """
        Zwróć zagregowany stan pamięci jako DensityMatrix.

        Jeśli pamięć nie zawiera żadnych stanów kwantowych:
            zwraca None
        """
        if current_time < 0.0:
            raise ValueError("Current time must be non-negative.")

        matrices = []
        weights = []

        for record in self._records:
            if record.state is None:
                continue

            dt = max(0.0, current_time - record.timestamp)

            weight = self._effective_weight(record, current_time)

            if weight <= 1e-15:
                continue

            dephased_state = dephase_density_matrix(
                record.state,
                dt=dt,
                rate=self.decoherence_rate,
            )

            matrices.append(dephased_state.data)
            weights.append(weight)

        if not matrices:
            return None

        weights_array = np.asarray(weights, dtype=float)
        total_weight = float(np.sum(weights_array))

        if total_weight <= 1e-15:
            return None

        weights_array = weights_array / total_weight

        aggregate = np.zeros_like(matrices[0], dtype=np.complex128)

        for weight, matrix in zip(weights_array, matrices):
            aggregate += weight * matrix

        aggregate = (aggregate + aggregate.conj().T) / 2.0

        result = DensityMatrix(aggregate)

        trace = np.trace(result.data)

        if abs(trace) > 1e-15 and abs(trace - 1.0) > 1e-10:
            result = result.normalized()

        return result

    def continuity(
        self,
        previous_state,
        current_time: float,
    ) -> float:
        """
        Ciągłość temporalna jako:

        continuity = 1 - D_B(previous_state, current_memory_state) / sqrt(2)

        Zakres:
        0 <= continuity <= 1
        """
        if previous_state is None:
            return 0.0

        current_state = self.memory_state(current_time)

        if current_state is None:
            return 0.0

        distance = bures_distance(previous_state, current_state)
        max_distance = float(np.sqrt(2.0))

        continuity = 1.0 - distance / max_distance

        return float(np.clip(continuity, 0.0, 1.0))