from __future__ import annotations

import math
from dataclasses import dataclass

from ..math.entropy import von_neumann_entropy


def _clip01(value: float) -> float:
    v = float(value)

    if not math.isfinite(v):
        raise ValueError("Value must be finite.")

    if v < 0.0:
        return 0.0

    if v > 1.0:
        return 1.0

    return v


@dataclass(frozen=True)
class QuantumMemoryEvidence:
    """
    Dowód odczytany z pamięci kwantowej lub klasycznej.

    Zakładana baza dla pierwszych dwóch wymiarów:

        index 0 = success
        index 1 = failure

    coherence:
        miara koherencji między success/failure
        dla stanu diagonalnego wynosi 0

    purity:
        1.0 dla stanu czystego
        mniej dla stanu mieszanego

    entropy:
        entropia von Neumanna

    uncertainty:
        znormalizowana entropia
    """

    available: bool
    success_probability: float
    failure_probability: float
    coherence: float
    purity: float
    entropy: float
    uncertainty: float


class QuantumMemoryReadout:
    """
    Minimalny odczyt stanu pamięci przez rho_memory.

    Działa z:
    - QuantumMemory
    - ClassicalExponentialMemory

    ale tylko QuantumMemory może zwrócić koherencję,
    jeśli przechowuje stany z off-diagonals.
    """

    def __init__(
        self,
        success_index: int = 0,
        failure_index: int = 1,
    ):
        success_index = int(success_index)
        failure_index = int(failure_index)

        if success_index < 0:
            raise ValueError("success_index must be non-negative.")

        if failure_index < 0:
            raise ValueError("failure_index must be non-negative.")

        if success_index == failure_index:
            raise ValueError("success_index and failure_index must differ.")

        self.success_index = success_index
        self.failure_index = failure_index

    def _empty(self) -> QuantumMemoryEvidence:
        return QuantumMemoryEvidence(
            available=False,
            success_probability=0.0,
            failure_probability=0.0,
            coherence=0.0,
            purity=0.0,
            entropy=0.0,
            uncertainty=0.0,
        )

    def read(self, memory, current_time: float) -> QuantumMemoryEvidence:
        """
        Czyta memory_state(current_time) i zwraca metryki.
        """
        if memory is None:
            return self._empty()

        if not hasattr(memory, "memory_state"):
            return self._empty()

        rho = memory.memory_state(current_time)

        if rho is None:
            return self._empty()

        data = rho.data
        dim = data.shape[0]

        max_index = max(self.success_index, self.failure_index)

        if dim <= max_index:
            return self._empty()

        success_probability = _clip01(
            float(data[self.success_index, self.success_index].real)
        )

        failure_probability = _clip01(
            float(data[self.failure_index, self.failure_index].real)
        )

        coherence = _clip01(
            float(
                2.0 * abs(
                    data[self.success_index, self.failure_index]
                )
            )
        )

        purity = float(rho.purity())

        entropy = float(von_neumann_entropy(rho))

        if dim > 1:
            max_entropy = math.log(float(dim))
        else:
            max_entropy = 1.0

        if max_entropy <= 0.0:
            uncertainty = 0.0
        else:
            uncertainty = _clip01(entropy / max_entropy)

        return QuantumMemoryEvidence(
            available=True,
            success_probability=success_probability,
            failure_probability=failure_probability,
            coherence=coherence,
            purity=purity,
            entropy=entropy,
            uncertainty=uncertainty,
        )