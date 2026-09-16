from __future__ import annotations

import math
from dataclasses import dataclass

from .quantum_memory_readout import QuantumMemoryEvidence


def _validate_unit_interval(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")

    return v


def _validate_non_negative(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return v


@dataclass(frozen=True)
class QuantumMemoryControlSignal:
    """
    Sygnał sterujący wygenerowany z odczytu pamięci kwantowej.

    strategy:
        continue | explore | exploit | recalibrate

    explore_bonus:
        bonus dla akcji trudnych / informacyjnych

    recalibrate_penalty:
        kara dla akcji trudnych, gdy system powinien być ostrożny
    """

    strategy: str
    explore_bonus: float
    recalibrate_penalty: float
    success_probability: float
    coherence: float
    uncertainty: float


class QuantumMemoryController:
    """
    Minimalny kontroler decyzyjny oparty na quantum memory readout.

    Reguły:

    1. Jeśli coherence jest wysoka:
        explore

    2. Jeśli uncertainty jest wysoka, ale coherence niska:
        recalibrate

    3. Jeśli success_probability jest wysoka:
        exploit

    4. W przeciwnym razie:
        continue
    """

    def __init__(
        self,
        coherence_explore_threshold: float = 0.3,
        uncertainty_recalibrate_threshold: float = 0.7,
        exploit_success_threshold: float = 0.6,
        explore_bonus: float = 0.2,
        recalibrate_penalty: float = 0.2,
    ):
        self.coherence_explore_threshold = _validate_unit_interval(
            "coherence_explore_threshold",
            coherence_explore_threshold,
        )

        self.uncertainty_recalibrate_threshold = _validate_unit_interval(
            "uncertainty_recalibrate_threshold",
            uncertainty_recalibrate_threshold,
        )

        self.exploit_success_threshold = _validate_unit_interval(
            "exploit_success_threshold",
            exploit_success_threshold,
        )

        self.explore_bonus = _validate_non_negative(
            "explore_bonus",
            explore_bonus,
        )

        self.recalibrate_penalty = _validate_non_negative(
            "recalibrate_penalty",
            recalibrate_penalty,
        )

    def control(
        self,
        evidence: QuantumMemoryEvidence,
    ) -> QuantumMemoryControlSignal:
        """
        Podejmuje decyzję strategiczną na podstawie dowodu z pamięci.
        """
        if evidence is None or not evidence.available:
            return QuantumMemoryControlSignal(
                strategy="continue",
                explore_bonus=0.0,
                recalibrate_penalty=0.0,
                success_probability=0.0,
                coherence=0.0,
                uncertainty=0.0,
            )

        # Wysoka koherencja: system ma nierozwiązaną superpozycję,
        # więc funkcjonalnie warto eksplorować.
        if evidence.coherence >= self.coherence_explore_threshold:
            return QuantumMemoryControlSignal(
                strategy="explore",
                explore_bonus=self.explore_bonus,
                recalibrate_penalty=0.0,
                success_probability=evidence.success_probability,
                coherence=evidence.coherence,
                uncertainty=evidence.uncertainty,
            )

        # Wysoka niepewność bez koherencji: system powinien kalibrować model.
        if evidence.uncertainty >= self.uncertainty_recalibrate_threshold:
            return QuantumMemoryControlSignal(
                strategy="recalibrate",
                explore_bonus=0.0,
                recalibrate_penalty=self.recalibrate_penalty,
                success_probability=evidence.success_probability,
                coherence=evidence.coherence,
                uncertainty=evidence.uncertainty,
            )

        # Wysokie prawdopodobieństwo sukcesu i niska niepewność: exploit.
        if evidence.success_probability >= self.exploit_success_threshold:
            return QuantumMemoryControlSignal(
                strategy="exploit",
                explore_bonus=0.0,
                recalibrate_penalty=0.0,
                success_probability=evidence.success_probability,
                coherence=evidence.coherence,
                uncertainty=evidence.uncertainty,
            )

        return QuantumMemoryControlSignal(
            strategy="continue",
            explore_bonus=0.0,
            recalibrate_penalty=0.0,
            success_probability=evidence.success_probability,
            coherence=evidence.coherence,
            uncertainty=evidence.uncertainty,
        )