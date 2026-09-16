from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _validate_unit_interval(name: str, value: float) -> float:
    """
    Walidacja wartości w zakresie [0, 1].
    """
    v = float(value)

    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")

    return v


def _clip01(value: float) -> float:
    """
    Przycięcie wartości do [0, 1].
    """
    v = float(value)

    if not math.isfinite(v):
        raise ValueError("Value must be finite.")

    if v < 0.0:
        return 0.0

    if v > 1.0:
        return 1.0

    return v


@dataclass(frozen=True)
class SelfModelEstimate:
    """
    Snapshot self-modelu.

    To nie jest „ja”.
    To jest estymata stanu i zdolności systemu.
    """

    capability_estimate: float
    uncertainty: float
    expected_performance: float
    conservative_performance: float
    samples: int
    mean_error: float
    last_update_time: float


class SelfModel:
    """
    Minimalny self-model jako estymata.

    capability:
        szacowana skuteczność systemu w [0, 1]

    uncertainty:
        niepewność self-modelu w [0, 1]

    learning_rate:
        jak szybko capability podąża za wynikiem

    uncertainty_rate:
        jak szybko uncertainty rośnie lub spada

    error_threshold:
        próg błędu, powyżej którego uncertainty rośnie
    """

    def __init__(
        self,
        initial_capability: float = 0.5,
        initial_uncertainty: float = 0.5,
        learning_rate: float = 0.1,
        uncertainty_rate: float = 0.1,
        error_threshold: float = 0.2,
        history_limit: int = 100,
    ):
        self.capability = _validate_unit_interval(
            "initial_capability",
            initial_capability,
        )

        self.uncertainty = _validate_unit_interval(
            "initial_uncertainty",
            initial_uncertainty,
        )

        self.learning_rate = _validate_unit_interval(
            "learning_rate",
            learning_rate,
        )

        self.uncertainty_rate = _validate_unit_interval(
            "uncertainty_rate",
            uncertainty_rate,
        )

        self.error_threshold = _validate_unit_interval(
            "error_threshold",
            error_threshold,
        )

        if int(history_limit) <= 0:
            raise ValueError("history_limit must be positive.")

        self.history_limit = int(history_limit)
        self.history = deque(maxlen=self.history_limit)

        self.samples = 0
        self.error_sum = 0.0
        self.last_update_time = 0.0

    def predict_performance(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predykcja przyszłej skuteczności.

        Na tym etapie context jest ignorowany,
        ale interfejs jest już gotowy pod przyszłe rozszerzenie.
        """
        return self.capability

    def conservative_estimate(self) -> float:
        """
        Estymata konserwatywna:

        capability * (1 - uncertainty)

        Jeśli uncertainty jest wysokie,
        system powinien ostrożniej planować działanie.
        """
        return self.capability * (1.0 - self.uncertainty)

    def record_outcome(
        self,
        performance: float,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Rejestruje wynik i aktualizuje self-model.

        performance:
            wartość z [0, 1]
            0 = pełna porażka
            1 = pełny sukces
        """
        if timestamp is None:
            timestamp = self.last_update_time + 1.0
        else:
            t = float(timestamp)

            if not math.isfinite(t) or t < 0.0:
                raise ValueError("timestamp must be non-negative and finite.")

            timestamp = t

        actual = _clip01(performance)
        predicted = self.predict_performance(context)
        error = abs(predicted - actual)

        # Aktualizacja capability.
        self.capability = (
            (1.0 - self.learning_rate) * self.capability
            + self.learning_rate * actual
        )

        # Aktualizacja uncertainty.
        if error > self.error_threshold:
            self.uncertainty = min(
                1.0,
                self.uncertainty + self.uncertainty_rate * error,
            )
        else:
            self.uncertainty = max(
                0.0,
                self.uncertainty - self.uncertainty_rate * (1.0 - error),
            )

        self.samples += 1
        self.error_sum += error
        self.last_update_time = timestamp

        self.history.append(
            {
                "timestamp": timestamp,
                "predicted": predicted,
                "actual": actual,
                "error": error,
                "context": context,
            }
        )

        return {
            "predicted": predicted,
            "actual": actual,
            "error": error,
            "capability": self.capability,
            "uncertainty": self.uncertainty,
        }

    def record_success(
        self,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Skrót: pełny sukces.
        """
        return self.record_outcome(
            performance=1.0,
            timestamp=timestamp,
            context=context,
        )

    def record_failure(
        self,
        timestamp: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Skrót: pełna porażka.
        """
        return self.record_outcome(
            performance=0.0,
            timestamp=timestamp,
            context=context,
        )

    def get_estimate(self) -> SelfModelEstimate:
        """
        Zwraca aktualny snapshot self-modelu.
        """
        mean_error = 0.0

        if self.samples > 0:
            mean_error = self.error_sum / float(self.samples)

        return SelfModelEstimate(
            capability_estimate=self.capability,
            uncertainty=self.uncertainty,
            expected_performance=self.predict_performance(),
            conservative_performance=self.conservative_estimate(),
            samples=self.samples,
            mean_error=mean_error,
            last_update_time=self.last_update_time,
        )