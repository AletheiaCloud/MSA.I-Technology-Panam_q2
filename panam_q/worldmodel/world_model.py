from __future__ import annotations

import math
from typing import Dict, Tuple


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


class WorldModel:
    """
    Minimalny model świata.

    Uczy się:
    - P(success | action)  [success_prob]
    - E[cost | action]     [expected_cost]

    Używa Exponential Moving Average (EMA).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        default_success_prob: float = 0.5,
        default_cost: float = 0.0,
    ):
        self.learning_rate = _validate_unit_interval("learning_rate", learning_rate)
        self.default_success_prob = _validate_unit_interval(
            "default_success_prob", default_success_prob
        )
        self.default_cost = _validate_non_negative("default_cost", default_cost)

        self._success_probs: Dict[str, float] = {}
        self._expected_costs: Dict[str, float] = {}
        self._samples: Dict[str, int] = {}

    def predict(self, action: str) -> Tuple[float, float]:
        """
        Zwraca (przewidywane_prawdopodobieństwo_sukcesu, przewidywany_koszt).
        """
        if not isinstance(action, str) or not action:
            raise ValueError("Action must be a non-empty string.")

        prob = self._success_probs.get(action, self.default_success_prob)
        cost = self._expected_costs.get(action, self.default_cost)
        return prob, cost

    def update(
        self,
        action: str,
        actual_performance: float,
        actual_cost: float,
    ) -> None:
        """
        Aktualizuje model na podstawie rzeczywistego wyniku (EMA).
        """
        actual_performance = _validate_unit_interval(
            "actual_performance", actual_performance
        )
        actual_cost = _validate_non_negative("actual_cost", actual_cost)

        if action not in self._success_probs:
            self._success_probs[action] = self.default_success_prob
            self._expected_costs[action] = self.default_cost
            self._samples[action] = 0

        lr = self.learning_rate

        self._success_probs[action] = (
            (1.0 - lr) * self._success_probs[action]
            + lr * actual_performance
        )

        self._expected_costs[action] = (
            (1.0 - lr) * self._expected_costs[action]
            + lr * actual_cost
        )

        self._samples[action] += 1

    def get_prediction_error(
        self,
        action: str,
        actual_performance: float,
    ) -> float:
        """
        Oblicza błąd predykcji ZANIM model zostanie zaktualizowany.
        """
        actual_performance = _validate_unit_interval(
            "actual_performance", actual_performance
        )
        predicted_prob, _ = self.predict(action)
        return float(abs(predicted_prob - actual_performance))

    def get_samples(self, action: str) -> int:
        return self._samples.get(action, 0)