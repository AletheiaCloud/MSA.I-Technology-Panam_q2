from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _validate_positive_int(name: str, value: int) -> int:
    v = int(value)

    if v <= 0:
        raise ValueError(f"{name} must be positive.")

    return v


def _validate_unit_interval(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")

    return v


@dataclass(frozen=True)
class MemoryEvidence:
    """
    Dowód z pamięci.

    To nie jest „wspomnienie autobiograficzne” w sensie fenomenalnym.
    To jest statystyczna agregacja historii akcji i wyników.
    """

    samples: int
    recent_success_rate: float
    action_samples: Dict[str, int] = field(default_factory=dict)
    action_success_rates: Dict[str, float] = field(default_factory=dict)
    uncertainty: float = 1.0
    biases: Dict[str, float] = field(default_factory=dict)


class MemoryReadout:
    """
    Minimalny odczyt pamięci.

    Czyta rekordy pamięci i oblicza:

    - success rate per action
    - recent success rate
    - uncertainty
    - bias względem baseline_success

    Bias:

        bias(action) = success_rate(action) - baseline_success

    Jeśli baseline_success = 0.5:

        action z success_rate 1.0 dostaje bias +0.5
        action z success_rate 0.0 dostaje bias -0.5
    """

    def __init__(
        self,
        window: int = 20,
        baseline_success: float = 0.5,
    ):
        self.window = _validate_positive_int("window", window)
        self.baseline_success = _validate_unit_interval(
            "baseline_success",
            baseline_success,
        )

    def read(self, memory, current_time: float) -> MemoryEvidence:
        """
        Czyta pamięć i zwraca MemoryEvidence.

        current_time jest trzymany w interfejsie,
        bo przyszła wersja może używać temporal weightingu.
        """
        if memory is None:
            return MemoryEvidence(
                samples=0,
                recent_success_rate=0.0,
                action_samples={},
                action_success_rates={},
                uncertainty=1.0,
                biases={},
            )

        records = memory.records()

        if not records:
            return MemoryEvidence(
                samples=0,
                recent_success_rate=0.0,
                action_samples={},
                action_success_rates={},
                uncertainty=1.0,
                biases={},
            )

        records = records[-self.window:]

        attempts: Dict[str, int] = {}
        successes: Dict[str, int] = {}

        total_attempts = 0
        total_successes = 0

        for record in records:
            payload = getattr(record, "payload", None)

            if payload is None:
                continue

            action = payload.get("action")
            success = payload.get("success")

            if action is None or success is None:
                continue

            attempts[action] = attempts.get(action, 0) + 1
            total_attempts += 1

            if bool(success):
                successes[action] = successes.get(action, 0) + 1
                total_successes += 1

        if total_attempts == 0:
            return MemoryEvidence(
                samples=0,
                recent_success_rate=0.0,
                action_samples={},
                action_success_rates={},
                uncertainty=1.0,
                biases={},
            )

        action_success_rates: Dict[str, float] = {}

        for action, count in attempts.items():
            action_success_rates[action] = (
                successes.get(action, 0) / float(count)
            )

        biases: Dict[str, float] = {}

        for action, rate in action_success_rates.items():
            biases[action] = rate - self.baseline_success

        recent_success_rate = (
            float(total_successes) / float(total_attempts)
        )

        uncertainty = 1.0 / (1.0 + float(total_attempts))

        return MemoryEvidence(
            samples=total_attempts,
            recent_success_rate=recent_success_rate,
            action_samples=dict(attempts),
            action_success_rates=action_success_rates,
            uncertainty=uncertainty,
            biases=biases,
        )