from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _validate_probability(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")

    return v


def _validate_cost(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return v


@dataclass(frozen=True)
class Outcome:
    """
    Wynik pojedynczej akcji w środowisku.

    performance:
        0.0 = porażka
        1.0 = sukces

    net_reward:
        performance - cost
    """

    action: str
    success: bool
    performance: float
    cost: float
    net_reward: float
    timestamp: float


class TaskEnvironment:
    """
    Minimalne środowisko zadań.

    Akcje:
    - easy: zadanie łatwe
    - hard: zadanie trudne

    Środowisko nie zna self-modelu agenta.
    """

    EASY = "easy"
    HARD = "hard"

    def __init__(
        self,
        seed: int = 0,
        easy_success_probability: float = 1.0,
        hard_success_probability: float = 0.3,
        easy_cost: float = 0.0,
        hard_cost: float = 0.2,
    ):
        self.seed = seed

        self.easy_success_probability = _validate_probability(
            "easy_success_probability",
            easy_success_probability,
        )

        self.hard_success_probability = _validate_probability(
            "hard_success_probability",
            hard_success_probability,
        )

        self.easy_cost = _validate_cost("easy_cost", easy_cost)
        self.hard_cost = _validate_cost("hard_cost", hard_cost)

        self._rng = random.Random(seed)

    def step(self, action: str, timestamp: float = 0.0) -> Outcome:
        """
        Wykonaj akcję w środowisku i zwróć wynik.
        """
        t = float(timestamp)

        if not math.isfinite(t) or t < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")

        if action == self.EASY:
            probability = self.easy_success_probability
            cost = self.easy_cost

        elif action == self.HARD:
            probability = self.hard_success_probability
            cost = self.hard_cost

        else:
            raise ValueError(f"Unknown action: {action}")

        success = self._rng.random() < probability

        performance = 1.0 if success else 0.0
        net_reward = performance - cost

        return Outcome(
            action=action,
            success=success,
            performance=performance,
            cost=cost,
            net_reward=net_reward,
            timestamp=t,
        )