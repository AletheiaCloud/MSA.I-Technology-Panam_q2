from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional


def _validate_probability(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return v


@dataclass(frozen=True)
class ContextualOutcome:
    action: str
    context: str
    success: bool
    performance: float
    cost: float
    net_reward: float
    timestamp: float


class ContextualTaskEnvironment:
    """
    Środowisko z ukrytym kontekstem.

    Context A:
        easy succeeds, hard fails

    Context B:
        easy fails, hard succeeds

    Kontekst może się zmieniać po określonej liczbie kroków
    lub losowo z prawdopodobieństwem switch_probability.
    """

    EASY = "easy"
    HARD = "hard"
    CONTEXT_A = "A"
    CONTEXT_B = "B"

    def __init__(
        self,
        seed: int = 0,
        switch_probability: float = 0.0,
        easy_cost: float = 0.0,
        hard_cost: float = 0.1,
    ):
        self.seed = seed
        self.switch_probability = _validate_probability(
            "switch_probability", switch_probability
        )
        self.easy_cost = easy_cost
        self.hard_cost = hard_cost

        self._rng = random.Random(seed)
        self._current_context = self.CONTEXT_A
        self._step_count = 0

    @property
    def current_context(self) -> str:
        """
        UWAGA: to jest dostęp tylko dla testów.
        Agent nie powinien mieć dostępu do tej właściwości.
        """
        return self._current_context

    def _maybe_switch_context(self) -> None:
        if self.switch_probability > 0.0:
            if self._rng.random() < self.switch_probability:
                if self._current_context == self.CONTEXT_A:
                    self._current_context = self.CONTEXT_B
                else:
                    self._current_context = self.CONTEXT_A

    def _get_success_probability(self, action: str) -> float:
        if self._current_context == self.CONTEXT_A:
            # Context A: easy succeeds, hard fails
            if action == self.EASY:
                return 1.0
            elif action == self.HARD:
                return 0.0
        elif self._current_context == self.CONTEXT_B:
            # Context B: easy fails, hard succeeds
            if action == self.EASY:
                return 0.0
            elif action == self.HARD:
                return 1.0

        return 0.5

    def _get_cost(self, action: str) -> float:
        if action == self.EASY:
            return self.easy_cost
        elif action == self.HARD:
            return self.hard_cost
        return 0.0

    def step(self, action: str, timestamp: float = 0.0) -> ContextualOutcome:
        if action not in (self.EASY, self.HARD):
            raise ValueError(f"Unknown action: {action}")

        self._step_count += 1
        self._maybe_switch_context()

        success_prob = self._get_success_probability(action)
        cost = self._get_cost(action)

        success = self._rng.random() < success_prob
        performance = 1.0 if success else 0.0
        net_reward = performance - cost

        return ContextualOutcome(
            action=action,
            context=self._current_context,
            success=success,
            performance=performance,
            cost=cost,
            net_reward=net_reward,
            timestamp=timestamp,
        )

    def force_context(self, context: str) -> None:
        """
        Wymusza kontekst. Tylko dla testów.
        """
        if context not in (self.CONTEXT_A, self.CONTEXT_B):
            raise ValueError(f"Unknown context: {context}")
        self._current_context = context