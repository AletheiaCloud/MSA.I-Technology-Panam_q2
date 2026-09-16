from __future__ import annotations

import math
from typing import Dict, Optional

from .decision_policy import ActionCandidate, DecisionPolicy


class MemoryAwareDecisionPolicy(DecisionPolicy):
    """
    DecisionPolicy rozszerzony o bias z pamięci.

    Jeśli memory_bias jest pusty:
        zachowanie identyczne jak zwykły DecisionPolicy.

    Jeśli memory_bias zawiera wartości:
        score akcji jest przesuwany o bias.
    """

    def __init__(
        self,
        self_model=None,
        world_model=None,
        use_self_model: bool = True,
        default_success_factor: float = 0.5,
        strategy: str = "continue",
        explore_bonus: float = 0.0,
        recalibrate_penalty: float = 0.0,
        memory_bias: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            self_model=self_model,
            world_model=world_model,
            use_self_model=use_self_model,
            default_success_factor=default_success_factor,
            strategy=strategy,
            explore_bonus=explore_bonus,
            recalibrate_penalty=recalibrate_penalty,
        )

        self.memory_bias = dict(memory_bias or {})

    def score_candidate(self, candidate: ActionCandidate) -> float:
        """
        Score = base score + memory bias.
        """
        base_score = super().score_candidate(candidate)

        bias = float(self.memory_bias.get(candidate.action, 0.0))

        if not math.isfinite(bias):
            raise ValueError("memory bias must be finite.")

        return float(base_score + bias)