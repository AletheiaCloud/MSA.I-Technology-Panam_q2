from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


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


def _validate_non_negative(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")

    return v


@dataclass(frozen=True)
class ActionCandidate:
    """
    Kandydat na akcję.

    base_value:
        bazowa wartość / nagroda akcji

    cost:
        koszt akcji

    difficulty:
        trudność akcji w [0, 1]

    difficulty = 0:
        akcja łatwa, self-model prawie nie wpływa na wartość

    difficulty = 1:
        akcja trudna, self-model mocno wpływa na wartość
    """

    action: str
    base_value: float
    cost: float = 0.0
    difficulty: float = 0.0

    def __post_init__(self):
        if not self.action:
            raise ValueError("Action name cannot be empty.")

        if not math.isfinite(self.base_value):
            raise ValueError("base_value must be finite.")

        if not math.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("cost must be finite and non-negative.")

        if not math.isfinite(self.difficulty):
            raise ValueError("difficulty must be finite.")

        if self.difficulty < 0.0 or self.difficulty > 1.0:
            raise ValueError("difficulty must be in [0, 1].")


@dataclass(frozen=True)
class DecisionResult:
    """
    Wynik decyzji.

    self_model_used:
        czy self-model faktycznie został użyty

    success_factor:
        współczynnik skuteczności użyty przy ocenie akcji

    strategy:
        aktywna strategia decyzyjna
    """

    chosen_action: str
    chosen_candidate: ActionCandidate
    scores: Dict[str, float]
    self_model_used: bool
    success_factor: float
    strategy: str


class DecisionPolicy:
    """
    Minimalna polityka decyzyjna.

    Score:
        score =
            base_value
            * world_success_probability
            * difficulty_factor
            - expected_cost
            + strategy_modifier

        difficulty_factor = (1 - difficulty) + difficulty * success_factor

    Strategy modifier:
        explore:
            dodaje bonus proporcjonalny do difficulty

        recalibrate:
            odejmuje karę proporcjonalną do difficulty

        exploit / continue:
            brak dodatkowej modyfikacji
    """

    STRATEGIES = {
        "continue",
        "explore",
        "exploit",
        "recalibrate",
    }

    def __init__(
        self,
        self_model=None,
        world_model=None,
        use_self_model: bool = True,
        default_success_factor: float = 0.5,
        strategy: str = "continue",
        explore_bonus: float = 0.0,
        recalibrate_penalty: float = 0.0,
    ):
        self.self_model = self_model
        self.world_model = world_model
        self.use_self_model = bool(use_self_model)
        self.default_success_factor = _clip01(default_success_factor)

        self.set_strategy(strategy)

        self.explore_bonus = _validate_non_negative(
            "explore_bonus",
            explore_bonus,
        )

        self.recalibrate_penalty = _validate_non_negative(
            "recalibrate_penalty",
            recalibrate_penalty,
        )

    def set_strategy(self, strategy: str) -> None:
        """
        Ustawia aktywną strategię decyzyjną.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy

    def _self_model_is_available(self) -> bool:
        if not self.use_self_model:
            return False

        if self.self_model is None:
            return False

        if hasattr(self.self_model, "conservative_estimate"):
            return True

        if hasattr(self.self_model, "capability") and hasattr(
            self.self_model,
            "uncertainty",
        ):
            return True

        return False

    def _success_factor(self) -> float:
        """
        Zwraca estymowaną skuteczność systemu.

        Jeśli self-model istnieje:
            success_factor = capability * (1 - uncertainty)

        Jeśli nie:
            success_factor = default_success_factor
        """
        if not self._self_model_is_available():
            return self.default_success_factor

        if hasattr(self.self_model, "conservative_estimate"):
            return _clip01(self.self_model.conservative_estimate())

        capability = float(self.self_model.capability)
        uncertainty = float(self.self_model.uncertainty)

        return _clip01(capability * (1.0 - uncertainty))

    def score_candidate(self, candidate: ActionCandidate) -> float:
        """
        Oblicza score dla jednej akcji.
        """
        success_factor = self._success_factor()

        difficulty_factor = (
            (1.0 - candidate.difficulty)
            + candidate.difficulty * success_factor
        )

        if self.world_model is not None:
            world_prob, world_cost = self.world_model.predict(candidate.action)

            world_prob = _clip01(world_prob)
            world_cost = float(world_cost)

            if not math.isfinite(world_cost) or world_cost < 0.0:
                raise ValueError("world_cost must be finite and non-negative.")

        else:
            world_prob = 1.0
            world_cost = candidate.cost

        score = (
            candidate.base_value
            * world_prob
            * difficulty_factor
            - world_cost
        )

        # Strategy modifier.
        if self.strategy == "explore":
            score += candidate.difficulty * self.explore_bonus

        elif self.strategy == "recalibrate":
            score -= candidate.difficulty * self.recalibrate_penalty

        return float(score)

    def rank(
        self,
        candidates: Iterable[ActionCandidate],
    ) -> List[tuple[ActionCandidate, float]]:
        """
        Sortuje akcje od najlepszej do najgorszej.

        Tie-break:
        jeśli score jest identyczny, wygrywa akcja wcześniejsza alfabetycznie.
        """
        ranked = []

        for candidate in candidates:
            score = self.score_candidate(candidate)
            ranked.append((candidate, score))

        ranked.sort(key=lambda item: (-item[1], item[0].action))

        return ranked

    def choose(
        self,
        candidates: Iterable[ActionCandidate],
    ) -> DecisionResult:
        """
        Wybiera najlepszą akcję.
        """
        ranked = self.rank(candidates)

        if not ranked:
            raise ValueError("Cannot choose from empty candidate list.")

        chosen_candidate, chosen_score = ranked[0]

        scores = {}

        for candidate, score in ranked:
            if candidate.action not in scores:
                scores[candidate.action] = score

        return DecisionResult(
            chosen_action=chosen_candidate.action,
            chosen_candidate=chosen_candidate,
            scores=scores,
            self_model_used=self._self_model_is_available(),
            success_factor=self._success_factor(),
            strategy=self.strategy,
        )