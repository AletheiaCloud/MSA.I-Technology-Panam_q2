from __future__ import annotations

from typing import Any, List, Optional

from .decision_policy import ActionCandidate


def _read_cost(environment: Any, name: str, default: float) -> float:
    """
    Czyta koszt z środowiska, jeśli środowisko go posiada.

    Jeśli environment is None:
        zwraca default
    """
    if environment is None:
        return default

    value = getattr(environment, name, default)

    return float(value)


def task_candidates(
    environment: Optional[Any] = None,
    easy_base_value: float = 0.6,
    hard_base_value: float = 1.0,
    easy_difficulty: float = 0.0,
    hard_difficulty: float = 1.0,
) -> List[ActionCandidate]:
    """
    Tworzy listę akcji dla decision policy.

    Jeśli environment istnieje:
        koszty są czytane ze środowiska.

    Jeśli environment nie istnieje:
        koszty wynoszą 0.
    """
    easy_cost = _read_cost(environment, "easy_cost", 0.0)
    hard_cost = _read_cost(environment, "hard_cost", 0.0)

    return [
        ActionCandidate(
            action="easy",
            base_value=easy_base_value,
            cost=easy_cost,
            difficulty=easy_difficulty,
        ),
        ActionCandidate(
            action="hard",
            base_value=hard_base_value,
            cost=hard_cost,
            difficulty=hard_difficulty,
        ),
    ]