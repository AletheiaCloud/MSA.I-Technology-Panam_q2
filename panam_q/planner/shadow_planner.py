from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from ..decision.decision_policy import ActionCandidate


def _clip01(value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError("Value must be finite.")
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class ShadowPlanner:
    """
    Minimalny Shadow Planner oparty na Quantum Decision Theory (QDT).

    W QDT prawdopodobieństwo wyboru akcji to:
    P(a) = f(a) + q(a)

    f(a) = klasyczna użyteczność (utility)
    q(a) = interference term (czynnik interferencji)

    Interference term modeluje awersję do niepewności:
    q(a) = - interference_weight * uncertainty(a) * difficulty(a)

    Jeśli agent jest bardzo pewny swoich możliwości (wysoki self_model),
    uncertainty spada, a interferencja zanika.
    """

    def __init__(
        self,
        interference_weight: float = 0.5,
        default_uncertainty: float = 0.5,
    ):
        self.interference_weight = float(interference_weight)
        self.default_uncertainty = _clip01(default_uncertainty)

        if not math.isfinite(self.interference_weight) or self.interference_weight < 0.0:
            raise ValueError("interference_weight must be finite and non-negative.")

    def evaluate(
        self,
        candidates: Iterable[ActionCandidate],
        world_model=None,
        self_model=None,
    ) -> List[Tuple[ActionCandidate, float, float, float]]:
        """
        Ocenia kandydatów i zwraca listę:
        (candidate, utility, interference, total_score)
        """
        results = []

        # 1. Oblicz klasyczną użyteczność (Utility)
        # Jeśli nie ma world_model, zakładamy P(success) = 1.0 (optymistyczny prior)
        utilities: Dict[str, float] = {}
        
        for candidate in candidates:
            if world_model is not None:
                world_prob, world_cost = world_model.predict(candidate.action)
                world_prob = _clip01(world_prob)
            else:
                world_prob = 1.0
                world_cost = candidate.cost

            utility = (candidate.base_value * world_prob) - world_cost
            utilities[candidate.action] = utility

        # 2. Oblicz Interference Term (q)
        # Zależy od niepewności (z world_model lub self_model) i trudności akcji.
        interferences: Dict[str, float] = {}

        for candidate in candidates:
            # Określ niepewność
            uncertainty = self.default_uncertainty

            if self_model is not None and hasattr(self_model, "uncertainty"):
                # Globalna niepewność agenta
                uncertainty = max(uncertainty, float(self_model.uncertainty))

            # Jeśli world model ma mało próbek dla tej akcji, niepewność rośnie
            if world_model is not None and hasattr(world_model, "get_samples"):
                samples = world_model.get_samples(candidate.action)
                if samples < 5:
                    # Mało danych = wysoka niepewność specyficzna dla akcji
                    uncertainty = max(uncertainty, 0.8)

            # Destruktywna interferencja: trudna akcja + wysoka niepewność = kara
            q = -self.interference_weight * uncertainty * candidate.difficulty
            interferences[candidate.action] = q

        # 3. Oblicz Total Score i znormalizuj (żeby nie było ujemnych prawdopodobieństw)
        raw_scores = []
        for candidate in candidates:
            u = utilities[candidate.action]
            q = interferences[candidate.action]
            raw_scores.append((candidate, u, q, u + q))

        # Zabezpieczenie przed ujemnymi wynikami (przesunięcie)
        min_score = min(s[3] for s in raw_scores)
        
        final_results = []
        for candidate, u, q, raw in raw_scores:
            # Przesuwamy tak, aby najgorszy wynik był >= 0.0
            safe_score = raw - min_score 
            final_results.append((candidate, u, q, safe_score))

        # Sortuj od najlepszego
        final_results.sort(key=lambda x: -x[3])

        return final_results

    def choose(
        self,
        candidates: Iterable[ActionCandidate],
        world_model=None,
        self_model=None,
    ) -> Tuple[ActionCandidate, Dict[str, float]]:
        """
        Wybiera najlepszą akcję na podstawie QDT.
        """
        evaluated = self.evaluate(candidates, world_model, self_model)
        
        if not evaluated:
            raise ValueError("Cannot choose from empty candidate list.")

        chosen_candidate = evaluated[0][0]
        
        scores = {}
        for candidate, u, q, safe_score in evaluated:
            scores[candidate.action] = safe_score

        return chosen_candidate, scores