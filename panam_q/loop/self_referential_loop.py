from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..decision.decision_policy import ActionCandidate, DecisionPolicy
from ..decision.candidate_factory import task_candidates


def default_candidates() -> List[ActionCandidate]:
    """
    Kompatybilna funkcja bez środowiska.

    Koszty są wtedy ustawione na 0.
    """
    return task_candidates(None)


class SelfReferentialLoop:
    """
    Minimalna pętla:

    decision
    → action
    → environment outcome
    → self-model update
    → next decision

    Od EXP-Q8:
    decision policy widzi koszty środowiska.
    """

    def __init__(
        self,
        environment,
        self_model=None,
        world_model=None,
        use_self_model: bool = True,
        default_success_factor: float = 0.5,
        candidates=None,
    ):
        self.environment = environment
        self.self_model = self_model
        self.world_model = world_model

        self.policy = DecisionPolicy(
            self_model=self_model,
            world_model=world_model,
            use_self_model=use_self_model,
            default_success_factor=default_success_factor,
        )

        if candidates is None:
            self.candidates = task_candidates(environment)
        else:
            self.candidates = list(candidates)

        self.history = []
        self.time = 0.0
    def step(self) -> Dict[str, Any]:
        """
        Wykonaj jeden cykl pętli.
        """
        decision = self.policy.choose(self.candidates)

        # 1. Oblicz błąd predykcji świata ZANIM zaktualizujesz model
        world_pred_error = None
        if self.world_model is not None:
            # Zakładamy, że przed pierwszym update'm world_model ma default_prob
            # Ale w loopie możemy to policzyć tuż przed update'm.
            pass

        outcome = self.environment.step(
            decision.chosen_action,
            timestamp=self.time,
        )

        # 2. Oblicz błąd predykcji na podstawie rzeczywistego wyniku
        if self.world_model is not None:
            world_pred_error = self.world_model.get_prediction_error(
                decision.chosen_action,
                outcome.performance,
            )

        # 3. Zaktualizuj World Model
        if self.world_model is not None:
            self.world_model.update(
                decision.chosen_action,
                outcome.performance,
                outcome.cost,
            )

        # 4. Zaktualizuj Self Model
        self_model_update = None
        if self.self_model is not None:
            self_model_update = self.self_model.record_outcome(
                outcome.performance,
                timestamp=self.time,
            )

        record = {
            "time": self.time,
            "chosen_action": decision.chosen_action,
            "scores": decision.scores,
            "self_model_used": decision.self_model_used,
            "success_factor": decision.success_factor,
            "success": outcome.success,
            "performance": outcome.performance,
            "cost": outcome.cost,
            "net_reward": outcome.net_reward,
        }

        if world_pred_error is not None:
            record["world_prediction_error"] = world_pred_error

        if self_model_update is not None:
            record["self_model_error"] = self_model_update["error"]
            record["post_capability"] = self_model_update["capability"]
            record["post_uncertainty"] = self_model_update["uncertainty"]

        self.history.append(record)
        self.time += 1.0

        return record

    def run(self, cycles: int) -> List[Dict[str, Any]]:
        """
        Wykonaj wiele cykli pętli.
        """
        cycles = int(cycles)

        if cycles < 0:
            raise ValueError("cycles must be non-negative.")

        for _ in range(cycles):
            self.step()

        return list(self.history)