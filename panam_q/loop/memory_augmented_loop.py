from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..decision.candidate_factory import task_candidates
from ..decision.memory_policy import MemoryAwareDecisionPolicy
from ..worldmodel.world_model import WorldModel
from ..selfmodel.self_model import SelfModel
from ..memory.quantum_memory import QuantumMemory
from ..memory.classical_memory import ClassicalExponentialMemory
from ..meta.meta_observer import MetaObserver
from ..memory.memory_readout import MemoryReadout
from ..memory.quantum_memory_readout import QuantumMemoryReadout
from ..memory.quantum_memory_controller import QuantumMemoryController
from ..planner.shadow_planner import ShadowPlanner


class ShadowDecisionResult:
    """
    Minimalny substytut DecisionResult dla Shadow Plannera.
    """
    pass


class MemoryAugmentedLoop:
    """
    Pętla z pamięcią kwantową lub klasyczną.

    EXP-Q13: meta observer
    EXP-Q16: memory readout
    EXP-Q18: quantum memory readout
    EXP-Q19: epistemic coherence
    EXP-Q20: regime shift gate
    EXP-Q21: shadow planner (QDT)

    Domyślnie nowe mechanizmy są wyłączone,
    żeby stare testy pozostawały ważne.
    """

    def __init__(
        self,
        environment,
        memory_type: str = "quantum",
        use_self_model: bool = True,
        use_world_model: bool = True,
        use_meta_observer: bool = False,
        use_memory_readout: bool = False,
        use_quantum_memory_readout: bool = False,
        use_epistemic_coherence: bool = False,
        learning_rate: float = 0.1,
        meta_window_size: int = 10,
        meta_error_threshold: float = 0.3,
        explore_bonus: float = 0.1,
        recalibrate_penalty: float = 0.1,
        memory_readout_weight: float = 0.2,
        memory_window: int = 20,
        memory_capacity: int = 64,
        memory_forgetting_rate: float = 0.5,
        memory_decoherence_rate: float = 0.05,
        quantum_coherence_threshold: float = 0.3,
        quantum_uncertainty_threshold: float = 0.7,
        quantum_exploit_success_threshold: float = 0.6,
        quantum_explore_bonus: float = 0.2,
        quantum_recalibrate_penalty: float = 0.2,
        regime_shift_threshold: float = 0.6,
        regime_shift_duration: int = 0,
        regime_shift_explore_bonus: float = 0.4,
        **kwargs,
    ):
        self.environment = environment
        self.memory_type = memory_type

        self.use_self_model = bool(use_self_model)
        self.use_world_model = bool(use_world_model)
        self.use_meta_observer = bool(use_meta_observer)
        self.use_memory_readout = bool(use_memory_readout)
        self.use_quantum_memory_readout = bool(use_quantum_memory_readout)
        self.use_epistemic_coherence = bool(use_epistemic_coherence)

        self.memory_readout_weight = float(memory_readout_weight)

        if not math.isfinite(self.memory_readout_weight):
            raise ValueError("memory_readout_weight must be finite.")

        memory_capacity = int(memory_capacity)

        if memory_capacity <= 0:
            raise ValueError("memory_capacity must be positive.")

        self.memory_capacity = memory_capacity

        # Tworzenie pamięci.
        if memory_type == "quantum":
            self.memory = QuantumMemory(
                capacity=memory_capacity,
                forgetting_rate=memory_forgetting_rate,
                decoherence_rate=memory_decoherence_rate,
            )
        elif memory_type == "classical":
            self.memory = ClassicalExponentialMemory(
                capacity=memory_capacity,
                forgetting_rate=memory_forgetting_rate,
            )
        elif memory_type == "none":
            self.memory = None
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

        # Tworzenie modeli.
        self.world_model = WorldModel(learning_rate=learning_rate) if use_world_model else None

        self.self_model = SelfModel(
            initial_capability=0.5,
            initial_uncertainty=0.5,
            learning_rate=learning_rate,
        ) if use_self_model else None

        # Meta observer.
        self.meta_observer = MetaObserver(
            window_size=meta_window_size,
            error_threshold=meta_error_threshold,
        ) if use_meta_observer else None

        # Memory readout oparty na payloadach.
        self.memory_readout = MemoryReadout(
            window=memory_window,
        ) if use_memory_readout else None

        # Quantum memory readout.
        if self.use_quantum_memory_readout:
            self.quantum_readout = QuantumMemoryReadout()
            self.quantum_controller = QuantumMemoryController(
                coherence_explore_threshold=quantum_coherence_threshold,
                uncertainty_recalibrate_threshold=quantum_uncertainty_threshold,
                exploit_success_threshold=quantum_exploit_success_threshold,
                explore_bonus=quantum_explore_bonus,
                recalibrate_penalty=quantum_recalibrate_penalty,
            )
        else:
            self.quantum_readout = None
            self.quantum_controller = None

        # Regime shift gate.
        self.regime_shift_threshold = float(regime_shift_threshold)
        self.regime_shift_duration = int(regime_shift_duration)
        self.regime_shift_explore_bonus = float(regime_shift_explore_bonus)

        if not math.isfinite(self.regime_shift_threshold) or self.regime_shift_threshold < 0.0:
            raise ValueError("regime_shift_threshold must be finite and non-negative.")

        if self.regime_shift_duration < 0:
            raise ValueError("regime_shift_duration must be non-negative.")

        if not math.isfinite(self.regime_shift_explore_bonus) or self.regime_shift_explore_bonus < 0.0:
            raise ValueError("regime_shift_explore_bonus must be finite and non-negative.")

        self._regime_shift_remaining = 0

        # Shadow Planner (EXP-Q21).
        self.use_shadow_planner = bool(kwargs.get("use_shadow_planner", False))
        self.shadow_interference = float(kwargs.get("shadow_interference_weight", 0.5))

        if self.use_shadow_planner:
            self.shadow_planner = ShadowPlanner(
                interference_weight=self.shadow_interference,
            )
        else:
            self.shadow_planner = None

        self.policy = MemoryAwareDecisionPolicy(
            self_model=self.self_model,
            world_model=self.world_model,
            use_self_model=use_self_model,
        )

        # Jeśli meta observer aktywny, strategia może wpływać na decyzje.
        if self.meta_observer is not None:
            self.policy.explore_bonus = explore_bonus
            self.policy.recalibrate_penalty = recalibrate_penalty

        self.candidates = task_candidates(environment)
        self.history: List[Dict[str, Any]] = []
        self.time = 0.0

    def step(self) -> Dict[str, Any]:
        # 0. Regime shift detection.
        last_world_error = 0.0

        if self.history:
            last_world_error = float(
                self.history[-1].get("world_prediction_error", 0.0)
            )

        if (
            last_world_error > self.regime_shift_threshold
            and self.regime_shift_duration > 0
        ):
            self._regime_shift_remaining = max(
                self._regime_shift_remaining,
                self.regime_shift_duration,
            )

        regime_shift_active = self._regime_shift_remaining > 0

        if regime_shift_active:
            self._regime_shift_remaining -= 1

        immediate_surprise = last_world_error > self.regime_shift_threshold

        # 1. Memory readout przed decyzją.
        memory_evidence = None

        if self.memory_readout is not None:
            memory_evidence = self.memory_readout.read(
                self.memory,
                self.time,
            )

            current_memory_weight = self.memory_readout_weight

            # Gating:
            # duży błąd świata wyłącza wpływ pamięci na decyzję.
            if immediate_surprise or regime_shift_active:
                current_memory_weight = 0.0

            self.policy.memory_bias = {
                action: bias * current_memory_weight
                for action, bias in memory_evidence.biases.items()
            }

        else:
            self.policy.memory_bias = {}

        # 2. Quantum memory readout przed decyzją.
        quantum_control = None

        if self.quantum_readout is not None and self.quantum_controller is not None:
            quantum_evidence = self.quantum_readout.read(
                self.memory,
                self.time,
            )

            if quantum_evidence.available:
                quantum_control = self.quantum_controller.control(quantum_evidence)

                self.policy.set_strategy(quantum_control.strategy)
                self.policy.explore_bonus = quantum_control.explore_bonus
                self.policy.recalibrate_penalty = quantum_control.recalibrate_penalty

        # 3. Regime shift override.
        # Jeśli wykryto zmianę reguł, wymuszamy eksplorację przez kilka kroków.
        if regime_shift_active:
            self.policy.set_strategy("explore")

            if self.policy.explore_bonus < self.regime_shift_explore_bonus:
                self.policy.explore_bonus = self.regime_shift_explore_bonus

        # 4. Decyzja.
        if self.shadow_planner is not None:
            # Quantum Decision Theory (QDT) Shadow Planning
            chosen_candidate, scores = self.shadow_planner.choose(
                self.candidates,
                world_model=self.world_model,
                self_model=self.self_model,
            )

            decision = ShadowDecisionResult()
            decision.chosen_action = chosen_candidate.action
            decision.chosen_candidate = chosen_candidate
            decision.scores = scores
            decision.self_model_used = (self.self_model is not None)
            decision.success_factor = self.policy._success_factor() if hasattr(self.policy, "_success_factor") else 0.5
            decision.strategy = self.policy.strategy if hasattr(self.policy, "strategy") else "shadow"
        else:
            decision = self.policy.choose(self.candidates)

        # 5. Predykcja world model przed update.
        world_pred_error = None
        pred_prob = None

        if self.world_model is not None:
            pred_prob, pred_cost = self.world_model.predict(decision.chosen_action)

        # 6. Wykonanie akcji w środowisku.
        outcome = self.environment.step(
            decision.chosen_action,
            timestamp=self.time,
        )

        # 7. Obliczenie błędu world model i update.
        if self.world_model is not None:
            world_pred_error = abs(pred_prob - outcome.performance)

            self.world_model.update(
                decision.chosen_action,
                outcome.performance,
                outcome.cost,
            )

        # 8. Update self model.
        self_model_update = None

        if self.self_model is not None:
            self_model_update = self.self_model.record_outcome(
                outcome.performance,
                timestamp=self.time,
            )

        # 9. Zapis w pamięci.
        if self.memory is not None:
            # Epistemic Coherence:
            # Jeśli agent jest bardzo zaskoczony, zapisuje stan superpozycji.
            if (
                self.use_epistemic_coherence
                and world_pred_error is not None
                and world_pred_error > 0.5
            ):
                state_vector = [1.0, 1.0]
            else:
                state_vector = [1.0, 0.0] if outcome.success else [0.0, 1.0]

            self.memory.store(
                timestamp=self.time,
                payload={
                    "action": decision.chosen_action,
                    "success": outcome.success,
                    "context": getattr(outcome, "context", None),
                },
                state=state_vector,
            )

        # 10. Meta observer.
        meta_state = None

        if self.meta_observer is not None:
            self_error = None

            if self_model_update is not None:
                self_error = self_model_update["error"]

            meta_state = self.meta_observer.update(
                world_prediction_error=world_pred_error,
                self_model_error=self_error,
                success=outcome.success,
                timestamp=self.time,
            )

            self.policy.set_strategy(meta_state["strategy"])

        # 11. Rejestracja.
        record = {
            "time": self.time,
            "chosen_action": decision.chosen_action,
            "success": outcome.success,
            "performance": outcome.performance,
            "cost": outcome.cost,
            "net_reward": outcome.net_reward,
            "memory_type": self.memory_type,
            "decision_strategy": decision.strategy,
            "regime_shift_active": regime_shift_active,
        }

        if world_pred_error is not None:
            record["world_prediction_error"] = world_pred_error

        if hasattr(outcome, "context"):
            record["true_context"] = outcome.context

        if self_model_update is not None:
            record["self_model_error"] = self_model_update["error"]
            record["post_capability"] = self_model_update["capability"]
            record["post_uncertainty"] = self_model_update["uncertainty"]

        if meta_state is not None:
            record["meta_strategy"] = meta_state["strategy"]
            record["meta_model_quality"] = meta_state["model_quality"]
            record["meta_confidence"] = meta_state["confidence"]
            record["meta_mean_world_error"] = meta_state["mean_world_error"]
            record["meta_mean_self_error"] = meta_state["mean_self_error"]

        if memory_evidence is not None:
            record["memory_samples"] = memory_evidence.samples
            record["memory_recent_success_rate"] = memory_evidence.recent_success_rate
            record["memory_bias"] = dict(self.policy.memory_bias)

        if quantum_control is not None:
            record["quantum_strategy"] = quantum_control.strategy
            record["quantum_success_probability"] = quantum_control.success_probability
            record["quantum_coherence"] = quantum_control.coherence
            record["quantum_uncertainty"] = quantum_control.uncertainty

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

    def get_cumulative_reward(self) -> float:
        return sum(r["net_reward"] for r in self.history)

    def get_adaptation_metrics(self) -> Dict[str, float]:
        """
        Oblicza metryki adaptacji.
        """
        if not self.history:
            return {}

        total_cycles = len(self.history)
        total_reward = self.get_cumulative_reward()

        pred_errors = [
            r["world_prediction_error"]
            for r in self.history
            if "world_prediction_error" in r
        ]

        mean_pred_error = 0.0

        if pred_errors:
            mean_pred_error = sum(pred_errors) / float(len(pred_errors))

        return {
            "total_cycles": total_cycles,
            "cumulative_reward": total_reward,
            "mean_prediction_error": mean_pred_error,
            "memory_type": self.memory_type,
        }