from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Optional


def _validate_unit_interval(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")

    return v


def _validate_positive(name: str, value: float) -> float:
    v = float(value)

    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")

    return v


def _validate_positive_int(name: str, value: int) -> int:
    v = int(value)

    if v <= 0:
        raise ValueError(f"{name} must be positive.")

    return v


def _clip01(value: float) -> float:
    v = float(value)

    if not math.isfinite(v):
        raise ValueError("Value must be finite.")

    if v < 0.0:
        return 0.0

    if v > 1.0:
        return 1.0

    return v


class MetaObserver:
    """
    Minimalny meta-observer.

    To nie jest świadomość.
    To nie jest magiczny obserwator.

    To jest funkcjonalny moduł monitorujący jakość modeli:

    - world_prediction_error
    - self_model_error

    Na ich podstawie szacuje:

    - model_quality
    - confidence
    - strategy recommendation
    """

    def __init__(
        self,
        window_size: int = 20,
        error_threshold: float = 0.3,
        confidence_k: float = 10.0,
    ):
        self.window_size = _validate_positive_int("window_size", window_size)
        self.error_threshold = _validate_unit_interval("error_threshold", error_threshold)
        self.confidence_k = _validate_positive("confidence_k", confidence_k)

        self._records = deque(maxlen=self.window_size)

        self.samples = 0
        self.model_quality = 0.5
        self.confidence = 0.0
        self.strategy = "continue"

    def update(
        self,
        world_prediction_error: Optional[float] = None,
        self_model_error: Optional[float] = None,
        success: Optional[bool] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aktualizuje meta-observer na podstawie nowych błędów modeli.
        """
        if timestamp is None:
            timestamp = float(self.samples)
        else:
            t = float(timestamp)

            if not math.isfinite(t) or t < 0.0:
                raise ValueError("timestamp must be finite and non-negative.")

            timestamp = t

        world_error = None
        self_error = None

        if world_prediction_error is not None:
            world_error = _clip01(world_prediction_error)

        if self_model_error is not None:
            self_error = _clip01(self_model_error)

        record = {
            "timestamp": timestamp,
            "world_prediction_error": world_error,
            "self_model_error": self_error,
            "success": success,
        }

        self._records.append(record)
        self.samples += 1

        world_errors = [
            r["world_prediction_error"]
            for r in self._records
            if r["world_prediction_error"] is not None
        ]

        self_errors = [
            r["self_model_error"]
            for r in self._records
            if r["self_model_error"] is not None
        ]

        mean_world_error = 0.0
        mean_self_error = 0.0

        if world_errors:
            mean_world_error = sum(world_errors) / float(len(world_errors))

        if self_errors:
            mean_self_error = sum(self_errors) / float(len(self_errors))

        # Jeśli są jakiekolwiek dane o błędach, aktualizuj jakość modelu.
        if world_errors or self_errors:
            if world_errors and self_errors:
                combined_error = 0.5 * mean_world_error + 0.5 * mean_self_error
            elif world_errors:
                combined_error = mean_world_error
            else:
                combined_error = mean_self_error

            instant_quality = 1.0 - combined_error

            alpha = 0.2

            self.model_quality = (
                (1.0 - alpha) * self.model_quality
                + alpha * instant_quality
            )

            self.model_quality = _clip01(self.model_quality)

        # Confidence rośnie wraz z liczbą próbek i jakością modelu.
        self.confidence = (
            float(self.samples) / (float(self.samples) + self.confidence_k)
        ) * self.model_quality

        self.confidence = _clip01(self.confidence)

        # Rekomendacja strategii.
        if mean_self_error > self.error_threshold:
            self.strategy = "recalibrate"
        elif mean_world_error > self.error_threshold:
            self.strategy = "explore"
        elif self.model_quality >= 0.7 and self.confidence >= 0.5:
            self.strategy = "exploit"
        else:
            self.strategy = "continue"

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """
        Zwraca aktualny snapshot meta-observera.
        """
        world_errors = [
            r["world_prediction_error"]
            for r in self._records
            if r["world_prediction_error"] is not None
        ]

        self_errors = [
            r["self_model_error"]
            for r in self._records
            if r["self_model_error"] is not None
        ]

        mean_world_error = 0.0
        mean_self_error = 0.0

        if world_errors:
            mean_world_error = sum(world_errors) / float(len(world_errors))

        if self_errors:
            mean_self_error = sum(self_errors) / float(len(self_errors))

        return {
            "samples": self.samples,
            "window_size": len(self._records),
            "model_quality": self.model_quality,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "mean_world_error": mean_world_error,
            "mean_self_error": mean_self_error,
        }