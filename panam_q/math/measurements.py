from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .density_matrix import DensityMatrix
from .distances import bures_distance
from .entropy import von_neumann_entropy


@dataclass
class MeasurementResult:
    """
    Wynik pomiaru.

    outcome:
        None -> pomiar nieselective / bez wyboru wyniku
        int  -> pomiar selective z wybranym wynikiem

    probability:
        prawdopodobieństwo zwróconego stanu / wyniku

    post_state:
        stan po pomiarze

    information_gain:
        oczekiwana redukcja entropii stanu

    disturbance:
        odległość Buresa między stanem sprzed pomiaru a stanem po pomiarze
    """

    outcome: Optional[int]
    probability: float
    post_state: DensityMatrix
    probabilities: Dict[int, float] = field(default_factory=dict)
    information_gain: float = 0.0
    disturbance: float = 0.0


def _as_density_matrix(state) -> DensityMatrix:
    if isinstance(state, DensityMatrix):
        return state.copy()

    return DensityMatrix(state)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2.0


def _clip_probability(value: float, tol: float = 1e-12) -> float:
    if value < 0.0:
        if value > -tol:
            return 0.0

        raise ValueError("Negative probability detected.")

    if value > 1.0:
        if value < 1.0 + tol:
            return 1.0

        raise ValueError("Probability greater than 1 detected.")

    return float(value)


def _make_post_state(matrix: np.ndarray, tol: float = 1e-12) -> DensityMatrix:
    matrix = _symmetrize(matrix)
    state = DensityMatrix(matrix)

    trace = np.trace(state.data)

    if abs(trace - 1.0) > tol and abs(trace) > tol:
        state = state.normalized()

    return state


def computational_projectors(dim: int):
    """
    Projektory pomiaru w bazie obliczeniowej:

    P_i = |i><i|
    """
    projectors = []

    for index in range(dim):
        projector = np.zeros((dim, dim), dtype=np.complex128)
        projector[index, index] = 1.0
        projectors.append(projector)

    return projectors


def projective_measurement(
    state,
    outcome: Optional[int] = None,
    tol: float = 1e-12,
) -> MeasurementResult:
    """
    Pomiar projekcyjny w bazie obliczeniowej.

    Jeśli outcome is None:
        zwraca stan po pomiarze nieselective:
        rho' = Σ_i P_i rho P_i

    Jeśli outcome is int:
        zwraca stan po pomiarze selective:
        rho'_i = P_i rho P_i / p_i
    """
    rho = _as_density_matrix(state)
    dim = rho.dim
    projectors = computational_projectors(dim)

    probabilities = []
    selective_posts = []
    unnormalized_posts = []

    for projector in projectors:
        unnormalized = projector @ rho.data @ projector.conj().T
        probability = _clip_probability(
            float(np.real(np.trace(unnormalized))),
            tol,
        )

        probabilities.append(probability)
        unnormalized_posts.append(unnormalized)

        if probability > tol:
            selective_posts.append(unnormalized / probability)
        else:
            selective_posts.append(np.zeros_like(rho.data))

    total_probability = float(sum(probabilities))

    if total_probability <= tol:
        raise ValueError("Measurement probabilities sum to zero.")

    if abs(total_probability - 1.0) > 1e-8:
        probabilities = [p / total_probability for p in probabilities]

    if outcome is None:
        post_matrix = np.zeros_like(rho.data)

        for projector in projectors:
            post_matrix += projector @ rho.data @ projector.conj().T

        post_state = _make_post_state(post_matrix, tol)
        selected_outcome = None
        selected_probability = 1.0

    else:
        if outcome < 0 or outcome >= dim:
            raise ValueError("Measurement outcome out of range.")

        probability = probabilities[outcome]

        if probability <= tol:
            raise ValueError("Measurement outcome has zero probability.")

        post_state = _make_post_state(selective_posts[outcome], tol)
        selected_outcome = outcome
        selected_probability = probability

    pre_entropy = von_neumann_entropy(rho, tol)

    expected_post_entropy = 0.0

    for index, probability in enumerate(probabilities):
        if probability > tol:
            expected_post_entropy += probability * von_neumann_entropy(
                DensityMatrix(selective_posts[index]),
                tol,
            )

    information_gain = max(0.0, pre_entropy - expected_post_entropy)
    disturbance = bures_distance(rho, post_state, tol)

    return MeasurementResult(
        outcome=selected_outcome,
        probability=selected_probability,
        post_state=post_state,
        probabilities={i: probabilities[i] for i in range(dim)},
        information_gain=information_gain,
        disturbance=disturbance,
    )


def _ancilla_unitary_qubit(strength: float) -> np.ndarray:
    """
    Unitarna operacja sprzężenia systemu z ancilla.

    Zwrócona unitarność implementuje słaby pomiar w bazie obliczeniowej.
    Po pomiarze ancilla daje efektywnie operatory Krausa:

    K_0 = diag(sqrt(1+s), sqrt(1-s)) / sqrt(2)
    K_1 = diag(sqrt(1-s), sqrt(1+s)) / sqrt(2)

    gdzie s = strength.

    s = 0  -> brak pomiaru
    s = 1  -> pomiar projekcyjny w bazie Z
    """
    if strength < 0.0 or strength > 1.0:
        raise ValueError("Measurement strength must be in [0, 1].")

    a = np.sqrt(1.0 + strength)
    b = np.sqrt(1.0 - strength)

    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    unitary = np.zeros((4, 4), dtype=np.complex128)

    # Podprzestrzeń systemu |0>:
    # indeksy: |0,0> = 0, |0,1> = 1
    unitary[0, 0] = a * inv_sqrt2
    unitary[1, 0] = b * inv_sqrt2
    unitary[0, 1] = b * inv_sqrt2
    unitary[1, 1] = -a * inv_sqrt2

    # Podprzestrzeń systemu |1>:
    # indeksy: |1,0> = 2, |1,1> = 3
    unitary[2, 2] = b * inv_sqrt2
    unitary[3, 2] = a * inv_sqrt2
    unitary[2, 3] = a * inv_sqrt2
    unitary[3, 3] = -b * inv_sqrt2

    return unitary

def _ancilla_projector(outcome: int, system_dim: int = 2) -> np.ndarray:
    """
    Projektor na wynik pomiaru ancilla.

    outcome = 0 -> ancilla w |0>
    outcome = 1 -> ancilla w |1>
    """
    dimension = system_dim * 2

    projector = np.zeros((dimension, dimension), dtype=np.complex128)

    for system_index in range(system_dim):
        index = system_index * 2 + outcome
        projector[index, index] = 1.0

    return projector


def _partial_trace_ancilla(matrix: np.ndarray) -> np.ndarray:
    """
    Ślad częściowy po ancilla.

    Zakłada porządek indeksów:
    (system, ancilla, system', ancilla')
    """
    reshaped = matrix.reshape(2, 2, 2, 2)

    return np.trace(reshaped, axis1=1, axis2=3)


def weak_measurement_qubit(
    state,
    strength: float,
    outcome: Optional[int] = None,
    tol: float = 1e-12,
) -> MeasurementResult:
    """
    Słaby pomiar qubitu przez ancilla.

    strength = 0 -> brak pomiaru
    strength = 1 -> pełny pomiar projekcyjny w bazie obliczeniowej

    Jeśli outcome is None:
        pomiar nieselective: suma po wynikach ancilla.

    Jeśli outcome is 0 or 1:
        pomiar selective z konkretnym wynikiem ancilla.
    """
    rho = _as_density_matrix(state)

    if rho.dim != 2:
        raise ValueError("Weak measurement currently supports qubits only.")

    unitary = _ancilla_unitary_qubit(strength)

    ancilla_state = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.complex128,
    )

    joint_state = np.kron(rho.data, ancilla_state)
    joint_state = unitary @ joint_state @ unitary.conj().T

    probabilities = []
    selective_posts = []
    unnormalized_systems = []

    for measurement_outcome in (0, 1):
        projector = _ancilla_projector(measurement_outcome, system_dim=2)

        unnormalized_joint = (
            projector
            @ joint_state
            @ projector.conj().T
        )

        probability = _clip_probability(
            float(np.real(np.trace(unnormalized_joint))),
            tol,
        )

        unnormalized_system = _partial_trace_ancilla(unnormalized_joint)

        probabilities.append(probability)
        unnormalized_systems.append(unnormalized_system)

        if probability > tol:
            selective_posts.append(unnormalized_system / probability)
        else:
            selective_posts.append(np.zeros((2, 2), dtype=np.complex128))

    total_probability = float(sum(probabilities))

    if total_probability <= tol:
        raise ValueError("Measurement probabilities sum to zero.")

    if abs(total_probability - 1.0) > 1e-8:
        probabilities = [p / total_probability for p in probabilities]

    if outcome is None:
        post_matrix = np.zeros((2, 2), dtype=np.complex128)

        for unnormalized_system in unnormalized_systems:
            post_matrix += unnormalized_system

        post_state = _make_post_state(post_matrix, tol)
        selected_outcome = None
        selected_probability = 1.0

    else:
        if outcome not in (0, 1):
            raise ValueError("Qubit weak measurement outcome must be 0 or 1.")

        probability = probabilities[outcome]

        if probability <= tol:
            raise ValueError("Measurement outcome has zero probability.")

        post_state = _make_post_state(selective_posts[outcome], tol)
        selected_outcome = outcome
        selected_probability = probability

    pre_entropy = von_neumann_entropy(rho, tol)

    expected_post_entropy = 0.0

    for index, probability in enumerate(probabilities):
        if probability > tol:
            expected_post_entropy += probability * von_neumann_entropy(
                DensityMatrix(selective_posts[index]),
                tol,
            )

    information_gain = max(0.0, pre_entropy - expected_post_entropy)
    disturbance = bures_distance(rho, post_state, tol)

    return MeasurementResult(
        outcome=selected_outcome,
        probability=selected_probability,
        post_state=post_state,
        probabilities={i: probabilities[i] for i in range(2)},
        information_gain=information_gain,
        disturbance=disturbance,
    )