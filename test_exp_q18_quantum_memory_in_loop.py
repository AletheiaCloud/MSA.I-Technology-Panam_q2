import pytest

from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop
from panam_q.memory.classical_memory import ClassicalExponentialMemory
from panam_q.memory.quantum_memory import QuantumMemory
from panam_q.memory.quantum_memory_controller import (
    QuantumMemoryControlSignal,
    QuantumMemoryController,
)
from panam_q.memory.quantum_memory_readout import QuantumMemoryEvidence


def make_evidence(
    available: bool = True,
    success_probability: float = 0.5,
    failure_probability: float = 0.5,
    coherence: float = 0.0,
    purity: float = 1.0,
    entropy: float = 0.0,
    uncertainty: float = 0.0,
) -> QuantumMemoryEvidence:
    return QuantumMemoryEvidence(
        available=available,
        success_probability=success_probability,
        failure_probability=failure_probability,
        coherence=coherence,
        purity=purity,
        entropy=entropy,
        uncertainty=uncertainty,
    )


def make_environment():
    return TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
        easy_cost=0.0,
        hard_cost=0.0,
    )


def test_controller_invalid_params():
    with pytest.raises(ValueError):
        QuantumMemoryController(coherence_explore_threshold=1.2)

    with pytest.raises(ValueError):
        QuantumMemoryController(uncertainty_recalibrate_threshold=-0.1)

    with pytest.raises(ValueError):
        QuantumMemoryController(exploit_success_threshold=2.0)

    with pytest.raises(ValueError):
        QuantumMemoryController(explore_bonus=-0.1)

    with pytest.raises(ValueError):
        QuantumMemoryController(recalibrate_penalty=-0.1)


def test_controller_unavailable_evidence():
    controller = QuantumMemoryController()

    evidence = make_evidence(available=False)
    signal = controller.control(evidence)

    assert isinstance(signal, QuantumMemoryControlSignal)
    assert signal.strategy == "continue"
    assert signal.explore_bonus == 0.0
    assert signal.recalibrate_penalty == 0.0


def test_controller_high_coherence_leads_to_explore():
    controller = QuantumMemoryController(
        coherence_explore_threshold=0.3,
        explore_bonus=0.25,
    )

    evidence = make_evidence(
        coherence=0.8,
        uncertainty=0.1,
    )

    signal = controller.control(evidence)

    assert signal.strategy == "explore"
    assert signal.explore_bonus == pytest.approx(0.25)
    assert signal.recalibrate_penalty == 0.0


def test_controller_high_uncertainty_leads_to_recalibrate():
    controller = QuantumMemoryController(
        coherence_explore_threshold=0.3,
        uncertainty_recalibrate_threshold=0.7,
        recalibrate_penalty=0.3,
    )

    evidence = make_evidence(
        coherence=0.0,
        uncertainty=0.9,
    )

    signal = controller.control(evidence)

    assert signal.strategy == "recalibrate"
    assert signal.explore_bonus == 0.0
    assert signal.recalibrate_penalty == pytest.approx(0.3)


def test_controller_high_success_leads_to_exploit():
    controller = QuantumMemoryController(
        coherence_explore_threshold=0.3,
        uncertainty_recalibrate_threshold=0.7,
        exploit_success_threshold=0.6,
    )

    evidence = make_evidence(
        success_probability=0.8,
        coherence=0.0,
        uncertainty=0.1,
    )

    signal = controller.control(evidence)

    assert signal.strategy == "exploit"


def test_loop_quantum_readout_disabled_by_default():
    environment = make_environment()

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=False,
        use_quantum_memory_readout=False,
    )

    record = loop.step()

    assert "quantum_strategy" not in record


def test_loop_quantum_coherent_memory_leads_to_explore_and_hard_choice():
    environment = make_environment()

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=False,
        use_quantum_memory_readout=True,
        quantum_coherence_threshold=0.3,
        quantum_explore_bonus=0.3,
    )

    # Stan koherentny: superpozycja success/failure.
    loop.memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=[1.0, 1.0],
    )

    record = loop.step()

    assert record["quantum_strategy"] == "explore"
    assert record["quantum_coherence"] > 0.9

    # Bez explore:
    # easy = 0.6
    # hard = 0.5
    #
    # Z explore bonus 0.3:
    # hard = 0.8
    assert record["chosen_action"] == "hard"


def test_loop_classical_mixed_memory_leads_to_recalibrate_and_easy_choice():
    environment = make_environment()

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="classical",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=False,
        use_quantum_memory_readout=True,
        quantum_uncertainty_threshold=0.7,
        quantum_recalibrate_penalty=0.3,
    )

    # Classical memory nie przechowuje koherencji.
    # Ten stan zostanie zredukowany do mieszanki diagonalnej.
    loop.memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=[1.0, 1.0],
    )

    record = loop.step()

    assert record["quantum_strategy"] == "recalibrate"
    assert record["quantum_uncertainty"] > 0.9
    assert record["quantum_coherence"] == pytest.approx(0.0, abs=1e-10)

    # Bez recalibrate:
    # easy = 0.6
    # hard = 0.5
    #
    # Z recalibrate penalty 0.3:
    # hard = 0.2
    assert record["chosen_action"] == "easy"


def test_quantum_and_classical_same_state_produce_different_control():
    environment_q = make_environment()
    environment_c = make_environment()

    loop_q = MemoryAugmentedLoop(
        environment=environment_q,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=False,
        use_quantum_memory_readout=True,
    )

    loop_c = MemoryAugmentedLoop(
        environment=environment_c,
        memory_type="classical",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=False,
        use_quantum_memory_readout=True,
    )

    # Ten sam stan wejściowy.
    loop_q.memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=[1.0, 1.0],
    )

    loop_c.memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": None},
        state=[1.0, 1.0],
    )

    record_q = loop_q.step()
    record_c = loop_c.step()

    assert record_q["quantum_strategy"] == "explore"
    assert record_c["quantum_strategy"] == "recalibrate"
    assert record_q["quantum_coherence"] > record_c["quantum_coherence"]