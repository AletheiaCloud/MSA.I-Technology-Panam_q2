import pytest

from panam_q.decision.decision_policy import ActionCandidate
from panam_q.decision.memory_policy import MemoryAwareDecisionPolicy
from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop
from panam_q.memory.memory_readout import MemoryReadout
from panam_q.memory.quantum_memory import QuantumMemory


def make_simple_candidates():
    return [
        ActionCandidate(
            action="easy",
            base_value=0.6,
            cost=0.0,
            difficulty=0.0,
        ),
        ActionCandidate(
            action="hard",
            base_value=1.0,
            cost=0.0,
            difficulty=1.0,
        ),
    ]


def test_memory_readout_empty_memory():
    memory = QuantumMemory(capacity=10)
    readout = MemoryReadout(window=10)

    evidence = readout.read(memory, current_time=0.0)

    assert evidence.samples == 0
    assert evidence.recent_success_rate == 0.0
    assert evidence.biases == {}
    assert evidence.uncertainty == 1.0


def test_memory_readout_action_success_rates():
    memory = QuantumMemory(capacity=10)
    readout = MemoryReadout(window=10)

    memory.store(
        timestamp=0.0,
        payload={"action": "hard", "success": True},
        state=[1.0, 0.0],
    )

    memory.store(
        timestamp=1.0,
        payload={"action": "hard", "success": False},
        state=[0.0, 1.0],
    )

    memory.store(
        timestamp=2.0,
        payload={"action": "easy", "success": True},
        state=[1.0, 0.0],
    )

    evidence = readout.read(memory, current_time=3.0)

    assert evidence.samples == 3
    assert evidence.action_samples["hard"] == 2
    assert evidence.action_samples["easy"] == 1
    assert evidence.action_success_rates["hard"] == pytest.approx(0.5)
    assert evidence.action_success_rates["easy"] == pytest.approx(1.0)
    assert evidence.recent_success_rate == pytest.approx(2.0 / 3.0)


def test_memory_readout_uncertainty_decreases_with_samples():
    memory_small = QuantumMemory(capacity=10)
    memory_large = QuantumMemory(capacity=10)

    readout = MemoryReadout(window=10)

    memory_small.store(
        timestamp=0.0,
        payload={"action": "easy", "success": True},
        state=[1.0, 0.0],
    )

    for i in range(5):
        memory_large.store(
            timestamp=float(i),
            payload={"action": "easy", "success": True},
            state=[1.0, 0.0],
        )

    evidence_small = readout.read(memory_small, current_time=1.0)
    evidence_large = readout.read(memory_large, current_time=5.0)

    assert evidence_large.uncertainty < evidence_small.uncertainty


def test_memory_policy_bias_changes_choice():
    candidates = make_simple_candidates()

    policy_no_bias = MemoryAwareDecisionPolicy(
        self_model=None,
        world_model=None,
        use_self_model=False,
        default_success_factor=0.5,
    )

    policy_with_bias = MemoryAwareDecisionPolicy(
        self_model=None,
        world_model=None,
        use_self_model=False,
        default_success_factor=0.5,
        memory_bias={"hard": 0.3},
    )

    result_no_bias = policy_no_bias.choose(candidates)
    result_with_bias = policy_with_bias.choose(candidates)

    # Bez bias:
    # easy score = 0.6
    # hard score = 1.0 * 0.5 = 0.5
    assert result_no_bias.chosen_action == "easy"

    # Z bias:
    # hard score = 0.5 + 0.3 = 0.8
    assert result_with_bias.chosen_action == "hard"


def test_loop_memory_readout_disabled_by_default():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
    )

    record = loop.step()

    assert "memory_samples" not in record
    assert loop.policy.memory_bias == {}


def test_loop_memory_readout_enabled_can_change_choice():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
        easy_cost=0.0,
        hard_cost=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=True,
        memory_readout_weight=0.5,
    )

    # Prefill memory: hard miał same sukcesy.
    for i in range(5):
        loop.memory.store(
            timestamp=float(i),
            payload={"action": "hard", "success": True},
            state=[1.0, 0.0],
        )

    record = loop.step()

    assert record["memory_samples"] == 5
    assert record["chosen_action"] == "hard"
    assert record["memory_bias"]["hard"] > 0.0


def test_loop_memory_readout_records_evidence():
    environment = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
    )

    loop = MemoryAugmentedLoop(
        environment=environment,
        memory_type="quantum",
        use_self_model=False,
        use_world_model=False,
        use_meta_observer=False,
        use_memory_readout=True,
        memory_readout_weight=0.2,
    )

    records = loop.run(2)

    # Pierwszy krok: pamięć pusta.
    assert records[0]["memory_samples"] == 0

    # Drugi krok: pamięć zawiera już pierwszy rekord.
    assert records[1]["memory_samples"] >= 1
    assert "memory_bias" in records[1]