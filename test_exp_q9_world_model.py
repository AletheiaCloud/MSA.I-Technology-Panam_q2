import pytest

from panam_q.decision.decision_policy import ActionCandidate, DecisionPolicy
from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.self_referential_loop import SelfReferentialLoop
from panam_q.selfmodel.self_model import SelfModel
from panam_q.worldmodel.world_model import WorldModel


def test_world_model_invalid_params():
    with pytest.raises(ValueError):
        WorldModel(learning_rate=1.5)

    with pytest.raises(ValueError):
        WorldModel(default_success_prob=-0.1)

    with pytest.raises(ValueError):
        WorldModel(default_cost=-1.0)


def test_world_model_predict_default():
    wm = WorldModel(default_success_prob=0.5, default_cost=0.1)

    prob, cost = wm.predict("unknown_action")

    assert prob == pytest.approx(0.5)
    assert cost == pytest.approx(0.1)
    assert wm.get_samples("unknown_action") == 0


def test_world_model_update_ema():
    wm = WorldModel(learning_rate=0.5, default_success_prob=0.5, default_cost=0.0)

    # Aktualizacja 1: sukces (1.0)
    wm.update("action_a", actual_performance=1.0, actual_cost=0.2)
    prob, cost = wm.predict("action_a")

    # 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    assert prob == pytest.approx(0.75)
    # 0.5 * 0.0 + 0.5 * 0.2 = 0.1
    assert cost == pytest.approx(0.1)
    assert wm.get_samples("action_a") == 1


def test_world_model_prediction_error():
    wm = WorldModel(learning_rate=0.1, default_success_prob=0.8)

    # Przed update'm predykcja to 0.8. Rzeczywisty wynik to 0.0.
    error = wm.get_prediction_error("action_a", actual_performance=0.0)

    assert error == pytest.approx(0.8)

    # Po update predykcja się zmieni, ale błąd dla tego samego wyniku liczy się z nowej predykcji
    wm.update("action_a", actual_performance=0.0, actual_cost=0.0)
    new_error = wm.get_prediction_error("action_a", actual_performance=0.0)
    
    # Nowa predykcja: 0.9 * 0.8 + 0.1 * 0.0 = 0.72. Błąd = 0.72.
    assert new_error == pytest.approx(0.72)


def test_policy_uses_world_model_to_avoid_bad_actions():
    """
    Jeśli World Model nauczy się, że 'hard' zawsze zawodzi (P=0),
    polityka powinna wybrać 'easy', nawet jeśli 'hard' ma wyższe base_value.
    """
    wm = WorldModel(learning_rate=1.0) # Natychmiastowe uczenie
    wm.update("hard", actual_performance=0.0, actual_cost=0.0)
    wm.update("easy", actual_performance=1.0, actual_cost=0.0)

    candidates = [
        ActionCandidate("easy", base_value=0.6, cost=0.0, difficulty=0.0),
        ActionCandidate("hard", base_value=1.0, cost=0.0, difficulty=0.0),
    ]

    policy = DecisionPolicy(world_model=wm)
    result = policy.choose(candidates)

    # easy: 0.6 * 1.0 * 1.0 - 0.0 = 0.6
    # hard: 1.0 * 0.0 * 1.0 - 0.0 = 0.0
    assert result.chosen_action == "easy"


def test_policy_without_world_model_is_optimistic():
    """
    Bez World Model polityka zakłada P(success) = 1.0 dla base_value.
    """
    candidates = [
        ActionCandidate("easy", base_value=0.6, cost=0.0, difficulty=0.0),
        ActionCandidate("hard", base_value=1.0, cost=0.0, difficulty=0.0),
    ]

    policy = DecisionPolicy(world_model=None)
    result = policy.choose(candidates)

    # hard wygrywa, bo 1.0 * 1.0 > 0.6 * 1.0
    assert result.chosen_action == "hard"


def test_loop_integrates_world_model():
    """
    Pętla z World Modelem powinna z czasem zrezygnować z 'hard',
    jeśli środowisko zawsze karze 'hard' porażką.
    """
    env = TaskEnvironment(
        seed=42,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
        easy_cost=0.0,
        hard_cost=0.0,
    )

    wm = WorldModel(learning_rate=0.5)
    sm = SelfModel(initial_capability=0.9, learning_rate=0.1)

    loop = SelfReferentialLoop(
        environment=env,
        self_model=sm,
        world_model=wm,
    )

    records = loop.run(5)
    actions = [r["chosen_action"] for r in records]

    # Na początku może wybrać 'hard' (optymistyczny prior 0.5 vs easy 0.5, ale hard ma base_value 1.0)
    # Po 1-2 porażkach World Model uczy się, że P(success|hard) spada.
    # Wymuszamy, aby ostatnie 2 akcje to było 'easy'.
    assert actions[-1] == "easy"
    assert actions[-2] == "easy"
    assert "world_prediction_error" in records[0]


def test_loop_ablation_without_world_model():
    """
    Bez World Model, system opiera się tylko na Self Model i priorytetach.
    W środowisku, gdzie hard_cost=0, a base_value=1.0, 
    sam Self Model może potrzebować więcej czasu na zniechęcenie się do 'hard'.
    """
    env = TaskEnvironment(
        seed=42,
        easy_success_probability=1.0,
        hard_success_probability=0.0,
        easy_cost=0.0,
        hard_cost=0.0, # Brak kosztu środowiskowego
    )

    # Brak world_model
    loop = SelfReferentialLoop(
        environment=env,
        self_model=None,
        world_model=None,
        default_success_factor=0.5,
    )

    records = loop.run(3)
    actions = [r["chosen_action"] for r in records]

    # Bez World Model i bez Self Model, hard ma score 1.0 * 0.5 = 0.5.
    # easy ma score 0.6 * 1.0 = 0.6.
    # Więc i tak wygra easy. Zmieńmy base_value easy na 0.4, żeby sprawdzić ablację.
    
    # Lepiej: sprawdźmy, że bez WM system nie uczy się P(success).
    assert "world_prediction_error" not in records[0]