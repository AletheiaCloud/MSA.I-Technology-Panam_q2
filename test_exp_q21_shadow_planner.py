import pytest

from panam_q.decision.decision_policy import ActionCandidate
from panam_q.env.task_environment import TaskEnvironment
from panam_q.loop.memory_augmented_loop import MemoryAugmentedLoop
from panam_q.planner.shadow_planner import ShadowPlanner
from panam_q.selfmodel.self_model import SelfModel
from panam_q.worldmodel.world_model import WorldModel


def make_candidates():
    return [
        ActionCandidate("safe", base_value=0.6, cost=0.0, difficulty=0.0),
        ActionCandidate("risky", base_value=1.0, cost=0.0, difficulty=1.0),
    ]


def test_shadow_planner_invalid_params():
    with pytest.raises(ValueError):
        ShadowPlanner(interference_weight=-0.1)


def test_shadow_planner_matches_classical_when_zero_interference():
    """
    Gdy interference_weight = 0, Shadow Planner powinien zachowywać się 
    identycznie jak klasyczna polityka (tylko Utility).
    """
    planner = ShadowPlanner(interference_weight=0.0)
    
    # Bez world_model i self_model, utility dla 'risky' = 1.0 * 1.0 = 1.0
    # utility dla 'safe' = 0.6 * 1.0 = 0.6
    chosen, scores = planner.choose(make_candidates())
    
    assert chosen.action == "risky"


def test_destructive_interference_penalizes_uncertain_hard_actions():
    """
    GŁÓWNY TEST QDT:
    Jeśli agent ma wysoką niepewność (np. wysoki self_model.uncertainty),
    trudna akcja ('risky') powinna otrzymać ujemny interference term (q),
    co może sprawić, że 'safe' stanie się lepszym wyborem.
    """
    planner = ShadowPlanner(interference_weight=1.5)
    
    # Agent jest bardzo niepewny swoich możliwości
    self_model = SelfModel(initial_capability=0.5, initial_uncertainty=0.9)
    
    chosen, scores = planner.choose(
        make_candidates(),
        self_model=self_model,
    )
    
    # Utility: safe=0.6, risky=1.0
    # Interference: safe=0 (difficulty=0), risky = -1.5 * 0.9 * 1.0 = -1.35
    # Total: safe=0.6, risky = -0.35
    # Po przesunięciu (safe_score), safe wygrywa.
    assert chosen.action == "safe"


def test_constructive_interference_ignores_uncertainty_for_easy_actions():
    """
    Jeśli akcja jest łatwa (difficulty=0), interferencja wynosi 0,
    niezależnie od tego, jak bardzo agent jest niepewny.
    """
    planner = ShadowPlanner(interference_weight=10.0) # Ekstremalna kara
    
    self_model = SelfModel(initial_capability=0.5, initial_uncertainty=1.0)
    
    candidates = [
        ActionCandidate("easy_A", base_value=0.6, cost=0.0, difficulty=0.0),
        ActionCandidate("easy_B", base_value=0.4, cost=0.0, difficulty=0.0),
    ]
    
    chosen, scores = planner.choose(candidates, self_model=self_model)
    
    # Żadna akcja nie jest trudna, więc nie ma kary. Wygrywa wyższa wartość.
    assert chosen.action == "easy_A"


def test_world_model_samples_reduce_interference():
    """
    Jeśli World Model ma dużo danych o akcji (wysokie samples),
    niepewność specyficzna dla akcji spada, więc interferencja maleje.
    """
    planner = ShadowPlanner(interference_weight=2.0)
    
    world_model = WorldModel(default_success_prob=0.5, learning_rate=0.5)
    
    # Uczymy world model, że 'risky' zawsze wygrywa (dużo próbek)
    for _ in range(10):
        world_model.update("risky", actual_performance=1.0, actual_cost=0.0)
        
    # 'safe' ma 0 próbek (wysoka niepewność)
    
    candidates = [
        ActionCandidate("safe", base_value=0.9, cost=0.0, difficulty=0.5),
        ActionCandidate("risky", base_value=0.9, cost=0.0, difficulty=0.5),
    ]
    
    chosen, scores = planner.choose(candidates, world_model=world_model)
    
    # Obie akcje mają tę samą wartość i trudność.
    # Ale 'safe' ma mało próbek -> wysoka niepewność -> duża kara (interference).
    # 'risky' ma dużo próbek -> niska niepewność -> mała kara.
    assert chosen.action == "risky"


def test_loop_integration_shadow_planner_changes_choice():
    """
    Test integracji z MemoryAugmentedLoop.
    Włączenie Shadow Plannera powinno zmienić decyzję w warunkach wysokiej niepewności.
    """
    env = TaskEnvironment(
        seed=1,
        easy_success_probability=1.0,
        hard_success_probability=0.0, # Środowisko karze 'hard'
    )
    
    # Pętla z Shadow Plannerem i wysoką początkową niepewnością
    loop = MemoryAugmentedLoop(
        environment=env,
        memory_type="none",
        use_self_model=True,
        use_world_model=False, # Wyłączamy WM, żeby wymusić użycie default_uncertainty
        use_meta_observer=False,
        use_shadow_planner=True,
        shadow_interference_weight=1.5,
        learning_rate=0.1,
    )
    
    # Ustawiamy ręcznie wysoką niepewność w self_model
    loop.self_model.uncertainty = 0.9
    
    record = loop.step()
    
    # Bez shadow plannera, default_success_factor=0.5 sprawiłby, że:
    # easy = 0.6 * 1.0 = 0.6
    # hard = 1.0 * 0.5 = 0.5 -> easy wygrywa
    # Ale z Shadow Plannerem (bez WM, default_prob=1.0):
    # utility: easy=0.6, hard=1.0
    # interference: easy=0, hard = -1.5 * 0.9 * 1.0 = -1.35
    # hard dostaje potężną karę za trudność + niepewność.
    assert record["chosen_action"] == "easy"