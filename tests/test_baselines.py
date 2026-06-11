"""
tests/test_baselines.py

Tests para GreedySelector, KnapsackSelector y RolloutSelector.

Ejecutar con:
    pytest tests/test_baselines.py -v
"""

import numpy as np
import pytest

from src.environment.problem import CANONICAL_TOPICS, Problem
from src.environment.student_model import StudentModel
from src.baselines.selector import ProblemSelector, SessionState
from src.baselines.greedy import GreedySelector
from src.baselines.knapsack import KnapsackSelector
from src.baselines.rollout import RolloutSelector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_problems(n: int = 8) -> list[Problem]:
    ratings = [800, 900, 1000, 1200, 1400, 1600, 1800, 2000]
    tags    = [
        ["math"], ["greedy"], ["dp"], ["graphs"],
        ["math", "dp"], ["greedy", "graphs"],
        ["dp", "trees"], ["math", "number theory"],
    ]
    return [
        Problem(
            problem_id      = f"{i+1}A",
            name            = f"Problem {i+1}",
            rating          = ratings[i % len(ratings)],
            tags_list       = tags[i % len(tags)],
            statement       = f"Statement {i}" * 5,
            difficulty_band = "800-1199",
        )
        for i in range(n)
    ]


def make_student(budget: float = 120.0) -> StudentModel:
    return StudentModel(
        topic_ratings      = {t: 1400 for t in CANONICAL_TOPICS},
        session_budget_min = budget,
        random_seed        = 42,
    )


def make_state(
    n_problems : int   = 8,
    budget     : float = 120.0,
    mask       : np.ndarray = None,
) -> SessionState:
    problems = make_problems(n_problems)
    student  = make_student(budget)
    if mask is None:
        mask = np.ones(n_problems, dtype=bool)
    return SessionState(
        student            = student,
        problems           = problems,
        available_mask     = mask,
        session_budget_min = budget,
    )


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

class TestSessionState:
    def test_time_left_norm_at_start(self):
        state = make_state()
        assert state.time_left_norm == pytest.approx(1.0)

    def test_fatigue_at_start(self):
        state = make_state()
        assert state.fatigue == 0.0

    def test_elos_por_tema_shape(self):
        state = make_state()
        assert state.elos_por_tema.shape == (len(CANONICAL_TOPICS),)

    def test_elos_por_tema_in_0_1(self):
        state = make_state()
        assert np.all(state.elos_por_tema >= 0.0)
        assert np.all(state.elos_por_tema <= 1.0)

    def test_available_problems_count(self):
        state = make_state(n_problems=8)
        assert len(state.available_problems()) == 8

    def test_available_problems_respects_mask(self):
        mask = np.ones(8, dtype=bool)
        mask[2] = False
        mask[5] = False
        state = make_state(mask=mask)
        assert len(state.available_problems()) == 6

    def test_p_exito_in_0_1(self):
        state = make_state()
        for _, p in state.available_problems():
            assert 0.0 <= state.p_exito(p) <= 1.0

    def test_delta_elo_esperado_positive(self):
        state = make_state()
        for _, p in state.available_problems():
            assert state.delta_elo_esperado(p) >= 0.0

    def test_t_estimado_positive(self):
        state = make_state()
        for _, p in state.available_problems():
            assert state.t_estimado(p) > 0.0


# ---------------------------------------------------------------------------
# ProblemSelector ABC
# ---------------------------------------------------------------------------

class TestProblemSelectorInterface:
    def test_greedy_is_problem_selector(self):
        assert isinstance(GreedySelector(), ProblemSelector)

    def test_knapsack_is_problem_selector(self):
        assert isinstance(KnapsackSelector(), ProblemSelector)

    def test_rollout_is_problem_selector(self):
        assert isinstance(RolloutSelector(), ProblemSelector)

    def test_all_have_reset(self):
        for cls in [GreedySelector, KnapsackSelector, RolloutSelector]:
            sel = cls()
            sel.reset()   # no debe lanzar excepcion


# ---------------------------------------------------------------------------
# GreedySelector
# ---------------------------------------------------------------------------

class TestGreedySelector:
    @pytest.fixture
    def greedy(self):
        return GreedySelector()

    def test_returns_valid_index(self, greedy):
        state  = make_state()
        action = greedy.select_action(state, state.available_mask)
        assert 0 <= action < state.n_problems

    def test_returns_available_problem(self, greedy):
        state  = make_state()
        action = greedy.select_action(state, state.available_mask)
        assert state.available_mask[action]

    def test_returns_n_when_no_available(self, greedy):
        state = make_state()
        mask  = np.zeros(state.n_problems, dtype=bool)
        action = greedy.select_action(state, mask)
        assert action == state.n_problems

    def test_respects_mask(self, greedy):
        mask = np.zeros(8, dtype=bool)
        mask[3] = True   # solo problema 3 disponible
        state  = make_state(mask=mask)
        action = greedy.select_action(state, mask)
        assert action == 3

    def test_score_method(self, greedy):
        state = make_state()
        for i in range(state.n_problems):
            score = greedy.score(state, i)
            assert isinstance(score, float)
            assert score >= 0.0

    def test_prefers_better_score(self, greedy):
        # Con suficiente presupuesto, el greedy no elige siempre el mas facil
        # sino el de mejor score (aprendizaje/tiempo)
        state  = make_state(budget=300.0)
        action = greedy.select_action(state, state.available_mask)
        best_score = greedy.score(state, action)
        for i, _ in state.available_problems():
            assert greedy.score(state, i) <= best_score + 1e-9


# ---------------------------------------------------------------------------
# KnapsackSelector
# ---------------------------------------------------------------------------

class TestKnapsackSelector:
    @pytest.fixture
    def knapsack(self):
        return KnapsackSelector(w=50.0)

    def test_returns_valid_index(self, knapsack):
        state  = make_state()
        action = knapsack.select_action(state, state.available_mask)
        assert 0 <= action < state.n_problems

    def test_returns_available_problem(self, knapsack):
        state  = make_state()
        action = knapsack.select_action(state, state.available_mask)
        assert state.available_mask[action]

    def test_returns_n_when_no_available(self, knapsack):
        state = make_state()
        mask  = np.zeros(state.n_problems, dtype=bool)
        action = knapsack.select_action(state, mask)
        assert action == state.n_problems

    def test_plan_respects_time_budget(self, knapsack):
        state = make_state(budget=30.0)
        knapsack._solve_knapsack(state, state.available_mask)
        plan = knapsack._plan
        total_time = sum(
            round(state.t_estimado(state.problems[i]))
            for i in plan
        )
        assert total_time <= 30.0

    def test_plan_ordered_by_difficulty(self, knapsack):
        state = make_state(budget=200.0)
        plan  = knapsack._solve_knapsack(state, state.available_mask)
        ratings = [state.problems[i].rating for i in plan]
        assert ratings == sorted(ratings)

    def test_reset_clears_plan(self, knapsack):
        state = make_state()
        knapsack.select_action(state, state.available_mask)
        knapsack.reset()
        assert knapsack._plan     == []
        assert knapsack._plan_pos == 0

    def test_sequential_calls_return_different_problems(self, knapsack):
        state   = make_state(budget=200.0)
        mask    = state.available_mask.copy()
        actions = []
        for _ in range(3):
            action = knapsack.select_action(state, mask)
            if action >= state.n_problems:
                break
            actions.append(action)
            mask[action] = False
        assert len(set(actions)) == len(actions)   # sin repeticiones


# ---------------------------------------------------------------------------
# RolloutSelector
# ---------------------------------------------------------------------------

class TestRolloutSelector:
    @pytest.fixture
    def rollout(self):
        return RolloutSelector()

    def test_returns_valid_index(self, rollout):
        state  = make_state()
        action = rollout.select_action(state, state.available_mask)
        assert 0 <= action < state.n_problems

    def test_returns_available_problem(self, rollout):
        state  = make_state()
        action = rollout.select_action(state, state.available_mask)
        assert state.available_mask[action]

    def test_returns_n_when_no_available(self, rollout):
        state = make_state()
        mask  = np.zeros(state.n_problems, dtype=bool)
        action = rollout.select_action(state, mask)
        assert action == state.n_problems

    def test_with_single_available(self, rollout):
        mask = np.zeros(8, dtype=bool)
        mask[4] = True
        state  = make_state(mask=mask)
        action = rollout.select_action(state, mask)
        assert action == 4

    def test_evaluates_two_steps(self, rollout):
        state  = make_state(budget=200.0)
        # Con presupuesto amplio, el rollout puede evaluar dos pasos
        action = rollout.select_action(state, state.available_mask)
        assert 0 <= action < state.n_problems

    def test_simulate_step_returns_state(self, rollout):
        state   = make_state(budget=200.0)
        sim     = rollout._simulate_step(state, 0, state.available_mask)
        assert sim is not None
        assert isinstance(sim, SessionState)

    def test_simulate_step_first_problem_marked_used(self, rollout):
        state = make_state(budget=200.0)
        sim   = rollout._simulate_step(state, 0, state.available_mask)
        assert not sim.available_mask[0]

    def test_simulate_step_returns_none_when_no_time(self, rollout):
        # Con presupuesto casi agotado
        state         = make_state(budget=120.0)
        state.student.time_spent_min = 119.0
        sim = rollout._simulate_step(state, 0, state.available_mask)
        assert sim is None

    def test_simulate_increases_fatigue(self, rollout):
        state      = make_state(budget=200.0)
        old_fatigue = state.student.fatigue
        sim         = rollout._simulate_step(state, 0, state.available_mask)
        if sim:
            assert sim.student.fatigue > old_fatigue


# ---------------------------------------------------------------------------
# Comparacion entre baselines (test de humo)
# ---------------------------------------------------------------------------

class TestBaselineComparison:
    """Verifica que los tres baselines pueden ejecutar un episodio completo."""

    def _run_episode(self, selector, budget: float = 120.0) -> float:
        """Simula un episodio completo y devuelve la recompensa total."""
        from src.environment.env import TrainingEnv
        problems = make_problems(8)
        env      = TrainingEnv(
            problems           = problems,
            session_budget_min = budget,
            random_seed        = 42,
        )
        obs, info = env.reset()
        total     = 0.0
        selector.reset()

        for _ in range(20):
            mask  = info["action_mask"]
            state = SessionState(
                student            = env.student,
                problems           = problems,
                available_mask     = mask,
                session_budget_min = budget,
            )
            action = selector.select_action(state, mask)
            if action >= len(problems):
                break
            if not mask[action]:
                break
            _, reward, terminated, _, info = env.step(action)
            total += reward
            if terminated:
                break

        return total

    def test_greedy_completes_episode(self):
        reward = self._run_episode(GreedySelector())
        assert isinstance(reward, float)

    def test_knapsack_completes_episode(self):
        reward = self._run_episode(KnapsackSelector())
        assert isinstance(reward, float)

    def test_rollout_completes_episode(self):
        reward = self._run_episode(RolloutSelector())
        assert isinstance(reward, float)