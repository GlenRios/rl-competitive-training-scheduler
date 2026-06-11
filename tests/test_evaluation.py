"""
tests/test_evaluation.py

Tests para metrics.py y evaluator.py.

Ejecutar con:
    pytest tests/test_evaluation.py -v
"""

import math
import pytest
import pandas as pd

from src.eval.metrics import (
    compute_episode_metrics,
    aggregate_metrics,
    _spearman_manual,
    _ranks,
    _topic_coverage,
)
from src.environment.problem import CANONICAL_TOPICS, Problem
from src.environment.student_generator import StudentProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_history(
    ratings   : list[int],
    solved    : list[bool] = None,
    times     : list[float] = None,
    rewards   : list[float] = None,
) -> list[dict]:
    n       = len(ratings)
    solved  = solved  or [True] * n
    times   = times   or [20.0] * n
    rewards = rewards or [10.0 if s else -2.0 for s in solved]
    history = []
    for i, (r, s, t, rw) in enumerate(zip(ratings, solved, times, rewards)):
        history.append({
            "step"          : i + 1,
            "problem_rating": r,
            "problem_tags"  : ["math", "dp"],
            "solved"        : s,
            "time_min"      : t,
            "reward"        : rw,
            "topic_deltas"  : {"math": 2.0, "dp": 1.5} if s else {},
            "new_global_rating": 1500.0,
            "fatigue"       : 0.1 * (i + 1),
            "time_remaining": 120.0 - t * (i + 1),
        })
    return history


def make_problems(n: int = 10) -> list[Problem]:
    return [
        Problem(
            problem_id      = f"{i+1}A",
            name            = f"P{i+1}",
            rating          = 800 + i * 150,
            tags_list       = ["math"] if i % 2 == 0 else ["dp", "graphs"],
            statement       = "stmt " * 10,
            difficulty_band = "800-1199",
        )
        for i in range(n)
    ]


def make_profile(rating: float = 1400.0) -> StudentProfile:
    return StudentProfile(
        archetype          = "Test student",
        global_rating      = rating,
        session_budget_min = 120.0,
        topic_ratings      = {t: int(rating) for t in CANONICAL_TOPICS},
        source             = "fallback",
    )


# ---------------------------------------------------------------------------
# Tests de metricas individuales
# ---------------------------------------------------------------------------

class TestTotalReward:
    def test_sums_rewards(self):
        history = make_history([1000, 1200], rewards=[10.0, -2.0])
        m = compute_episode_metrics(history, 1400.0, 1405.0)
        assert m["total_reward"] == pytest.approx(8.0)

    def test_zero_for_empty_history(self):
        m = compute_episode_metrics([], 1400.0, 1400.0)
        assert m["total_reward"] == 0.0


class TestSpearmanProgression:
    def test_perfect_ascending_order(self):
        # Ratings en orden ascendente -> Spearman = 1.0
        history = make_history([800, 1000, 1200, 1500, 1800])
        m = compute_episode_metrics(history, 1400.0, 1410.0)
        assert m["spearman_progression"] == pytest.approx(1.0, abs=0.01)

    def test_perfect_descending_order(self):
        # Ratings en orden descendente -> Spearman = -1.0
        history = make_history([1800, 1500, 1200, 1000, 800])
        m = compute_episode_metrics(history, 1400.0, 1410.0)
        assert m["spearman_progression"] == pytest.approx(-1.0, abs=0.01)

    def test_single_problem_returns_zero(self):
        history = make_history([1500])
        m = compute_episode_metrics(history, 1400.0, 1402.0)
        assert m["spearman_progression"] == 0.0

    def test_value_between_minus1_and_1(self):
        history = make_history([1200, 800, 1600, 1000, 2000])
        m = compute_episode_metrics(history, 1400.0, 1410.0)
        assert -1.0 <= m["spearman_progression"] <= 1.0


class TestTopicCoverage:
    def test_counts_unique_canonical_tags(self):
        history = [
            {"solved": True,  "topic_deltas": {"math": 2.0, "dp": 1.5}, "problem_tags": ["math", "dp"]},
            {"solved": True,  "topic_deltas": {"graphs": 3.0},           "problem_tags": ["graphs"]},
            {"solved": False, "topic_deltas": {},                         "problem_tags": ["math"]},
        ]
        count = _topic_coverage(history)
        assert count == 3   # math, dp, graphs

    def test_unknown_tags_not_counted(self):
        history = [
            {"solved": True, "topic_deltas": {"unknown_xyz": 1.0}, "problem_tags": ["unknown_xyz"]},
        ]
        assert _topic_coverage(history) == 0

    def test_empty_history_zero(self):
        assert _topic_coverage([]) == 0


class TestSuccessRate:
    def test_all_solved(self):
        history = make_history([1000, 1200], solved=[True, True])
        m = compute_episode_metrics(history, 1400.0, 1408.0)
        assert m["success_rate"] == pytest.approx(1.0)

    def test_none_solved(self):
        history = make_history([1000, 1200], solved=[False, False])
        m = compute_episode_metrics(history, 1400.0, 1400.0)
        assert m["success_rate"] == pytest.approx(0.0)

    def test_half_solved(self):
        history = make_history([1000, 1200], solved=[True, False])
        m = compute_episode_metrics(history, 1400.0, 1404.0)
        assert m["success_rate"] == pytest.approx(0.5)


class TestTimeUsed:
    def test_sums_time(self):
        history = make_history([1000, 1200], times=[15.0, 25.0])
        m = compute_episode_metrics(history, 1400.0, 1408.0)
        assert m["time_used_min"] == pytest.approx(40.0)


class TestRatingImprovement:
    def test_positive_improvement(self):
        m = compute_episode_metrics(make_history([1000]), 1400.0, 1408.0)
        assert m["rating_improvement"] == pytest.approx(8.0)

    def test_zero_improvement(self):
        m = compute_episode_metrics(make_history([1000], solved=[False]), 1400.0, 1400.0)
        assert m["rating_improvement"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests de spearman manual
# ---------------------------------------------------------------------------

class TestSpearmanManual:
    def test_identical_to_perfect_ascending(self):
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        assert _spearman_manual(x, y) == pytest.approx(1.0)

    def test_identical_to_perfect_descending(self):
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        assert _spearman_manual(x, y) == pytest.approx(-1.0)

    def test_tied_values_handled(self):
        # No debe lanzar excepcion con empates
        x = [1, 2, 3]
        y = [10, 10, 20]
        result = _spearman_manual(x, y)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Tests de agregacion
# ---------------------------------------------------------------------------

class TestAggregateMetrics:
    def test_returns_mean_and_std(self):
        episodes = [
            {"total_reward": 10.0, "success_rate": 0.8, "topic_coverage": 5,
             "spearman_progression": 0.7, "time_used_min": 60.0,
             "rating_improvement": 5.0, "n_attempted": 3, "n_solved": 2,
             "compute_time_s": 0.1},
            {"total_reward": 20.0, "success_rate": 0.6, "topic_coverage": 4,
             "spearman_progression": 0.5, "time_used_min": 80.0,
             "rating_improvement": 8.0, "n_attempted": 4, "n_solved": 3,
             "compute_time_s": 0.2},
        ]
        agg = aggregate_metrics(episodes)
        assert "total_reward_mean" in agg
        assert "total_reward_std" in agg
        assert agg["total_reward_mean"] == pytest.approx(15.0)

    def test_empty_returns_empty_dict(self):
        assert aggregate_metrics([]) == {}

    def test_std_zero_for_identical_values(self):
        episodes = [
            {"total_reward": 5.0, "success_rate": 0.5, "topic_coverage": 3,
             "spearman_progression": 0.3, "time_used_min": 50.0,
             "rating_improvement": 3.0, "n_attempted": 2, "n_solved": 1,
             "compute_time_s": 0.1},
        ] * 3
        agg = aggregate_metrics(episodes)
        assert agg["total_reward_std"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests del Evaluator
# ---------------------------------------------------------------------------

class TestEvaluator:
    @pytest.fixture
    def problems(self):
        return make_problems(10)

    @pytest.fixture
    def profiles(self):
        return [make_profile(1200 + i * 100) for i in range(5)]

    @pytest.fixture
    def mock_selector(self):
        """Selector que siempre elige la primera accion valida."""
        from src.baselines.selector import ProblemSelector
        class FirstAvailableSelector(ProblemSelector):
            def select_action(self, state, available_mask, **kwargs):
                for i, available in enumerate(available_mask):
                    if available:
                        return i
                return len(available_mask)  # parar
        return FirstAvailableSelector()

    def test_run_returns_dataframe(self, problems, profiles, mock_selector):
        from src.eval.evaluator import Evaluator
        ev = Evaluator(problems=problems, n_episodes=3, random_seed=42)
        df = ev.run({"first": mock_selector}, profiles=profiles)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, problems, profiles, mock_selector):
        from src.eval.evaluator import Evaluator
        ev = Evaluator(problems=problems, n_episodes=5, random_seed=42)
        df = ev.run({"first": mock_selector}, profiles=profiles)
        assert len(df) == 5   # 1 selector * 5 episodios

    def test_required_columns_present(self, problems, profiles, mock_selector):
        from src.eval.evaluator import Evaluator
        ev = Evaluator(problems=problems, n_episodes=3, random_seed=42)
        df = ev.run({"first": mock_selector}, profiles=profiles)
        required = {
            "selector", "archetype", "global_rating", "episode",
            "total_reward", "spearman_progression", "topic_coverage",
            "success_rate", "time_used_min", "rating_improvement",
            "n_attempted", "n_solved", "compute_time_s",
        }
        assert required.issubset(df.columns)

    def test_multiple_selectors_all_evaluated(self, problems, profiles):
        from src.eval.evaluator import Evaluator
        from src.baselines.selector import ProblemSelector
        class DummySelector(ProblemSelector):
            def select_action(self, state, available_mask, **kwargs):
                for i, a in enumerate(available_mask):
                    if a: return i
                return len(available_mask)

        ev = Evaluator(problems=problems, n_episodes=3, random_seed=0)
        df = ev.run({
            "A": DummySelector(),
            "B": DummySelector(),
        }, profiles=profiles)
        assert set(df["selector"].unique()) == {"A", "B"}
        assert len(df) == 6   # 2 selectores * 3 episodios

    def test_episode_numbers_sequential(self, problems, profiles, mock_selector):
        from src.eval.evaluator import Evaluator
        ev = Evaluator(problems=problems, n_episodes=4, random_seed=0)
        df = ev.run({"s": mock_selector}, profiles=profiles)
        assert list(df["episode"]) == [1, 2, 3, 4]

    def test_time_used_within_budget(self, problems, profiles, mock_selector):
        from src.eval.evaluator import Evaluator
        ev = Evaluator(problems=problems, n_episodes=3, random_seed=0)
        df = ev.run({"s": mock_selector}, profiles=profiles)
        budget = profiles[0].session_budget_min
        assert (df["time_used_min"] <= budget + 1.0).all()

    def test_success_rate_between_0_and_1(self, problems, profiles, mock_selector):
        from src.eval.evaluator import Evaluator
        ev = Evaluator(problems=problems, n_episodes=3, random_seed=0)
        df = ev.run({"s": mock_selector}, profiles=profiles)
        assert (df["success_rate"] >= 0.0).all()
        assert (df["success_rate"] <= 1.0).all()