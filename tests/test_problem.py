"""
tests/test_problem.py

Ejecutar con:
    pytest tests/test_problem.py -v
"""

import pandas as pd
import pytest

from src.environment.problem import (
    Problem,
    CANONICAL_TOPICS,
    N_TOPICS,
    _TOPIC_IDX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_problem() -> Problem:
    return Problem(
        problem_id      = "325A",
        name            = "Theatre Square",
        rating          = 1500,
        tags_list       = ["math", "dp", "unknown_tag"],
        statement       = "Find the minimum number of tiles to cover a square.",
        difficulty_band = "1500-1699",
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "problem_id":      ["1A", "2B", "3C"],
        "name":            ["Problem A", "Problem B", "Problem C"],
        "rating":          [1000, 1500, 2000],
        "tags_list":       [["math"], ["dp", "graphs"], ["greedy", "trees"]],
        "statement":       ["Statement A " * 5, "Statement B " * 5, "Statement C " * 5],
        "difficulty_band": ["800-1199", "1200-1599", "1600-1999"],
        "time_limit_ms":   [pd.NA, 2000, pd.NA],
        "solved_count":    [50000, pd.NA, 10000],
    })


def make_obs_kwargs(
    student_rating=1500,
    student_fatigue=0.0,
    session_budget_min=120.0,
    time_spent_min=0.0,
    p_solve=0.72,
) -> dict:
    return dict(
        student_rating=student_rating,
        student_fatigue=student_fatigue,
        session_budget_min=session_budget_min,
        time_spent_min=time_spent_min,
        p_solve=p_solve,
    )


# ---------------------------------------------------------------------------
# CANONICAL_TOPICS
# ---------------------------------------------------------------------------

class TestCanonicalTopics:
    def test_has_20_topics(self):
        assert len(CANONICAL_TOPICS) == 20

    def test_n_topics_matches_list(self):
        assert N_TOPICS == len(CANONICAL_TOPICS)

    def test_topic_idx_covers_all(self):
        assert len(_TOPIC_IDX) == N_TOPICS

    def test_common_topics_present(self):
        for topic in ["math", "dp", "greedy", "graphs", "implementation"]:
            assert topic in CANONICAL_TOPICS

    def test_no_duplicate_topics(self):
        assert len(CANONICAL_TOPICS) == len(set(CANONICAL_TOPICS))


# ---------------------------------------------------------------------------
# from_row()
# ---------------------------------------------------------------------------

class TestFromRow:
    def test_builds_problem_from_row(self, sample_df):
        p = Problem.from_row(sample_df.iloc[0])
        assert isinstance(p, Problem)

    def test_problem_id_correct(self, sample_df):
        p = Problem.from_row(sample_df.iloc[0])
        assert p.problem_id == "1A"

    def test_rating_correct(self, sample_df):
        p = Problem.from_row(sample_df.iloc[1])
        assert p.rating == 1500

    def test_tags_list_correct(self, sample_df):
        p = Problem.from_row(sample_df.iloc[1])
        assert "dp" in p.tags_list
        assert "graphs" in p.tags_list

    def test_tags_list_as_string_parsed(self):
        row = pd.Series({
            "problem_id": "10A", "name": "X", "rating": 1200,
            "tags_list": "['math', 'greedy']",   # string, no lista
            "statement": "A" * 50,
            "difficulty_band": "1200-1599",
            "time_limit_ms": pd.NA, "solved_count": pd.NA,
        })
        p = Problem.from_row(row)
        assert "math" in p.tags_list
        assert "greedy" in p.tags_list

    def test_optional_fields_none_when_na(self, sample_df):
        p = Problem.from_row(sample_df.iloc[0])   # time_limit_ms es NA
        assert p.time_limit_ms is None

    def test_optional_fields_set_when_available(self, sample_df):
        p = Problem.from_row(sample_df.iloc[1])   # time_limit_ms = 2000
        assert p.time_limit_ms == 2000

    def test_tags_lowercase(self):
        row = pd.Series({
            "problem_id": "5A", "name": "Y", "rating": 1000,
            "tags_list": ["MATH", "DP"],
            "statement": "B" * 50,
            "difficulty_band": "800-1199",
            "time_limit_ms": pd.NA, "solved_count": pd.NA,
        })
        p = Problem.from_row(row)
        assert all(t == t.lower() for t in p.tags_list)


# ---------------------------------------------------------------------------
# from_dataframe()
# ---------------------------------------------------------------------------

class TestFromDataframe:
    def test_returns_list_of_problems(self, sample_df):
        problems = Problem.from_dataframe(sample_df)
        assert all(isinstance(p, Problem) for p in problems)

    def test_correct_count(self, sample_df):
        problems = Problem.from_dataframe(sample_df)
        assert len(problems) == 3

    def test_order_preserved(self, sample_df):
        problems = Problem.from_dataframe(sample_df)
        assert problems[0].problem_id == "1A"
        assert problems[1].problem_id == "2B"
        assert problems[2].problem_id == "3C"


# ---------------------------------------------------------------------------
# to_observation_vector()
# ---------------------------------------------------------------------------

class TestObservationVector:
    def test_correct_length(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        assert len(vec) == 4 + N_TOPICS   # 24

    def test_all_values_numeric(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        assert all(isinstance(v, float) for v in vec)

    def test_rating_norm_in_0_1(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        assert 0.0 <= vec[0] <= 1.0

    def test_gap_norm_in_minus1_1(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        assert -1.0 <= vec[1] <= 1.0

    def test_p_solve_preserved(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs(p_solve=0.55))
        assert vec[2] == 0.55

    def test_time_norm_in_0_1(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        assert 0.0 <= vec[3] <= 1.0

    def test_onehot_is_binary(self, sample_problem):
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        onehot = vec[4:]
        assert all(v in (0.0, 1.0) for v in onehot)

    def test_known_tags_set_in_onehot(self, sample_problem):
        # sample_problem tiene "math" y "dp"
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        onehot = vec[4:]
        math_idx = CANONICAL_TOPICS.index("math")
        dp_idx   = CANONICAL_TOPICS.index("dp")
        assert onehot[math_idx] == 1.0
        assert onehot[dp_idx]   == 1.0

    def test_unknown_tags_not_set(self, sample_problem):
        # "unknown_tag" no está en CANONICAL_TOPICS → no debe cambiar el onehot
        vec = sample_problem.to_observation_vector(**make_obs_kwargs())
        onehot = vec[4:]
        # Solo math (1) y dp (3) deben ser 1
        assert sum(onehot) == 2.0

    def test_positive_gap_yields_higher_time_norm(self):
        # Problema difícil (gap positivo) debe tener time_norm mayor
        p_easy = Problem("1A", "", 800,  ["math"], "s" * 50, "800-1199")
        p_hard = Problem("2A", "", 2500, ["math"], "s" * 50, "2400-2799")
        kwargs = make_obs_kwargs(student_rating=1500)
        vec_easy = p_easy.to_observation_vector(**kwargs)
        vec_hard = p_hard.to_observation_vector(**kwargs)
        assert vec_hard[3] > vec_easy[3]

    def test_negative_gap_yields_lower_time_norm(self):
        p_below = Problem("1A", "", 800,  ["math"], "s" * 50, "800-1199")
        p_level = Problem("2A", "", 1500, ["math"], "s" * 50, "1500-1699")
        kwargs  = make_obs_kwargs(student_rating=1500)
        vec_below = p_below.to_observation_vector(**kwargs)
        vec_level = p_level.to_observation_vector(**kwargs)
        assert vec_below[3] < vec_level[3]

    def test_fatigue_increases_time_norm(self, sample_problem):
        vec_fresh    = sample_problem.to_observation_vector(**make_obs_kwargs(student_fatigue=0.0))
        vec_fatigued = sample_problem.to_observation_vector(**make_obs_kwargs(student_fatigue=0.9))
        assert vec_fatigued[3] >= vec_fresh[3]


# ---------------------------------------------------------------------------
# Propiedades
# ---------------------------------------------------------------------------

class TestProperties:
    def test_obs_dim_is_24(self, sample_problem):
        assert sample_problem.obs_dim == 24

    def test_canonical_tags_filters_unknown(self, sample_problem):
        # "unknown_tag" no debe aparecer
        assert "unknown_tag" not in sample_problem.canonical_tags
        assert "math" in sample_problem.canonical_tags
        assert "dp" in sample_problem.canonical_tags

    def test_n_canonical_tags(self, sample_problem):
        assert sample_problem.n_canonical_tags == 2

    def test_is_untagged_false_when_has_tags(self, sample_problem):
        assert not sample_problem.is_untagged

    def test_is_untagged_true_when_only_unknown_tags(self):
        p = Problem("X", "", 1200, ["unknown_tag_1", "xyz"], "s" * 50, "1200-1599")
        assert p.is_untagged

    def test_repr_contains_problem_id(self, sample_problem):
        assert "325A" in repr(sample_problem)

    def test_repr_contains_rating(self, sample_problem):
        assert "1500" in repr(sample_problem)