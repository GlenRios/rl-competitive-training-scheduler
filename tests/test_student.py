"""
tests/test_student.py

Tests para StudentModel con ELO por tema y StudentProfileGenerator.

Ejecutar con:
    pytest tests/test_student.py -v
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from src.environment.problem import CANONICAL_TOPICS, N_TOPICS
from src.environment.student_model import StudentModel, AttemptOutcome
from src.environment.student_generator import (
    StudentProfileGenerator, StudentProfile, _ARCHETYPES
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def student():
    """Estudiante con topic_ratings uniformes en 1500."""
    return StudentModel(
        topic_ratings      = {t: 1500 for t in CANONICAL_TOPICS},
        session_budget_min = 300,
        random_seed        = 42,
    )

@pytest.fixture
def specialist():
    """Estudiante fuerte en math/dp, debil en graphs."""
    ratings = {t: 1200 for t in CANONICAL_TOPICS}
    ratings["math"] = 1800
    ratings["dp"]   = 1700
    ratings["graphs"] = 900
    return StudentModel(
        topic_ratings      = ratings,
        session_budget_min = 300,
        random_seed        = 0,
    )

@pytest.fixture
def generator():
    return StudentProfileGenerator(use_llm=False, random_seed=42)


# ---------------------------------------------------------------------------
# state_vector con 24 componentes
# ---------------------------------------------------------------------------

class TestStateVector:
    def test_has_24_components(self, student):
        assert len(student.state_vector) == 24

    def test_all_in_0_1(self, student):
        for v in student.state_vector:
            assert 0.0 <= v <= 1.0, f"Valor fuera de rango: {v}"

    def test_topic_ratings_reflected(self, specialist):
        vec = specialist.state_vector
        topic_norms = vec[4:]  # primeros 4 son global, fatigue, time, solve_rate
        math_idx    = CANONICAL_TOPICS.index("math")
        graphs_idx  = CANONICAL_TOPICS.index("graphs")
        # math (1800) debe tener norm mayor que graphs (900)
        assert topic_norms[math_idx] > topic_norms[graphs_idx]


# ---------------------------------------------------------------------------
# Rating efectivo
# ---------------------------------------------------------------------------

class TestEffectiveRating:
    def test_uses_global_when_no_canonical_tags(self, student):
        r_ef = student._effective_rating(["unknown_topic"])
        assert r_ef == student.global_rating

    def test_uses_mean_of_topic_ratings(self, specialist):
        # math=1800, dp=1700 -> mean=1750
        r_ef = specialist._effective_rating(["math", "dp"])
        assert r_ef == pytest.approx(1750.0)

    def test_single_tag(self, specialist):
        r_ef = specialist._effective_rating(["math"])
        assert r_ef == pytest.approx(1800.0)

    def test_weak_topic_lowers_effective_rating(self, specialist):
        r_ef_strong = specialist._effective_rating(["math"])
        r_ef_mixed  = specialist._effective_rating(["math", "graphs"])
        assert r_ef_strong > r_ef_mixed


# ---------------------------------------------------------------------------
# Probabilidad basada en rating efectivo
# ---------------------------------------------------------------------------

class TestProbabilityWithTopicRatings:
    def test_specialist_higher_p_on_strong_topic(self, specialist):
        # math=1800 vs problema math 1600: deberia ser alta
        p_strong = specialist.probability_of_solving(1600, ["math"])
        # graphs=900 vs problema graphs 1600: deberia ser baja
        p_weak   = specialist.probability_of_solving(1600, ["graphs"])
        assert p_strong > p_weak

    def test_mixed_tags_uses_mean(self, specialist):
        # math=1800, graphs=900 -> r_ef=1350 vs problema 1500
        p_mixed = specialist.probability_of_solving(1500, ["math", "graphs"])
        # Solo math: r_ef=1800 vs 1500 -> mas alta
        p_math  = specialist.probability_of_solving(1500, ["math"])
        assert p_math > p_mixed

    def test_probability_in_0_1(self, student):
        for rating in [800, 1200, 1500, 1800, 2400]:
            p = student.probability_of_solving(rating, ["math", "dp"])
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Actualizacion ELO con factor de desafio
# ---------------------------------------------------------------------------

class TestTopicEloUpdate:
    def test_easy_problem_gives_minimal_delta(self, student):
        # dp=1500 vs problema dp=800: gap=-700 -> challenge muy bajo
        deltas = student._update_topic_ratings(800, ["dp"])
        assert deltas["dp"] < 1.0   # ganancia minima

    def test_hard_problem_gives_large_delta(self, student):
        # dp=1500 vs problema dp=2200: gap=+700 -> challenge alto
        deltas = student._update_topic_ratings(2200, ["dp"])
        assert deltas["dp"] > 5.0   # ganancia significativa

    def test_equal_level_gives_moderate_delta(self, student):
        # dp=1500 vs problema dp=1500: ganancia moderada
        deltas = student._update_topic_ratings(1500, ["dp"])
        assert 1.0 < deltas["dp"] < 5.0

    def test_delta_increases_monotonically_with_difficulty(self, student):
        deltas = []
        for d_p in [800, 1000, 1200, 1500, 1800, 2000, 2400]:
            s = StudentModel(
                topic_ratings={t: 1500 for t in CANONICAL_TOPICS},
                random_seed=0
            )
            d = s._update_topic_ratings(d_p, ["dp"])
            deltas.append(d["dp"])
        assert all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))

    def test_multiple_tags_all_updated(self, student):
        deltas = student._update_topic_ratings(1700, ["math", "dp", "graphs"])
        assert "math"   in deltas
        assert "dp"     in deltas
        assert "graphs" in deltas

    def test_unknown_tag_ignored(self, student):
        deltas = student._update_topic_ratings(1500, ["math", "unknown_xyz"])
        assert "unknown_xyz" not in deltas
        assert "math" in deltas

    def test_topic_rating_increases_after_solve(self, student):
        old = student.topic_ratings["dp"]
        student._update_topic_ratings(1700, ["dp"])
        assert student.topic_ratings["dp"] > old

    def test_topic_rating_capped_at_max(self):
        s = StudentModel(topic_ratings={t: 2990 for t in CANONICAL_TOPICS})
        s._update_topic_ratings(3500, ["dp"])
        assert s.topic_ratings["dp"] <= 3000


# ---------------------------------------------------------------------------
# Global rating como ponderado
# ---------------------------------------------------------------------------

class TestGlobalRating:
    def test_initial_global_is_mean_of_topics(self, student):
        # Todos en 1500 -> global = 1500
        assert student.global_rating == pytest.approx(1500.0, abs=1.0)

    def test_global_updates_after_solve(self, student):
        old_global = student.global_rating
        student._update_topic_ratings(2000, ["dp"])
        student._update_global_rating()
        assert student.global_rating >= old_global


# ---------------------------------------------------------------------------
# attempt() integrado
# ---------------------------------------------------------------------------

class TestAttemptIntegrated:
    def test_returns_attempt_outcome(self, student):
        assert isinstance(student.attempt(1500, ["math"]), AttemptOutcome)

    def test_outcome_has_topic_deltas(self, student):
        outcome = student.attempt(1500, ["math", "dp"])
        assert isinstance(outcome.topic_deltas, dict)

    def test_topic_deltas_empty_when_failed(self, student):
        # Intentar problema casi imposible -> casi siempre falla
        s = StudentModel(
            topic_ratings={t: 800 for t in CANONICAL_TOPICS},
            session_budget_min=600,
            random_seed=5
        )
        outcomes = [s.attempt(3500, ["dp"]) for _ in range(10)]
        failures = [o for o in outcomes if not o.solved]
        if failures:
            assert all(len(o.topic_deltas) == 0 for o in failures)

    def test_topic_deltas_populated_when_solved(self, student):
        # Usar problema facil y muchos intentos para asegurar al menos uno resuelto
        s = StudentModel(
            topic_ratings={t: 2000 for t in CANONICAL_TOPICS},
            session_budget_min=600,
            random_seed=42
        )
        outcomes = [s.attempt(800, ["math"]) for _ in range(10)]
        successes = [o for o in outcomes if o.solved]
        if successes:
            assert all(len(o.topic_deltas) > 0 for o in successes)

    def test_reward_positive_when_solved(self, student):
        s = StudentModel(
            topic_ratings={t: 2000 for t in CANONICAL_TOPICS},
            session_budget_min=600,
            random_seed=1
        )
        outcomes = [s.attempt(800, ["math"]) for _ in range(10)]
        successes = [o for o in outcomes if o.solved]
        if successes:
            assert all(o.reward > 0 for o in successes)

    def test_reward_equals_r_fracaso_when_failed(self, student):
        s = StudentModel(
            topic_ratings={t: 800 for t in CANONICAL_TOPICS},
            session_budget_min=600,
            random_seed=7
        )
        outcomes = [s.attempt(3500, ["dp"]) for _ in range(5)]
        failures = [o for o in outcomes if not o.solved]
        if failures:
            assert all(o.reward == s.r_fracaso for o in failures)

    def test_reset_restores_topic_ratings(self, specialist):
        specialist.attempt(2000, ["math"])
        specialist.reset()
        assert specialist.topic_ratings["math"] == pytest.approx(1800.0)
        assert specialist.topic_ratings["graphs"] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# StudentProfileGenerator
# ---------------------------------------------------------------------------

class TestStudentProfileGenerator:
    def test_generate_returns_profile(self, generator):
        profile = generator.generate()
        assert isinstance(profile, StudentProfile)

    def test_profile_has_all_canonical_topics(self, generator):
        profile = generator.generate()
        for topic in CANONICAL_TOPICS:
            assert topic in profile.topic_ratings

    def test_profile_ratings_in_valid_range(self, generator):
        for _ in range(5):
            profile = generator.generate()
            for rating in profile.topic_ratings.values():
                assert 800 <= rating <= 3000

    def test_profile_budget_in_valid_range(self, generator):
        for _ in range(5):
            profile = generator.generate()
            assert 60 <= profile.session_budget_min <= 180

    def test_fallback_source_marked(self, generator):
        profile = generator.generate()
        assert profile.source == "fallback"

    def test_to_student_model_returns_student(self, generator):
        profile = generator.generate()
        student = profile.to_student_model(random_seed=0)
        assert isinstance(student, StudentModel)

    def test_student_has_correct_topic_ratings(self, generator):
        profile = generator.generate()
        student = profile.to_student_model()
        for topic in CANONICAL_TOPICS:
            assert student.topic_ratings[topic] == pytest.approx(
                profile.topic_ratings[topic], abs=1.0
            )

    def test_generate_batch_returns_n_profiles(self, generator):
        batch = generator.generate_batch(5)
        assert len(batch) == 5
        assert all(isinstance(p, StudentProfile) for p in batch)

    def test_llm_response_parsed_correctly(self, generator):
        """Simula una respuesta LLM valida y verifica el parsing."""
        fake_response = {
            "archetype": "Test student",
            "global_rating": 1400,
            "session_budget_min": 120,
            "topic_ratings": {t: 1400 for t in CANONICAL_TOPICS}
        }
        import json
        raw = json.dumps(fake_response)
        profile = generator._parse_llm_response(raw)
        assert profile is not None
        assert profile.global_rating == pytest.approx(1400.0)
        assert profile.source == "llm"

    def test_malformed_json_returns_none(self, generator):
        profile = generator._parse_llm_response("not json at all")
        assert profile is None

    def test_llm_fallback_on_connection_error(self):
        import requests
        gen = StudentProfileGenerator(use_llm=True, random_seed=0)
        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            profile = gen.generate()
        assert profile is not None
        assert profile.source == "fallback"

    def test_llm_profile_source_marked(self):
        import json
        gen = StudentProfileGenerator(use_llm=True, random_seed=0)
        fake_json = json.dumps({
            "archetype": "LLM student",
            "global_rating": 1500,
            "session_budget_min": 120,
            "topic_ratings": {t: 1500 for t in CANONICAL_TOPICS}
        })
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"response": fake_json}
        with patch("requests.post", return_value=mock_resp):
            profile = gen.generate()
        assert profile.source == "llm"

    def test_missing_topics_filled_with_default(self, generator):
        """Si el LLM omite algunos topics, se rellenan con valor por defecto."""
        import json
        partial = {
            "archetype": "Partial student",
            "global_rating": 1200,
            "session_budget_min": 120,
            "topic_ratings": {"math": 1500, "dp": 1400}  # solo 2 topics
        }
        profile = generator._validate_and_build(partial, source="llm")
        assert profile is not None
        assert len(profile.topic_ratings) == N_TOPICS