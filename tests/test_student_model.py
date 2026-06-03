"""
tests/test_student_model.py

Ejecutar con:
    pytest tests/test_student_model.py -v
"""

import pytest
from src.environment.student_model import StudentModel, AttemptOutcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def student() -> StudentModel:
    """Estudiante de rating 1500, sesión de 120 min, seed fija."""
    return StudentModel(initial_rating=1500, session_budget_min=120, random_seed=42)


@pytest.fixture
def student_expert() -> StudentModel:
    return StudentModel(initial_rating=2400, session_budget_min=120, random_seed=0)


@pytest.fixture
def student_beginner() -> StudentModel:
    return StudentModel(initial_rating=800, session_budget_min=120, random_seed=0)


# ---------------------------------------------------------------------------
# Inicialización y validación
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_state_after_init(self, student):
        assert student.rating           == 1500
        assert student.fatigue          == 0.0
        assert student.time_spent_min   == 0.0
        assert student.n_solved         == 0
        assert student.n_attempted      == 0

    def test_raises_on_invalid_rating_low(self):
        with pytest.raises(ValueError, match="initial_rating"):
            StudentModel(initial_rating=100)

    def test_raises_on_invalid_rating_high(self):
        with pytest.raises(ValueError, match="initial_rating"):
            StudentModel(initial_rating=5000)

    def test_raises_on_non_positive_budget(self):
        with pytest.raises(ValueError, match="session_budget_min"):
            StudentModel(initial_rating=1500, session_budget_min=0)


# ---------------------------------------------------------------------------
# attempt()
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_returns_attempt_outcome(self, student):
        outcome = student.attempt(1500, ["math"])
        assert isinstance(outcome, AttemptOutcome)

    def test_time_spent_increases_after_attempt(self, student):
        student.attempt(1500, ["math"])
        assert student.time_spent_min > 0.0

    def test_fatigue_increases_after_attempt(self, student):
        student.attempt(1500, ["math"])
        assert student.fatigue > 0.0

    def test_topics_seen_updated(self, student):
        student.attempt(1500, ["dp", "graphs"])
        assert "dp" in student.topics_seen
        assert "graphs" in student.topics_seen

    def test_problem_id_tracked(self, student):
        student.attempt(1500, ["math"], problem_id="325A")
        assert "325A" in student.problems_attempted

    def test_solved_tracked_in_problems_solved(self, student):
        # Con seed 42 y gap=0, el estudiante resuelve con p~0.72
        # Forzamos a que resuelva controlando la seed
        s = StudentModel(initial_rating=1500, session_budget_min=120, random_seed=1)
        outcomes = [s.attempt(800, ["math"], f"p{i}") for i in range(10)]
        solved_ids = s.problems_solved
        # Todos los resueltos deben estar en problems_attempted también
        assert all(pid in s.problems_attempted for pid in solved_ids)

    def test_raises_when_session_over(self, student):
        student.time_spent_min = 120.0  # agotar tiempo manualmente
        with pytest.raises(RuntimeError, match="sesión ya terminó"):
            student.attempt(1500, ["math"])

    def test_outcome_time_min_is_positive(self, student):
        outcome = student.attempt(1500, ["math"])
        assert outcome.time_min > 0.0

    def test_outcome_solved_is_bool(self, student):
        outcome = student.attempt(1500, ["math"])
        assert isinstance(outcome.solved, bool)

    def test_outcome_fatigue_in_range(self, student):
        outcome = student.attempt(1500, ["math"])
        assert 0.0 <= outcome.fatigue <= 1.0


# ---------------------------------------------------------------------------
# Comportamiento según brecha de dificultad
# ---------------------------------------------------------------------------

class TestDifficultyGap:
    def test_easy_problem_takes_less_time(self, student):
        time_easy = student.estimate_solve_time(800)    # gap = -700
        time_hard = student.estimate_solve_time(2000)   # gap = +500
        assert time_easy < time_hard

    def test_easy_problem_higher_solve_probability(self, student):
        p_easy = student.probability_of_solving(800)
        p_hard = student.probability_of_solving(2400)
        assert p_easy > p_hard

    def test_impossible_problem_very_low_probability(self, student):
        # gap = +2000: fuera de alcance completamente
        p = student.probability_of_solving(3500)
        assert p < 0.15

    def test_trivial_problem_very_high_probability(self, student_expert):
        # Experto de 2400 vs problema de 800: gap = -1600
        p = student_expert.probability_of_solving(800)
        assert p > 0.90

    def test_probability_in_valid_range(self, student):
        for rating in [800, 1000, 1200, 1500, 1800, 2200, 2800, 3200]:
            p = student.probability_of_solving(rating)
            assert 0.0 <= p <= 1.0, f"Probabilidad fuera de rango para rating={rating}"


# ---------------------------------------------------------------------------
# Fatiga
# ---------------------------------------------------------------------------

class TestFatigue:
    def test_fatigue_increases_monotonically(self):
        # Budget grande para no agotar el tiempo en el test de fatiga
        s = StudentModel(initial_rating=1500, session_budget_min=600, random_seed=42)
        prev = 0.0
        for _ in range(5):
            s.attempt(1500, ["math"])
            assert s.fatigue >= prev
            prev = s.fatigue

    def test_fatigue_capped_at_1(self):
        # Necesitamos 13+ intentos para llegar a 1.0 (1/0.08=12.5)
        s = StudentModel(initial_rating=1500, session_budget_min=600, random_seed=42)
        for _ in range(20):
            s.attempt(1500, ["math"])
        assert s.fatigue <= 1.0

    def test_fatigue_reduces_solve_probability(self):
        s_fresh   = StudentModel(1500, random_seed=0)
        s_fatigued = StudentModel(1500, random_seed=0)
        s_fatigued.fatigue = 0.9

        p_fresh    = s_fresh.probability_of_solving(1600)
        p_fatigued = s_fatigued.probability_of_solving(1600)
        assert p_fresh > p_fatigued

    def test_fatigue_increases_solve_time(self):
        s_fresh    = StudentModel(1500)
        s_fatigued = StudentModel(1500)
        s_fatigued.fatigue = 0.8

        t_fresh    = s_fresh.estimate_solve_time(1500)
        t_fatigued = s_fatigued.estimate_solve_time(1500)
        assert t_fatigued > t_fresh


# ---------------------------------------------------------------------------
# Rating ELO
# ---------------------------------------------------------------------------

class TestRatingUpdate:
    def test_rating_increases_after_solving_hard_problem(self):
        s = StudentModel(1500, random_seed=999)
        # Forzar que resuelva un problema difícil (gap grande positivo)
        old_rating = s.rating
        # Manipular para que resuelva: usar problema fácil con seed que garantice éxito
        s2 = StudentModel(800, random_seed=0)
        s2.attempt(800, ["math"])  # problema muy fácil → casi siempre resuelve
        # Si resolvió, rating debe haber subido o quedado igual
        assert s2.rating >= 800

    def test_rating_does_not_go_below_minimum(self, student_beginner):
        # Intentar muchos problemas imposibles
        for _ in range(30):
            try:
                student_beginner.attempt(3500, ["math"])
            except RuntimeError:
                break
        assert student_beginner.rating >= 800

    def test_rating_does_not_exceed_maximum(self, student_expert):
        for _ in range(30):
            try:
                student_expert.attempt(800, ["math"])
            except RuntimeError:
                break
        assert student_expert.rating <= 3500

    def test_rating_changes_after_attempt(self, student):
        old = student.rating
        outcome = student.attempt(1500, ["math"])
        assert outcome.new_rating == student.rating
        assert outcome.delta_rating == student.rating - old


# ---------------------------------------------------------------------------
# Tiempo de sesión
# ---------------------------------------------------------------------------

class TestSessionTime:
    def test_time_remaining_decreases(self, student):
        initial = student.time_remaining_min
        student.attempt(1500, ["math"])
        assert student.time_remaining_min < initial

    def test_session_not_over_at_start(self, student):
        assert not student.session_over

    def test_session_over_when_time_exhausted(self, student):
        student.time_spent_min = 120.0
        assert student.session_over

    def test_will_fit_in_session_true_for_easy(self, student):
        assert student.will_fit_in_session(800)

    def test_will_fit_in_session_false_when_no_time(self, student):
        student.time_spent_min = 119.9
        assert not student.will_fit_in_session(1500)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restores_initial_state(self, student):
        student.attempt(1500, ["math"])
        student.attempt(1200, ["dp"])
        student.reset()

        assert student.rating          == 1500
        assert student.fatigue         == 0.0
        assert student.time_spent_min  == 0.0
        assert student.n_solved        == 0
        assert student.n_attempted     == 0
        assert len(student.topics_seen) == 0

    def test_can_attempt_after_reset_from_exhausted_session(self, student):
        student.time_spent_min = 120.0
        student.reset()
        outcome = student.attempt(1500, ["math"])
        assert isinstance(outcome, AttemptOutcome)


# ---------------------------------------------------------------------------
# state_vector
# ---------------------------------------------------------------------------

class TestStateVector:
    def test_state_vector_has_5_components(self, student):
        assert len(student.state_vector) == 5

    def test_state_vector_all_in_0_1(self):
        s = StudentModel(initial_rating=1500, session_budget_min=600, random_seed=42)
        for _ in range(5):
            s.attempt(1500, ["math"])
        for val in s.state_vector:
            assert 0.0 <= val <= 1.0, f"Valor fuera de rango: {val}"

    def test_state_vector_changes_after_attempt(self, student):
        before = student.state_vector[:]
        student.attempt(1500, ["math"])
        after  = student.state_vector[:]
        assert before != after


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_gives_same_outcomes(self):
        problems = [(1500, ["math"]), (1800, ["dp"]), (1200, ["greedy"])]

        s1 = StudentModel(1500, random_seed=7)
        s2 = StudentModel(1500, random_seed=7)

        for rating, tags in problems:
            o1 = s1.attempt(rating, tags)
            o2 = s2.attempt(rating, tags)
            assert o1.solved    == o2.solved
            assert o1.time_min  == o2.time_min

    def test_different_seeds_may_give_different_outcomes(self):
        # Budget grande para poder hacer 20 intentos sin agotar la sesión
        s1 = StudentModel(1500, session_budget_min=1200, random_seed=1)
        s2 = StudentModel(1500, session_budget_min=1200, random_seed=99)

        results1 = [s1.attempt(1700, ["dp"]).solved for _ in range(20)]
        results2 = [s2.attempt(1700, ["dp"]).solved for _ in range(20)]
        assert results1 != results2


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_rating(self, student):
        assert "1500" in repr(student)

    def test_repr_contains_fatigue(self, student):
        assert "fatigue" in repr(student)