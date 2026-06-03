"""
tests/test_student_model.py

Ejecutar con:
    pytest tests/test_student_model.py -v
"""

import math
import pytest
from src.environment.student_model import StudentModel, AttemptOutcome


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def student():
    return StudentModel(initial_rating=1500, session_budget_min=120, random_seed=42)

@pytest.fixture
def student_large():
    """Presupuesto grande para tests que hacen múltiples intentos."""
    return StudentModel(initial_rating=1500, session_budget_min=600, random_seed=42)


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_state(self, student):
        assert student.rating          == 1500
        assert student.fatigue         == 0.0
        assert student.time_spent_min  == 0.0
        assert student.n_solved        == 0
        assert student.n_attempted     == 0

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
# Sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_zero_is_half(self):
        assert StudentModel._sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_positive_above_half(self):
        assert StudentModel._sigmoid(1.0) > 0.5

    def test_sigmoid_negative_below_half(self):
        assert StudentModel._sigmoid(-1.0) < 0.5

    def test_sigmoid_large_positive_near_one(self):
        assert StudentModel._sigmoid(10.0) > 0.99

    def test_sigmoid_large_negative_near_zero(self):
        assert StudentModel._sigmoid(-10.0) < 0.01

    def test_sigmoid_example_from_spec(self):
        # Ejemplo del enunciado: x = -0.45 → P ≈ 0.39
        # x = (1600-1800)/400 + 0.2 - 0.5*0.3 = -0.5 + 0.2 - 0.15 = -0.45
        p = StudentModel._sigmoid(-0.45)
        assert p == pytest.approx(0.3894, abs=0.01)


# ---------------------------------------------------------------------------
# probability_of_solving — fórmula σ((R_s - d_p)/θ + δ_topics - λ·F)
# ---------------------------------------------------------------------------

class TestProbabilityOfSolving:
    def test_returns_float_in_0_1(self, student):
        p = student.probability_of_solving(1500, ["math"])
        assert 0.0 < p < 1.0

    def test_equal_rating_gives_near_half(self, student):
        # gap=0, δ=-0.25 (sin maestría), F=0 → x=-0.25 → σ(-0.25)≈0.438
        p = student.probability_of_solving(1500, ["math"])
        import math
        expected = 1 / (1 + math.exp(0.25))
        assert p == pytest.approx(expected, abs=0.01)

    def test_easier_problem_higher_probability(self, student):
        p_easy = student.probability_of_solving(800, ["math"])
        p_hard = student.probability_of_solving(2400, ["math"])
        assert p_easy > p_hard

    def test_fatigue_reduces_probability(self):
        s_fresh    = StudentModel(1500, random_seed=0)
        s_fatigued = StudentModel(1500, random_seed=0)
        s_fatigued.fatigue = 0.8
        p_fresh    = s_fresh.probability_of_solving(1600, ["dp"])
        p_fatigued = s_fatigued.probability_of_solving(1600, ["dp"])
        assert p_fresh > p_fatigued

    def test_topic_mastery_increases_probability(self):
        # Comparar δ_topics con y sin maestría en el mismo estudiante y estado
        s = StudentModel(1500, random_seed=0)
        delta_before = s._topic_affinity(["math"])  # sin maestría → -0.25
        # Simular maestría directamente para aislar el efecto
        s._topic_solves["math"] = 10
        delta_after = s._topic_affinity(["math"])   # con maestría → > -0.25
        assert delta_after > delta_before

    def test_manual_example_from_spec(self):
        # R=1600, d_p=1800, θ=400, λ=0.5, F=0.3, δ=-0.25 (sin maestría)
        # x = (1600-1800)/400 + (-0.25) - 0.5*0.3 = -0.5 - 0.25 - 0.15 = -0.90
        s = StudentModel(initial_rating=1600)
        s.fatigue = 0.3
        x = (1600 - 1800) / 400 + (-0.25) - 0.5 * 0.3
        expected = 1 / (1 + math.exp(-x))
        p = s.probability_of_solving(1800, ["math"])
        assert p == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# _solve_time_base — T_min + (T_max - T_min) · f_dif · g_temas
# ---------------------------------------------------------------------------

class TestSolveTimeBase:
    def test_easy_problem_near_t_min(self, student):
        # gap muy negativo → f_dif ≈ 0 → T_base ≈ T_min
        t = student._solve_time_base(800, ["math"])
        assert t == pytest.approx(student.t_min, abs=5.0)

    def test_hard_problem_near_t_max(self, student):
        # gap muy positivo → f_dif ≈ 1 → T_base ≈ T_max
        t = student._solve_time_base(3000, ["math"])
        assert t >= student.t_max * 0.85

    def test_harder_problem_takes_more_time(self, student):
        t_easy = student._solve_time_base(1000, ["math"])
        t_hard = student._solve_time_base(2000, ["math"])
        assert t_hard > t_easy

    def test_more_topics_increases_time(self, student):
        t_one  = student._solve_time_base(1500, ["math"])
        t_three = student._solve_time_base(1500, ["math", "dp", "graphs"])
        assert t_three > t_one

    def test_g_temas_formula(self, student):
        # g_temas = 1 + α*(n-1) = 1 + 0.2*2 = 1.4 con 3 temas
        f_dif   = max(0, min(1, (1500 - 1500 + 200) / 800))   # = 0.25
        g_temas = 1 + 0.2 * (3 - 1)   # = 1.4
        expected = student.t_min + (student.t_max - student.t_min) * f_dif * g_temas
        actual   = student._solve_time_base(1500, ["math", "dp", "graphs"])
        assert actual == pytest.approx(expected, abs=0.1)

    def test_f_dificultad_clipped_at_zero(self, student):
        # Problema muy fácil: f_dif no puede ser negativo
        t = student._solve_time_base(800, ["math"])
        assert t >= student.t_min

    def test_f_dificultad_clipped_at_one(self, student):
        # Problema imposible: f_dif no puede superar 1
        t = student._solve_time_base(3500, ["math"])
        assert t <= student.t_max * 2.0   # g_temas puede superar 1


# ---------------------------------------------------------------------------
# _topic_affinity
# ---------------------------------------------------------------------------

class TestTopicAffinity:
    def test_zero_mastery_gives_negative_delta(self, student):
        # Sin resolver ningún problema → maestría=0 → mean=0 → delta=0-0.25=-0.25
        delta = student._topic_affinity(["dp", "graphs"])
        assert delta == pytest.approx(-0.25, abs=0.01)

    def test_unknown_tags_give_zero(self, student):
        delta = student._topic_affinity(["unknown_topic_xyz"])
        assert delta == 0.0

    def test_delta_increases_with_solves(self):
        s = StudentModel(1500, random_seed=0)
        before = s._topic_affinity(["math"])        # sin maestría → -0.25
        s._topic_solves["math"] = 10               # simular maestría directamente
        after  = s._topic_affinity(["math"])
        assert after > before

    def test_delta_capped_at_max(self, student_large):
        # Muchas resoluciones → delta no supera _DELTA_MAX
        for _ in range(50):
            student_large.attempt(800, ["math"])
        delta = student_large._topic_affinity(["math"])
        assert delta <= 0.5

    def test_delta_capped_at_min(self, student):
        delta = student._topic_affinity(["math"])
        assert delta >= -0.5


# ---------------------------------------------------------------------------
# _update_rating — ELO: ΔR = C*(1 - P) si resuelve, 0 si fracasa
# ---------------------------------------------------------------------------

class TestUpdateRating:
    def test_solving_increases_rating(self, student):
        old = student.rating
        delta = student._update_rating(p_solve=0.3, solved=True)
        assert student.rating > old
        assert delta > 0

    def test_failing_does_not_change_rating(self, student):
        old = student.rating
        delta = student._update_rating(p_solve=0.6, solved=False)
        assert student.rating == old
        assert delta == 0

    def test_harder_problem_gives_more_rating(self, student):
        s1 = StudentModel(1500)
        s2 = StudentModel(1500)
        d1 = s1._update_rating(p_solve=0.2, solved=True)   # difícil → más ΔR
        d2 = s2._update_rating(p_solve=0.8, solved=True)   # fácil   → menos ΔR
        assert d1 > d2

    def test_delta_formula_c_times_one_minus_p(self, student):
        # ΔR = round(C * (1 - p)) = round(10 * 0.6) = 6
        delta = student._update_rating(p_solve=0.4, solved=True)
        assert delta == round(student.c_elo * (1 - 0.4))

    def test_rating_capped_at_max(self):
        s = StudentModel(initial_rating=3498)
        s._update_rating(p_solve=0.01, solved=True)   # delta grande
        assert s.rating <= 3500


# ---------------------------------------------------------------------------
# attempt() — integración completa
# ---------------------------------------------------------------------------

class TestAttempt:
    def test_returns_attempt_outcome(self, student):
        assert isinstance(student.attempt(1500, ["math"]), AttemptOutcome)

    def test_outcome_has_p_solve(self, student):
        outcome = student.attempt(1500, ["math"])
        assert 0.0 < outcome.p_solve < 1.0

    def test_outcome_has_reward(self, student):
        outcome = student.attempt(1500, ["math"])
        assert isinstance(outcome.reward, float)

    def test_reward_positive_if_solved(self):
        # Aseguramos que resuelve usando problema muy fácil y seed específica
        s = StudentModel(initial_rating=2400, session_budget_min=600, random_seed=1)
        outcome = s.attempt(800, ["math"])
        if outcome.solved:
            assert outcome.reward > 0

    def test_reward_negative_if_failed(self):
        # Problema imposible → casi siempre falla
        s = StudentModel(initial_rating=800, session_budget_min=600, random_seed=5)
        outcomes = [s.attempt(3500, ["math"]) for _ in range(10)]
        failures = [o for o in outcomes if not o.solved]
        if failures:
            assert all(o.reward == s.r_fracaso for o in failures)

    def test_time_min_positive(self, student):
        assert student.attempt(1500, ["math"]).time_min > 0.0

    def test_time_spent_increases(self, student):
        student.attempt(1500, ["math"])
        assert student.time_spent_min > 0.0

    def test_fatigue_increases(self, student):
        student.attempt(1500, ["math"])
        assert student.fatigue > 0.0

    def test_topics_seen_updated(self, student):
        student.attempt(1500, ["dp", "graphs"], "1A")
        assert "dp" in student.topics_seen
        assert "graphs" in student.topics_seen

    def test_raises_when_session_over(self, student):
        student.time_spent_min = 120.0
        with pytest.raises(RuntimeError, match="sesión ya terminó"):
            student.attempt(1500, ["math"])

    def test_failed_consumes_less_time(self):
        # T_fracaso = β * T_intento, que es < T_intento (con el mismo ε)
        # Verificamos que β < 1 implica que el tiempo de fracaso es menor
        s = StudentModel(initial_rating=800, session_budget_min=600, random_seed=7)
        t_base = s._solve_time_base(3500, ["math"])
        outcomes = [s.attempt(3500, ["math"]) for _ in range(5)]
        fails = [o for o in outcomes if not o.solved]
        if fails:
            # T_fracaso ≤ β * T_base * (1 + EPS) = 0.5 * t_base * 1.2
            assert all(o.time_min <= t_base * (s.beta + 0.3) for o in fails)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restores_all_state(self, student):
        student.attempt(1500, ["math"])
        student.reset()
        assert student.rating         == 1500
        assert student.fatigue        == 0.0
        assert student.time_spent_min == 0.0
        assert student.n_solved       == 0
        assert student.n_attempted    == 0
        assert len(student.topics_seen) == 0

    def test_topic_solves_reset(self, student_large):
        student_large.attempt(800, ["math"])
        student_large.reset()
        assert student_large._topic_solves["math"] == 0

    def test_can_attempt_after_reset(self, student):
        student.time_spent_min = 120.0
        student.reset()
        assert isinstance(student.attempt(1500, ["math"]), AttemptOutcome)


# ---------------------------------------------------------------------------
# state_vector y propiedades
# ---------------------------------------------------------------------------

class TestStateVector:
    def test_has_5_components(self, student):
        assert len(student.state_vector) == 5

    def test_all_in_0_1(self, student_large):
        for _ in range(5):
            student_large.attempt(1500, ["math"])
        for v in student_large.state_vector:
            assert 0.0 <= v <= 1.0

    def test_changes_after_attempt(self, student):
        before = student.state_vector[:]
        student.attempt(1500, ["math"])
        assert before != student.state_vector


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_outcomes(self):
        s1 = StudentModel(1500, session_budget_min=600, random_seed=7)
        s2 = StudentModel(1500, session_budget_min=600, random_seed=7)
        for _ in range(5):
            o1 = s1.attempt(1700, ["dp"])
            o2 = s2.attempt(1700, ["dp"])
            assert o1.solved   == o2.solved
            assert o1.time_min == o2.time_min