"""
student_model.py -- Modelo del estudiante virtual con ELO por tema

Formulas implementadas
-----------------------
Rating efectivo:
    R_ef = mean(topic_ratings[t] for t in tags del problema)

Probabilidad de exito:
    P_exito = sigma((R_ef - d_p) / theta  -  lambda_f * F)

Tiempo de resolucion:
    T_think   = T_min + (T_max - T_min) * f_dif * g_temas
    T_read    = T_READ_BASE * (1 + 0.2*(n_temas-1))
    T_code    = 3 * (rating/1600) * (1 + 0.15*(n_temas-1))
    T_debug   = 0.25 * T_code
    T_total   = T_think + T_read + T_code + T_debug

Actualizacion ELO por tema (solo si resuelve):
    gap_t       = d_p - topic_ratings[t]
    P_t         = sigma((topic_ratings[t] - d_p) / theta)
    challenge_t = sigma(gap_t / theta)
    delta_R_t   = C * (1 - P_t) * challenge_t

Fatiga:
    fatiga += time_min * fatigue_per_minute   (proporcional al tiempo invertido)

Recompensa:
    Si resuelve: r = r_exito + mean(delta_R_t) + topic_bonus + challenge_bonus + progression_bonus
    Si fracasa:  r = r_fracaso
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

from src.environment.problem import CANONICAL_TOPICS, N_TOPICS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parametros por defecto
# ---------------------------------------------------------------------------

_THETA          = 400.0
_LAMBDA_FATIGUE = 0.5
_T_MIN          = 5.0
_T_MAX          = 65.0
_T_READ_BASE    = 3.0
_DELTA0         = 200.0
_DELTA_MAX_TIME = 800.0
_ALPHA          = 0.2
_BETA           = 0.6
_EPS            = 0.25
_C_ELO          = 10.0
_R_EXITO        = 10.0
_R_FRACASO      = -2.0
_MIN_RATING     = 800
_MAX_RATING     = 3500
_DEFAULT_TOPIC_RATING = 1200

# Tags meta (no algoritmicos): aparecen en ~60-70% de problemas de Codeforces
# Se excluyen de diversidad, repeticion y topic_bonus para no distorsionar el aprendizaje
META_TAGS = frozenset(["implementation", "brute force"])


# ---------------------------------------------------------------------------
# AttemptOutcome
# ---------------------------------------------------------------------------

@dataclass
class AttemptOutcome:
    """Resultado completo de un intento sobre un problema."""
    solved             : bool
    time_min           : float
    p_solve            : float
    reward             : float
    topic_deltas       : dict
    new_global_rating  : float
    fatigue            : float
    time_remaining_min : float
    session_over       : bool


# ---------------------------------------------------------------------------
# StudentModel
# ---------------------------------------------------------------------------

class StudentModel:
    """Simula un estudiante con ELO independiente por tema."""

    def __init__(
        self,
        topic_ratings        : Optional[dict] = None,
        global_rating        : Optional[float] = None,
        session_budget_min   : float  = 120.0,
        random_seed          : Optional[int] = None,
        default_topic_rating : float  = _DEFAULT_TOPIC_RATING,
        theta                : float  = _THETA,
        lambda_fatigue       : float  = _LAMBDA_FATIGUE,
        fatigue_per_minute   : float  = 0.012,
        t_min                : float  = _T_MIN,
        t_max                : float  = _T_MAX,
        delta0               : float  = _DELTA0,
        delta_max_time       : float  = _DELTA_MAX_TIME,
        alpha                : float  = _ALPHA,
        beta                 : float  = _BETA,
        c_elo                : float  = _C_ELO,
        r_exito              : float  = _R_EXITO,
        r_fracaso            : float  = _R_FRACASO,
        r_topic_new          : float  = 2.0,
        challenge_weight     : float  = 3.0,
        progression_weight   : float  = 2.0,
        trivial_penalty      : float  = -4.0,
        stagnation_penalty   : float  = -2.0,
        stagnation_window    : int    = 3,
        topic_overuse_penalty: float  = -6.0,
        topic_overuse_thresh : float  = 0.30,
        topic_diversity_bonus: float  =  1.5,
    ) -> None:
        if session_budget_min <= 0:
            raise ValueError("session_budget_min debe ser positivo.")

        self.session_budget_min   = session_budget_min
        self.default_topic_rating = default_topic_rating
        self.theta                = theta
        self.lambda_fatigue       = lambda_fatigue
        self.fatigue_per_minute   = fatigue_per_minute
        self.t_min                = t_min
        self.t_max                = t_max
        self.delta0               = delta0
        self.delta_max_time       = delta_max_time
        self.alpha                = alpha
        self.beta                 = beta
        self.c_elo                = c_elo
        self.r_exito              = r_exito
        self.r_fracaso            = r_fracaso
        self.r_topic_new          = r_topic_new
        self.challenge_weight     = challenge_weight
        self.progression_weight   = progression_weight
        self.trivial_penalty_val  = trivial_penalty
        self.stagnation_penalty    = stagnation_penalty
        self.stagnation_window     = stagnation_window
        self.topic_overuse_penalty = topic_overuse_penalty  # -6.0
        self.topic_overuse_thresh  = topic_overuse_thresh
        self.topic_diversity_bonus = topic_diversity_bonus
        self._rng                  = random.Random(random_seed)

        # Inicializar topic_ratings
        self._initial_topic_ratings: dict[str, float] = {}
        for topic in CANONICAL_TOPICS:
            val = float(topic_ratings[topic]) if topic_ratings and topic in topic_ratings \
                  else default_topic_rating
            self._initial_topic_ratings[topic] = max(_MIN_RATING, min(_MAX_RATING, val))

        self._initial_global_rating = float(global_rating) if global_rating is not None \
            else sum(self._initial_topic_ratings.values()) / N_TOPICS

        # Estado de sesion
        self.topic_ratings        : dict[str, float] = {}
        self.global_rating        : float             = 0.0
        self.fatigue              : float             = 0.0
        self.time_spent_min       : float             = 0.0
        self.problems_solved      : list[str]         = []
        self.problems_attempted   : list[str]         = []
        self.topics_seen          : set[str]          = set()
        self._topic_attempts      : dict[str, int]    = {}
        self._last_problem_rating : float             = 0.0
        self._recent_ratings      : list[float]       = []  # ultimos N ratings

        self.reset()

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def attempt(
        self,
        problem_rating : int,
        problem_tags   : list[str],
        problem_id     : str = "",
    ) -> AttemptOutcome:
        if self.session_over:
            raise RuntimeError("La sesion ya termino. Llama a reset().")

        r_ef    = self._effective_rating(problem_tags)
        p_solve = self.probability_of_solving(problem_rating, problem_tags)
        t_base  = self._solve_time_base(problem_rating, r_ef, problem_tags)
        epsilon = self._rng.uniform(-_EPS, _EPS)
        t_real  = t_base * (1.0 + epsilon)
        solved  = self._rng.random() <= p_solve
        time_min = t_real if solved else self.beta * t_real
        time_min = round(max(1.0, time_min), 1)

        # Actualizar ELO y calcular recompensa
        topic_deltas: dict = {}
        if solved:
            topic_deltas = self._update_topic_ratings(problem_rating, problem_tags)
            self._update_global_rating()
            mean_delta = sum(topic_deltas.values()) / max(1, len(topic_deltas))

            # Bonus por tema nuevo
            canonical_tags = [t for t in problem_tags
                              if t in self.topic_ratings and t not in META_TAGS]
            new_topics     = [t for t in canonical_tags if t not in self.topics_seen]
            topic_bonus    = self.r_topic_new if new_topics else 0.0

            # Bonus/penalizacion por nivel de reto (basado en gap absoluto)
            challenge_bonus = self._compute_challenge_bonus(problem_rating)

            # Bonus por progresion de dificultad
            progression_bonus = self._compute_progression_bonus(problem_rating)

            # Penalizacion por estancamiento (muchos faciles seguidos)
            stagnation_pen = self._compute_stagnation_penalty(problem_rating)

            # Penalizacion por repeticion excesiva de temas
            repetition_pen = self._compute_topic_repetition_penalty(problem_tags)

            reward = self.r_exito + mean_delta + topic_bonus + challenge_bonus \
                     + progression_bonus + stagnation_pen + repetition_pen
        else:
            reward = self.r_fracaso

        # Actualizar estado
        self.time_spent_min   += time_min
        self.fatigue           = min(1.0, self.fatigue + time_min * self.fatigue_per_minute)
        self.topics_seen.update(problem_tags)
        self._last_problem_rating = float(problem_rating)
        self._recent_ratings.append(float(problem_rating))
        if len(self._recent_ratings) > self.stagnation_window:
            self._recent_ratings.pop(0)

        for tag in problem_tags:
            if tag in self._topic_attempts:
                self._topic_attempts[tag] += 1

        if problem_id:
            self.problems_attempted.append(problem_id)
            if solved:
                self.problems_solved.append(problem_id)

        return AttemptOutcome(
            solved             = solved,
            time_min           = time_min,
            p_solve            = round(p_solve, 4),
            reward             = round(reward, 2),
            topic_deltas       = topic_deltas,
            new_global_rating  = round(self.global_rating, 1),
            fatigue            = round(self.fatigue, 4),
            time_remaining_min = round(self.time_remaining_min, 1),
            session_over       = self.session_over,
        )

    def reset(self) -> None:
        self.topic_ratings        = dict(self._initial_topic_ratings)
        self.global_rating        = self._initial_global_rating
        self.fatigue              = 0.0
        self.time_spent_min       = 0.0
        self.problems_solved      = []
        self.problems_attempted   = []
        self.topics_seen          = set()
        self._topic_attempts      = {t: 0 for t in CANONICAL_TOPICS}
        self._last_problem_rating = 0.0
        self._recent_ratings      = []

    # ------------------------------------------------------------------
    # Formulas publicas (solo consulta, sin efectos de estado)
    # ------------------------------------------------------------------

    def probability_of_solving(self, problem_rating: int, problem_tags: list) -> float:
        r_ef = self._effective_rating(problem_tags)
        x    = (r_ef - problem_rating) / self.theta - self.lambda_fatigue * self.fatigue
        return round(self._sigmoid(x), 4)

    def estimate_solve_time(self, problem_rating: int, problem_tags: list) -> float:
        r_ef = self._effective_rating(problem_tags)
        return round(self._solve_time_base(problem_rating, r_ef, problem_tags), 1)

    def will_fit_in_session(self, problem_rating: int, problem_tags: list) -> bool:
        return self.estimate_solve_time(problem_rating, problem_tags) <= self.time_remaining_min

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def rating(self) -> float:
        return self.global_rating

    @property
    def time_remaining_min(self) -> float:
        return max(0.0, self.session_budget_min - self.time_spent_min)

    @property
    def session_over(self) -> bool:
        return self.time_remaining_min <= 0.0

    @property
    def n_solved(self) -> int:
        return len(self.problems_solved)

    @property
    def n_attempted(self) -> int:
        return len(self.problems_attempted)

    @property
    def solve_rate(self) -> float:
        return self.n_solved / self.n_attempted if self.n_attempted else 0.0

    @property
    def state_vector(self) -> list:
        """Vector de estado de 24 componentes para el agente DQN."""
        r_range        = _MAX_RATING - _MIN_RATING
        global_norm    = (self.global_rating - _MIN_RATING) / r_range
        time_used_norm = self.time_spent_min / self.session_budget_min
        topic_norms    = [
            (self.topic_ratings[t] - _MIN_RATING) / r_range
            for t in CANONICAL_TOPICS
        ]
        return [
            round(global_norm,     4),
            round(self.fatigue,    4),
            round(time_used_norm,  4),
            round(self.solve_rate, 4),
        ] + [round(v, 4) for v in topic_norms]

    # ------------------------------------------------------------------
    # Calculos de recompensa
    # ------------------------------------------------------------------

    def _compute_challenge_bonus(self, problem_rating: int) -> float:
        """Bonus basado en gap absoluto respecto al rating global del estudiante.

        gap < -300 : trivial -> penalizacion fuerte (independiente de fatiga)
        gap  0-200 : zona ideal de desarrollo -> bonus maximo
        gap  > 600 : imposible -> penalizacion leve
        """
        gap = problem_rating - self.global_rating

        if gap < -300:
            return self.trivial_penalty_val  # siempre -4.0, sin importar fatiga

        if gap > 600:
            return -1.5  # demasiado dificil

        if gap >= 0:
            if gap <= 200:
                # Zona ideal: bonus crece linealmente hasta gap=200
                return round(self.challenge_weight * (gap / 200.0), 3)
            else:
                # Sigue siendo util pero decrece
                return round(self.challenge_weight * max(0.0, 1.0 - (gap - 200) / 400.0), 3)
        else:
            # gap en [-300, 0]: algo facil, bonus pequeno
            return round(self.challenge_weight * 0.2 * (1.0 + gap / 300.0), 3)

    def _compute_progression_bonus(self, problem_rating: int) -> float:
        """Bonus por escoger problema mas dificil que el anterior."""
        if self._last_problem_rating == 0.0:
            return 0.0
        delta = problem_rating - self._last_problem_rating
        if delta >= 300:
            return self.progression_weight          # gran salto
        elif delta >= 0:
            return round(self.progression_weight * delta / 300.0, 3)
        else:
            return round(max(-1.0, delta / 500.0), 3)  # regresion leve

    def _compute_topic_repetition_penalty(self, problem_tags: list) -> float:
        """Penaliza cuando un tema aparece demasiado frecuentemente.

        La penalizacion escala de forma no lineal: cuanto mas por encima del
        umbral este la frecuencia, mucho mayor es la penalizacion.

        Umbral: 30% (si un tema ocupa mas del 30% de los intentos -> penalizar)
        Maximo: topic_overuse_penalty (default -3.0)
        """
        if self.n_attempted < 3:
            return 0.0

        canonical = [t for t in problem_tags
                     if t in self._topic_attempts and t not in META_TAGS]
        if not canonical:
            return 0.0  # solo meta-tags: sin penalizacion

        total_pen = 0.0
        for tag in canonical:
            freq = self._topic_attempts.get(tag, 0) / self.n_attempted
            if freq > self.topic_overuse_thresh:
                excess    = (freq - self.topic_overuse_thresh) / (1.0 - self.topic_overuse_thresh)
                total_pen += self.topic_overuse_penalty * excess

        return round(total_pen, 3)

    def _compute_diversity_bonus(self, problem_tags: list) -> float:
        """Bonus por elegir temas infrautilizados.

        Complementa la penalizacion por repeticion: mientras esta castiga
        los temas sobreusados, este bonus premia los temas poco explorados.

        Un tema con frecuencia 0% da el bonus maximo; uno en el umbral da 0.
        """
        if self.n_attempted < 3:
            return 0.0

        # Excluir meta-tags: implementation/brute force no son temas algoritmicos
        canonical = [t for t in problem_tags
                     if t in self._topic_attempts and t not in META_TAGS]
        if not canonical:
            return 0.0  # solo meta-tags: sin bonus

        total_bonus = 0.0
        for tag in canonical:
            freq = self._topic_attempts.get(tag, 0) / self.n_attempted
            if freq < self.topic_overuse_thresh:
                underuse     = (self.topic_overuse_thresh - freq) / self.topic_overuse_thresh
                total_bonus += self.topic_diversity_bonus * underuse

        return round(total_bonus / max(1, len(canonical)), 3)

    def _compute_stagnation_penalty(self, problem_rating: int) -> float:
        """Penalizacion si los ultimos N problemas incluyendo el actual son triviales.

        Solo aplica si el problema ACTUAL tambien es trivial (gap < -300).
        No penaliza cuando el estudiante intenta mejorar eligiendo algo mas dificil.
        """
        # El problema actual no es trivial -> no penalizar aunque los previos lo fueran
        if self.global_rating - problem_rating <= 300:
            return 0.0

        if len(self._recent_ratings) < self.stagnation_window - 1:
            return 0.0

        trivial_prev = sum(
            1 for r in self._recent_ratings[-(self.stagnation_window - 1):]
            if self.global_rating - r > 300
        )
        if trivial_prev >= self.stagnation_window - 1:
            return self.stagnation_penalty  # todos triviales incluyendo el actual
        return 0.0

    # ------------------------------------------------------------------
    # Metodos privados
    # ------------------------------------------------------------------

    def _effective_rating(self, problem_tags: list) -> float:
        canonical = [t for t in problem_tags if t in self.topic_ratings]
        if not canonical:
            return self.global_rating
        return sum(self.topic_ratings[t] for t in canonical) / len(canonical)

    def _update_topic_ratings(self, problem_rating: int, problem_tags: list) -> dict:
        deltas: dict = {}
        for tag in problem_tags:
            if tag not in self.topic_ratings:
                continue
            r_t         = self.topic_ratings[tag]
            gap_t       = problem_rating - r_t
            p_t         = self._sigmoid((r_t - problem_rating) / self.theta)
            challenge_t = self._sigmoid(gap_t / self.theta)
            delta       = round(self.c_elo * (1.0 - p_t) * challenge_t, 3)
            self.topic_ratings[tag] = max(_MIN_RATING, min(_MAX_RATING, r_t + delta))
            deltas[tag] = delta
        return deltas

    def _update_global_rating(self) -> None:
        total_w = 0.0
        w_sum   = 0.0
        for t in CANONICAL_TOPICS:
            w       = self._topic_attempts.get(t, 0) + 1
            w_sum  += self.topic_ratings[t] * w
            total_w += w
        self.global_rating = w_sum / total_w

    def _solve_time_base(self, problem_rating: int, r_ef: float, problem_tags: list) -> float:
        """T_total = T_think + T_read + T_code + T_debug"""
        n_temas = max(1, len(problem_tags))

        # T_think: ocurrirsele la solucion (puede ser 5 min para trivial)
        f_dif   = (problem_rating - r_ef + self.delta0) / self.delta_max_time
        f_dif   = max(0.0, min(1.0, f_dif))
        g_temas = 1.0 + self.alpha * (n_temas - 1)
        t_think = self.t_min + (self.t_max - self.t_min) * f_dif * g_temas

        # T_read: leer y entender el enunciado (siempre presente)
        t_read  = _T_READ_BASE * (1.0 + 0.2 * (n_temas - 1))

        # T_code: escribir la implementacion (escala con dificultad absoluta)
        t_code  = 3.0 * (problem_rating / 1600.0) * (1.0 + 0.15 * (n_temas - 1))

        # T_debug: testear y corregir (~25% del tiempo de codeo)
        t_debug = 0.25 * t_code

        return round(t_think + t_read + t_code + t_debug, 2)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def __repr__(self) -> str:
        top3 = sorted(self.topic_ratings.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"{t}={r:.0f}" for t, r in top3)
        return (
            f"StudentModel(global={self.global_rating:.0f}, "
            f"top3=[{top3_str}], "
            f"fatigue={self.fatigue:.2f}, "
            f"solved={self.n_solved}/{self.n_attempted})"
        )