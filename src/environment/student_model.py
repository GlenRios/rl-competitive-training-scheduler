"""
student_model.py -- Modelo del estudiante virtual con ELO por tema

Cambios v3 (basados en analisis comparativo DQN vs baselines):
    1. c_elo reducido a 3.0 (aprendizaje mas gradual, menos volatilidad)
    2. delta ELO limitado a [-15, +30] por problema
    3. factor de fatiga en la ganancia ELO (fatigado -> aprende menos)
    4. _effective_rating usa minimo de ELOs (no media) -- el eslabon mas debil limita
    5. lambda_fatigue aumentado a 1.2 (fatiga afecta mas al exito)
    6. fatigue_per_minute aumentado a 0.020
    7. fatigue_mult en t_think (fatigado -> resuelve mas lento)
    8. pesos de recompensa aumentados: progression=5.0, challenge=4.0
    9. stagnation_penalty=-6.0, topic_overuse_penalty=-10.0
    10. recompensa final por eficiencia de tiempo en env.py
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
_LAMBDA_FATIGUE = 1.2      # aumentado de 0.5 a 1.2
_T_MIN          = 5.0
_T_MAX          = 65.0
_T_READ_BASE    = 3.0
_DELTA0         = 200.0
_DELTA_MAX_TIME = 800.0
_ALPHA          = 0.2
_BETA           = 0.6
_EPS            = 0.25
_C_ELO          = 3.0      # reducido de 10.0 a 3.0
_R_EXITO        = 10.0
_R_FRACASO      = -2.0
_MIN_RATING     = 800
_MAX_RATING     = 3500
_DEFAULT_TOPIC_RATING = 1200

# Tags meta (no algoritmicos): se excluyen de diversidad y repeticion
META_TAGS = frozenset(["implementation", "brute force"])


# ---------------------------------------------------------------------------
# AttemptOutcome
# ---------------------------------------------------------------------------

@dataclass
class AttemptOutcome:
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

    def __init__(
        self,
        topic_ratings        : Optional[dict] = None,
        global_rating        : Optional[float] = None,
        session_budget_min   : float  = 120.0,
        random_seed          : Optional[int] = None,
        default_topic_rating : float  = _DEFAULT_TOPIC_RATING,
        theta                : float  = _THETA,
        lambda_fatigue       : float  = _LAMBDA_FATIGUE,
        fatigue_per_minute   : float  = 0.020,
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
        challenge_weight     : float  = 4.0,
        progression_weight   : float  = 5.0,
        trivial_penalty      : float  = -4.0,
        stagnation_penalty   : float  = -6.0,
        stagnation_window    : int    = 3,
        topic_overuse_penalty: float  = -10.0,
        topic_overuse_thresh : float  = 0.30,
        topic_diversity_bonus: float  =  1.5,
    ) -> None:
        if session_budget_min <= 0:
            raise ValueError("session_budget_min debe ser positivo.")

        self.session_budget_min    = session_budget_min
        self.default_topic_rating  = default_topic_rating
        self.theta                 = theta
        self.lambda_fatigue        = lambda_fatigue
        self.fatigue_per_minute    = fatigue_per_minute
        self.t_min                 = t_min
        self.t_max                 = t_max
        self.delta0                = delta0
        self.delta_max_time        = delta_max_time
        self.alpha                 = alpha
        self.beta                  = beta
        self.c_elo                 = c_elo
        self.r_exito               = r_exito
        self.r_fracaso             = r_fracaso
        self.r_topic_new           = r_topic_new
        self.challenge_weight      = challenge_weight
        self.progression_weight    = progression_weight
        self.trivial_penalty_val   = trivial_penalty
        self.stagnation_penalty    = stagnation_penalty
        self.stagnation_window     = stagnation_window
        self.topic_overuse_penalty = topic_overuse_penalty
        self.topic_overuse_thresh  = topic_overuse_thresh
        self.topic_diversity_bonus = topic_diversity_bonus
        self._rng                  = random.Random(random_seed)

        self._initial_topic_ratings: dict[str, float] = {}
        for topic in CANONICAL_TOPICS:
            val = float(topic_ratings[topic]) if topic_ratings and topic in topic_ratings \
                  else default_topic_rating
            self._initial_topic_ratings[topic] = max(_MIN_RATING, min(_MAX_RATING, val))

        self._initial_global_rating = float(global_rating) if global_rating is not None \
            else sum(self._initial_topic_ratings.values()) / N_TOPICS

        self.topic_ratings        : dict[str, float] = {}
        self.global_rating        : float             = 0.0
        self.fatigue              : float             = 0.0
        self.time_spent_min       : float             = 0.0
        self.problems_solved      : list[str]         = []
        self.problems_attempted   : list[str]         = []
        self.topics_seen          : set[str]          = set()
        self._topic_attempts      : dict[str, int]    = {}
        self._last_problem_rating : float             = 0.0
        self._recent_ratings      : list[float]       = []
        # Para limitar caidas catastroficas de rating global
        self._session_start_rating: float             = 0.0

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

        topic_deltas: dict = {}
        if solved:
            topic_deltas = self._update_topic_ratings(problem_rating, problem_tags)
            self._update_global_rating()
            mean_delta = sum(topic_deltas.values()) / max(1, len(topic_deltas))

            # Bonus por tema nuevo (excluir meta-tags)
            canonical_tags = [t for t in problem_tags
                              if t in self.topic_ratings and t not in META_TAGS]
            new_topics     = [t for t in canonical_tags if t not in self.topics_seen]
            topic_bonus    = self.r_topic_new if new_topics else 0.0

            challenge_bonus   = self._compute_challenge_bonus(problem_rating)
            progression_bonus = self._compute_progression_bonus(problem_rating)
            stagnation_pen    = self._compute_stagnation_penalty(problem_rating)
            repetition_pen    = self._compute_topic_repetition_penalty(problem_tags)
            diversity_bonus   = self._compute_diversity_bonus(problem_tags)

            reward = (self.r_exito + mean_delta + topic_bonus + challenge_bonus
                      + progression_bonus + stagnation_pen + repetition_pen + diversity_bonus)
        else:
            reward = self.r_fracaso

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
        self.topic_ratings         = dict(self._initial_topic_ratings)
        self.global_rating         = self._initial_global_rating
        self._session_start_rating = self._initial_global_rating
        self.fatigue               = 0.0
        self.time_spent_min        = 0.0
        self.problems_solved       = []
        self.problems_attempted    = []
        self.topics_seen           = set()
        self._topic_attempts       = {t: 0 for t in CANONICAL_TOPICS}
        self._last_problem_rating  = 0.0
        self._recent_ratings       = []

    # ------------------------------------------------------------------
    # Formulas publicas
    # ------------------------------------------------------------------

    def probability_of_solving(self, problem_rating: int, problem_tags: list) -> float:
        """P = sigma((R_ef - d_p) / theta - lambda * F)
        Usa el MINIMO de ELOs por tema (no la media).
        """
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
    # Calculo de recompensa
    # ------------------------------------------------------------------

    def _compute_challenge_bonus(self, problem_rating: int) -> float:
        gap = problem_rating - self.global_rating
        if gap < -300:
            return self.trivial_penalty_val
        if gap > 600:
            return -1.5
        if gap >= 0:
            if gap <= 200:
                return round(self.challenge_weight * (gap / 200.0), 3)
            else:
                return round(self.challenge_weight * max(0.0, 1.0 - (gap - 200) / 400.0), 3)
        else:
            return round(self.challenge_weight * 0.2 * (1.0 + gap / 300.0), 3)

    def _compute_progression_bonus(self, problem_rating: int) -> float:
        if self._last_problem_rating == 0.0:
            return 0.0
        delta = problem_rating - self._last_problem_rating
        if delta >= 300:
            return self.progression_weight
        elif delta >= 0:
            return round(self.progression_weight * delta / 300.0, 3)
        else:
            return round(max(-2.0, delta / 300.0), 3)

    def _compute_stagnation_penalty(self, problem_rating: int) -> float:
        if self.global_rating - problem_rating <= 300:
            return 0.0
        if len(self._recent_ratings) < self.stagnation_window - 1:
            return 0.0
        trivial_prev = sum(
            1 for r in self._recent_ratings[-(self.stagnation_window - 1):]
            if self.global_rating - r > 300
        )
        if trivial_prev >= self.stagnation_window - 1:
            return self.stagnation_penalty
        return 0.0

    def _compute_topic_repetition_penalty(self, problem_tags: list) -> float:
        if self.n_attempted < 3:
            return 0.0
        canonical = [t for t in problem_tags
                     if t in self._topic_attempts and t not in META_TAGS]
        if not canonical:
            return 0.0
        total_pen = 0.0
        for tag in canonical:
            freq = self._topic_attempts.get(tag, 0) / self.n_attempted
            if freq > self.topic_overuse_thresh:
                excess    = (freq - self.topic_overuse_thresh) / (1.0 - self.topic_overuse_thresh)
                total_pen += self.topic_overuse_penalty * excess
        return round(total_pen, 3)

    def _compute_diversity_bonus(self, problem_tags: list) -> float:
        if self.n_attempted < 3:
            return 0.0
        canonical = [t for t in problem_tags
                     if t in self._topic_attempts and t not in META_TAGS]
        if not canonical:
            return 0.0
        total_bonus = 0.0
        for tag in canonical:
            freq = self._topic_attempts.get(tag, 0) / self.n_attempted
            if freq < self.topic_overuse_thresh:
                underuse     = (self.topic_overuse_thresh - freq) / self.topic_overuse_thresh
                total_bonus += self.topic_diversity_bonus * underuse
        return round(total_bonus / max(1, len(canonical)), 3)

    # ------------------------------------------------------------------
    # Formulas internas
    # ------------------------------------------------------------------

    def _effective_rating(self, problem_tags: list) -> float:
        """Usa el MINIMO de topic_ratings -- el eslabon mas debil limita el exito."""
        canonical = [t for t in problem_tags if t in self.topic_ratings]
        if not canonical:
            return self.global_rating
        return min(self.topic_ratings[t] for t in canonical)

    def _update_topic_ratings(self, problem_rating: int, problem_tags: list) -> dict:
        """ELO por tema con:
        - factor de fatiga (fatigado aprende menos)
        - delta limitado a [-15, +30]
        """
        deltas: dict = {}
        fatigue_factor = max(0.3, 1.0 - 0.5 * self.fatigue)  # min 0.3 para no anular el aprendizaje

        for tag in problem_tags:
            if tag not in self.topic_ratings:
                continue
            r_t         = self.topic_ratings[tag]
            gap_t       = problem_rating - r_t
            p_t         = self._sigmoid((r_t - problem_rating) / self.theta)
            challenge_t = self._sigmoid(gap_t / self.theta)
            delta       = self.c_elo * (1.0 - p_t) * challenge_t * fatigue_factor
            delta       = max(-15.0, min(30.0, delta))   # limite [-15, +30]
            delta       = round(delta, 3)

            self.topic_ratings[tag] = max(_MIN_RATING, min(_MAX_RATING, r_t + delta))
            deltas[tag] = delta
        return deltas

    def _update_global_rating(self) -> None:
        """Promedio ponderado con suelo: no puede caer mas de 50 pts respecto al inicio."""
        total_w = 0.0
        w_sum   = 0.0
        for t in CANONICAL_TOPICS:
            w       = self._topic_attempts.get(t, 0) + 1
            w_sum  += self.topic_ratings[t] * w
            total_w += w
        new_global = w_sum / total_w
        # Limitar caida catastrofica: max -50 puntos respecto al inicio de sesion
        floor = self._session_start_rating - 50.0
        self.global_rating = max(floor, new_global)

    def _solve_time_base(self, problem_rating: int, r_ef: float, problem_tags: list) -> float:
        """T_total = T_think * fatigue_mult + T_read + T_code + T_debug

        Cambio v3: T_think se multiplica por (1 + fatiga) -- fatigado resuelve mas lento.
        """
        n_temas = max(1, len(problem_tags))

        f_dif   = (problem_rating - r_ef + self.delta0) / self.delta_max_time
        f_dif   = max(0.0, min(1.0, f_dif))
        g_temas = 1.0 + self.alpha * (n_temas - 1)
        t_think_base = self.t_min + (self.t_max - self.t_min) * f_dif * g_temas

        # Fatiga incrementa el tiempo de pensar/resolver
        fatigue_mult = 1.0 + self.fatigue
        t_think      = t_think_base * fatigue_mult

        t_read  = _T_READ_BASE * (1.0 + 0.2 * (n_temas - 1))
        t_code  = 3.0 * (problem_rating / 1600.0) * (1.0 + 0.15 * (n_temas - 1))
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