"""
student_model.py -- Modulo 5: Modelo del estudiante virtual

Cambios respecto a la version anterior
----------------------------------------
- Rating por tema (topic_ratings) en lugar de un unico rating global.
- R_efectivo para un problema = media de topic_ratings de sus tags.
- Probabilidad de exito basada en R_efectivo.
- Ganancia ELO por tema con factor de desafio:
      delta_R_t = C * (1 - P_t) * challenge_t
  donde challenge_t = sigma(gap_t / theta), garantizando que
  problemas faciles para el tema dan ganancias minimas.
- Rating global derivado del promedio ponderado de topic_ratings.

Formulas implementadas
-----------------------

Rating efectivo:
    R_ef = mean(topic_ratings[t] for t in tags del problema)
           Si el problema no tiene tags canonicos -> usa global_rating

Probabilidad de exito:
    P_exito = sigma((R_ef - d_p) / theta  -  lambda_f * F)

Tiempo base:
    f_dif   = clip((d_p - R_ef + delta0) / delta_max, 0, 1)
    g_temas = 1 + alpha * (n_temas - 1)
    T_base  = T_min + (T_max - T_min) * f_dif * g_temas
    T_real  = T_base * (1 + eps),  eps ~ U(-0.2, 0.2)
    T_fail  = beta * T_real

Actualizacion ELO por tema (solo si resuelve):
    Para cada tag t en tags del problema:
        gap_t       = d_p - topic_ratings[t]
        P_t         = sigma((topic_ratings[t] - d_p) / theta)
        challenge_t = sigma(gap_t / theta)
        delta_R_t   = C * (1 - P_t) * challenge_t
        topic_ratings[t] += delta_R_t

Rating global:
    peso[t] = intentos_con_tema[t] + 1
    global_rating = sum(topic_ratings[t] * peso[t]) / sum(pesos)

Recompensa:
    Si resuelve: r = r_exito + mean(delta_R_t para t en tags)
    Si fracasa:  r = r_fracaso
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from src.environment.problem import CANONICAL_TOPICS, N_TOPICS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parametros por defecto
# ---------------------------------------------------------------------------

_THETA          = 400.0
_LAMBDA_FATIGUE = 0.5
_T_MIN          = 5.0
_T_MAX          = 60.0
_DELTA0         = 200.0
_DELTA_MAX_TIME = 800.0
_ALPHA          = 0.2
_BETA           = 0.5
_EPS            = 0.2
_C_ELO          = 10.0
_R_EXITO        = 10.0
_R_FRACASO      = -2.0

_MIN_RATING     = 800
_MAX_RATING     = 3500
_DEFAULT_TOPIC_RATING = 1200


# ---------------------------------------------------------------------------
# AttemptOutcome
# ---------------------------------------------------------------------------

@dataclass
class AttemptOutcome:
    """Resultado completo de un intento sobre un problema.

    Attributes
    ----------
    solved              : bool
    time_min            : float
    p_solve             : float   -- probabilidad calculada antes del intento
    reward              : float   -- recompensa inmediata
    topic_deltas        : dict    -- {tag: delta_rating} para tags del problema
    new_global_rating   : float   -- rating global tras el intento
    fatigue             : float
    time_remaining_min  : float
    session_over        : bool
    """
    solved             : bool
    time_min           : float
    p_solve            : float
    reward             : float
    topic_deltas       : dict[str, float]
    new_global_rating  : float
    fatigue            : float
    time_remaining_min : float
    session_over       : bool


# ---------------------------------------------------------------------------
# StudentModel
# ---------------------------------------------------------------------------

class StudentModel:
    """Simula un estudiante con ELO independiente por tema.

    Parameters
    ----------
    topic_ratings       : dict[str, float] | None
        Rating inicial por tema canonico. Si None, todos empiezan en
        default_topic_rating.
    global_rating       : float | None
        Rating global inicial. Si None se calcula desde topic_ratings.
        Solo se usa como fallback cuando el problema no tiene tags canonicos.
    session_budget_min  : float
    random_seed         : int | None
    default_topic_rating: float
        Rating inicial para temas sin valor explicito (default 1200).
    theta               : float
    lambda_fatigue      : float
    fatigue_per_problem : float
    t_min, t_max        : float
    delta0, delta_max_time : float
    alpha, beta         : float
    c_elo               : float
    r_exito, r_fracaso  : float
    """

    def __init__(
        self,
        topic_ratings        : Optional[dict[str, float]] = None,
        global_rating        : Optional[float]            = None,
        session_budget_min   : float  = 120.0,
        random_seed          : Optional[int] = None,
        default_topic_rating : float  = _DEFAULT_TOPIC_RATING,
        theta                : float  = _THETA,
        lambda_fatigue       : float  = _LAMBDA_FATIGUE,
        fatigue_per_problem  : float  = 0.08,
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
        challenge_weight     : float  = 2.0,
        progression_weight   : float  = 1.5,
        trivial_penalty      : float  = -3.0,
    ) -> None:
        if session_budget_min <= 0:
            raise ValueError("session_budget_min debe ser positivo.")

        self.session_budget_min   = session_budget_min
        self.default_topic_rating = default_topic_rating
        self.theta                = theta
        self.lambda_fatigue       = lambda_fatigue
        self.fatigue_per_problem  = fatigue_per_problem
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
        self._rng                 = random.Random(random_seed)

        # Inicializar topic_ratings
        self._initial_topic_ratings: dict[str, float] = {}
        for topic in CANONICAL_TOPICS:
            if topic_ratings and topic in topic_ratings:
                val = float(topic_ratings[topic])
            else:
                val = default_topic_rating
            val = max(_MIN_RATING, min(_MAX_RATING, val))
            self._initial_topic_ratings[topic] = val

        # Calcular global_rating inicial
        if global_rating is not None:
            self._initial_global_rating = float(global_rating)
        else:
            self._initial_global_rating = float(
                sum(self._initial_topic_ratings.values()) / N_TOPICS
            )

        # Estado de sesion (se resetea en reset())
        self.topic_ratings        : dict[str, float] = {}
        self.global_rating        : float             = 0.0
        self.fatigue              : float             = 0.0
        self.time_spent_min       : float             = 0.0
        self.problems_solved      : list[str]         = []
        self.problems_attempted   : list[str]         = []
        self.topics_seen          : set[str]          = set()
        self._topic_attempts      : dict[str, int]    = {}
        self._last_problem_rating : float             = 0.0

        self.reset()

    # ------------------------------------------------------------------
    # API publica principal
    # ------------------------------------------------------------------

    def attempt(
        self,
        problem_rating : int,
        problem_tags   : list[str],
        problem_id     : str = "",
    ) -> AttemptOutcome:
        """Simula el intento del estudiante en un problema."""
        if self.session_over:
            raise RuntimeError(
                "La sesion ya termino. Llama a reset() para iniciar una nueva."
            )

        # 1. Rating efectivo para este problema
        r_ef = self._effective_rating(problem_tags)

        # 2. Probabilidad de exito
        p_solve = self.probability_of_solving(problem_rating, problem_tags)

        # 3. Tiempo base
        t_base    = self._solve_time_base(problem_rating, r_ef, problem_tags)
        epsilon   = self._rng.uniform(-_EPS, _EPS)
        t_intento = t_base * (1.0 + epsilon)

        # 4. Resultado estocastico
        solved   = self._rng.random() <= p_solve
        time_min = t_intento if solved else self.beta * t_intento
        time_min = round(max(1.0, time_min), 1)

        # 5. Actualizar ELO por tema y calcular recompensa
        topic_deltas: dict[str, float] = {}
        if solved:
            topic_deltas = self._update_topic_ratings(problem_rating, problem_tags)
            self._update_global_rating()
            mean_delta = sum(topic_deltas.values()) / max(1, len(topic_deltas))

            # Bonus por tema nuevo (fomenta diversidad tematica)
            canonical_tags  = [t for t in problem_tags if t in self.topic_ratings]
            new_topics      = [t for t in canonical_tags if t not in self.topics_seen]
            topic_bonus     = self.r_topic_new if new_topics else 0.0

            # Bonus/penalizacion por nivel de reto adecuado
            challenge_bonus = self._compute_challenge_bonus(p_solve)

            # Bonus por progresion de dificultad (problema mas dificil que el anterior)
            progression_bonus = self._compute_progression_bonus(problem_rating)

            reward = self.r_exito + mean_delta + topic_bonus + challenge_bonus + progression_bonus
        else:
            reward = self.r_fracaso

        self._last_problem_rating = float(problem_rating)

        # 6. Actualizar estado de sesion
        self.time_spent_min += time_min
        self.fatigue         = min(1.0, self.fatigue + self.fatigue_per_problem)
        self.topics_seen.update(problem_tags)

        for tag in problem_tags:
            if tag in self._topic_attempts:
                self._topic_attempts[tag] += 1

        if problem_id:
            self.problems_attempted.append(problem_id)
            if solved:
                self.problems_solved.append(problem_id)

        logger.debug(
            f"attempt pid={problem_id!r} | r_ef={r_ef:.0f} | "
            f"d_p={problem_rating} | p={p_solve:.3f} | "
            f"solved={solved} | time={time_min:.1f}m | "
            f"reward={reward:.1f} | global={self.global_rating:.0f}"
        )

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
        """Reinicia el estado para un nuevo episodio."""
        self.topic_ratings      = dict(self._initial_topic_ratings)
        self.global_rating      = self._initial_global_rating
        self.fatigue            = 0.0
        self.time_spent_min     = 0.0
        self.problems_solved    = []
        self.problems_attempted = []
        self.topics_seen        = set()
        self._topic_attempts      = {t: 0 for t in CANONICAL_TOPICS}
        self._last_problem_rating = 0.0

    # ------------------------------------------------------------------
    # Formulas publicas (sin efecto de estado)
    # ------------------------------------------------------------------

    def probability_of_solving(
        self,
        problem_rating : int,
        problem_tags   : list[str],
    ) -> float:
        """P_exito = sigma((R_ef - d_p) / theta  -  lambda_f * F)"""
        r_ef = self._effective_rating(problem_tags)
        x    = (r_ef - problem_rating) / self.theta - self.lambda_fatigue * self.fatigue
        return round(self._sigmoid(x), 4)

    def estimate_solve_time(
        self,
        problem_rating : int,
        problem_tags   : list[str],
    ) -> float:
        """Tiempo estimado de resolucion en minutos (sin modificar estado)."""
        r_ef   = self._effective_rating(problem_tags)
        t_base = self._solve_time_base(problem_rating, r_ef, problem_tags)
        return round(t_base, 1)

    def will_fit_in_session(
        self, problem_rating: int, problem_tags: list[str]
    ) -> bool:
        return self.estimate_solve_time(problem_rating, problem_tags) <= self.time_remaining_min

    # ------------------------------------------------------------------
    # Propiedades de estado
    # ------------------------------------------------------------------

    @property
    def rating(self) -> float:
        """Alias de global_rating para compatibilidad con Problem y ObservationBuilder."""
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
    def state_vector(self) -> list[float]:
        """Vector de estado para el agente DQN.

        Returns
        -------
        list[float] de longitud 4 + N_TOPICS = 24:
            [global_rating_norm, fatigue, time_used_norm, solve_rate,
             topic_rating_norm_0, ..., topic_rating_norm_19]
        """
        r_range        = _MAX_RATING - _MIN_RATING
        global_norm    = (self.global_rating - _MIN_RATING) / r_range
        time_used_norm = self.time_spent_min / self.session_budget_min

        topic_norms = [
            (self.topic_ratings[t] - _MIN_RATING) / r_range
            for t in CANONICAL_TOPICS
        ]

        return [
            round(global_norm,    4),
            round(self.fatigue,   4),
            round(time_used_norm, 4),
            round(self.solve_rate, 4),
        ] + [round(v, 4) for v in topic_norms]

    # ------------------------------------------------------------------
    # Metodos privados
    # ------------------------------------------------------------------

    def _effective_rating(self, problem_tags: list[str]) -> float:
        """R_ef = media de topic_ratings para los tags canonicos del problema."""
        canonical = [t for t in problem_tags if t in self.topic_ratings]
        if not canonical:
            return self.global_rating
        return sum(self.topic_ratings[t] for t in canonical) / len(canonical)

    def _update_topic_ratings(
        self, problem_rating: int, problem_tags: list[str]
    ) -> dict[str, float]:
        """Actualiza ELO por tema con factor de desafio.

        Para cada tag t canonico del problema:
            gap_t       = d_p - topic_ratings[t]
            P_t         = sigma((topic_ratings[t] - d_p) / theta)
            challenge_t = sigma(gap_t / theta)
            delta_R_t   = C * (1 - P_t) * challenge_t
            topic_ratings[t] += delta_R_t

        Returns
        -------
        dict[str, float] -- {tag: delta aplicado}
        """
        deltas: dict[str, float] = {}
        for tag in problem_tags:
            if tag not in self.topic_ratings:
                continue
            r_t         = self.topic_ratings[tag]
            gap_t       = problem_rating - r_t
            p_t         = self._sigmoid((r_t - problem_rating) / self.theta)
            challenge_t = self._sigmoid(gap_t / self.theta)
            delta       = self.c_elo * (1.0 - p_t) * challenge_t
            delta       = round(delta, 3)

            new_val     = max(_MIN_RATING, min(_MAX_RATING, r_t + delta))
            self.topic_ratings[tag] = new_val
            deltas[tag] = delta

        return deltas

    def _update_global_rating(self) -> None:
        """global_rating = promedio ponderado por intentos por tema."""
        total_weight = 0.0
        weighted_sum = 0.0
        for t in CANONICAL_TOPICS:
            w             = self._topic_attempts.get(t, 0) + 1
            weighted_sum += self.topic_ratings[t] * w
            total_weight += w
        self.global_rating = weighted_sum / total_weight

    def _solve_time_base(
        self, problem_rating: int, r_ef: float, problem_tags: list[str]
    ) -> float:
        """T_base = T_min + (T_max - T_min) * f_dif * g_temas"""
        f_dif   = (problem_rating - r_ef + self.delta0) / self.delta_max_time
        f_dif   = max(0.0, min(1.0, f_dif))
        n_temas = max(1, len(problem_tags))
        g_temas = 1.0 + self.alpha * (n_temas - 1)
        return self.t_min + (self.t_max - self.t_min) * f_dif * g_temas

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _compute_challenge_bonus(self, p_solve: float) -> float:
        """Bonus por reto adecuado.

        - p_solve en [0.35, 0.75]: zona ideal de aprendizaje → bonus maximo
        - p_solve > 0.88: problema trivial → penalizacion
        - p_solve < 0.15: problema imposible → penalizacion leve
        """
        if p_solve > 0.80:
            return self.trivial_penalty_val     # muy facil
        if p_solve < 0.15:
            return -1.0                          # imposible
        # Funcion campana con maximo en p=0.55
        center    = 0.55
        half_span = 0.35
        dist      = abs(p_solve - center) / half_span
        return round(self.challenge_weight * max(0.0, 1.0 - dist), 3)

    def _compute_progression_bonus(self, problem_rating: int) -> float:
        """Bonus por progresion de dificultad.

        Premia cuando el problema actual es mas dificil que el anterior.
        Penaliza ligeramente si va muy para atras.
        """
        if self._last_problem_rating == 0.0:
            return 0.0   # primer problema, sin referencia
        delta_rating = problem_rating - self._last_problem_rating
        if delta_rating >= 200:
            return self.progression_weight       # gran salto hacia arriba
        elif delta_rating >= 0:
            return round(self.progression_weight * delta_rating / 200, 3)
        else:
            # Regresion: penalizacion proporcional
            return round(max(-1.0, delta_rating / 400), 3)

    def __repr__(self) -> str:
        top3 = sorted(
            self.topic_ratings.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top3_str = ", ".join(f"{t}={r:.0f}" for t, r in top3)
        return (
            f"StudentModel(global={self.global_rating:.0f}, "
            f"top3=[{top3_str}], "
            f"fatigue={self.fatigue:.2f}, "
            f"solved={self.n_solved}/{self.n_attempted})"
        )