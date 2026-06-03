"""
student_model.py — Módulo 5: Modelo del estudiante virtual

Fórmulas implementadas
----------------------

Probabilidad de éxito:
    P_éxito = σ((R_s - d_p) / θ + δ_topics - λ·F)
    σ(x) = 1 / (1 + e^(-x))

    R_s      : rating actual del estudiante
    d_p      : dificultad del problema (rating)
    θ        : parámetro de escala (default 400)
    δ_topics : bono por afinidad temática ∈ [-0.5, 0.5]
    λ        : factor de penalización por fatiga (default 0.5)
    F        : fatiga actual ∈ [0, 1]

Tiempo base de resolución:
    T_base = T_min + (T_max - T_min) · f_dificultad(p, s) · g_temas(p)
    f_dificultad = clip((d_p - R_s + Δ0) / Δ_max, 0, 1)
    g_temas      = 1 + α · (n_temas - 1)
    T_intento    = T_base · (1 + ε),   ε ~ U(-0.2, 0.2)
    T_fracaso    = β · T_intento

Actualización de rating (ELO simplificado):
    Si resuelve : ΔR = C · (1 - P_éxito)  → rating sube más si el problema era difícil
    Si fracasa  : ΔR = 0                   → sin cambio de rating

Recompensa inmediata:
    Si resuelve : r = r_éxito + ΔR
    Si fracasa  : r = r_fracaso

Uso
---
    from src.environment.student_model import StudentModel

    student = StudentModel(initial_rating=1500, session_budget_min=120)
    outcome = student.attempt(problem_rating=1600, problem_tags=["dp", "graphs"])

    print(outcome.solved)    # True / False
    print(outcome.reward)    # recompensa inmediata
    print(outcome.p_solve)   # probabilidad calculada
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros por defecto del modelo (todos configurables en __init__)
# ---------------------------------------------------------------------------

# Probabilidad
_THETA          = 400.0   # escala de sensibilidad al gap de rating
_LAMBDA_FATIGUE = 0.5     # penalización por fatiga
_DELTA_MAX      = 0.5     # bono máximo/mínimo de afinidad temática

# Tiempo
_T_MIN  = 5.0             # minutos mínimos de resolución
_T_MAX  = 60.0            # minutos máximos de resolución
_DELTA0 = 200.0           # desplazamiento del factor de dificultad
_DELTA_MAX_TIME = 800.0   # rango de referencia para f_dificultad
_ALPHA  = 0.2             # coeficiente de complejidad temática
_BETA   = 0.5             # fracción de tiempo consumida al fracasar
_EPS    = 0.2             # amplitud de estocasticidad en tiempo ± 20%

# Rating ELO
_C_ELO  = 10              # cambio máximo de rating por problema resuelto

# Recompensa
_R_EXITO   =  10.0        # recompensa fija por resolver
_R_FRACASO =  -2.0        # penalización fija por fallar

# Límites de rating válidos en Codeforces
_MIN_RATING = 800
_MAX_RATING = 3500

# Temas canónicos (alineados con Problem.CANONICAL_TOPICS)
from src.environment.problem import CANONICAL_TOPICS, N_TOPICS


# ---------------------------------------------------------------------------
# AttemptOutcome
# ---------------------------------------------------------------------------

@dataclass
class AttemptOutcome:
    """Resultado completo de un intento sobre un problema.

    Attributes
    ----------
    solved             : bool   — True si el estudiante lo resolvió
    time_min           : float  — minutos consumidos en el intento
    p_solve            : float  — probabilidad calculada antes del intento
    reward             : float  — recompensa inmediata (r_éxito+ΔR o r_fracaso)
    delta_rating       : int    — cambio de rating producido
    new_rating         : int    — rating del estudiante tras el intento
    fatigue            : float  — fatiga tras el intento [0, 1]
    time_remaining_min : float  — minutos restantes en la sesión
    session_over       : bool   — True si la sesión terminó tras el intento
    """
    solved             : bool
    time_min           : float
    p_solve            : float
    reward             : float
    delta_rating       : int
    new_rating         : int
    fatigue            : float
    time_remaining_min : float
    session_over       : bool


# ---------------------------------------------------------------------------
# StudentModel
# ---------------------------------------------------------------------------

class StudentModel:
    """Simula el comportamiento de un estudiante en una sesión de entrenamiento.

    Parameters
    ----------
    initial_rating      : int    — rating inicial (800–3500)
    session_budget_min  : float  — duración máxima de la sesión en minutos
    random_seed         : int | None — semilla para reproducibilidad
    theta               : float  — escala de sensibilidad al gap (default 400)
    lambda_fatigue      : float  — factor de penalización por fatiga (default 0.5)
    fatigue_per_problem : float  — incremento de fatiga por intento (default 0.08)
    t_min               : float  — tiempo mínimo de resolución en minutos
    t_max               : float  — tiempo máximo de resolución en minutos
    delta0              : float  — desplazamiento en f_dificultad
    delta_max_time      : float  — rango de referencia en f_dificultad
    alpha               : float  — coeficiente de complejidad temática
    beta                : float  — fracción de tiempo consumida al fracasar
    c_elo               : int    — cambio máximo de rating por problema resuelto
    r_exito             : float  — recompensa fija por resolver
    r_fracaso           : float  — penalización fija por fallar
    """

    def __init__(
        self,
        initial_rating      : int   = 1500,
        session_budget_min  : float = 120.0,
        random_seed         : Optional[int] = None,
        theta               : float = _THETA,
        lambda_fatigue      : float = _LAMBDA_FATIGUE,
        fatigue_per_problem : float = 0.08,
        t_min               : float = _T_MIN,
        t_max               : float = _T_MAX,
        delta0              : float = _DELTA0,
        delta_max_time      : float = _DELTA_MAX_TIME,
        alpha               : float = _ALPHA,
        beta                : float = _BETA,
        c_elo               : int   = _C_ELO,
        r_exito             : float = _R_EXITO,
        r_fracaso           : float = _R_FRACASO,
    ) -> None:
        if not (_MIN_RATING <= initial_rating <= _MAX_RATING):
            raise ValueError(
                f"initial_rating debe estar entre {_MIN_RATING} y {_MAX_RATING}. "
                f"Recibido: {initial_rating}"
            )
        if session_budget_min <= 0:
            raise ValueError("session_budget_min debe ser positivo.")

        # Perfil fijo del estudiante
        self.initial_rating     = initial_rating
        self.session_budget_min = session_budget_min

        # Hiperparámetros del modelo
        self.theta               = theta
        self.lambda_fatigue      = lambda_fatigue
        self.fatigue_per_problem = fatigue_per_problem
        self.t_min               = t_min
        self.t_max               = t_max
        self.delta0              = delta0
        self.delta_max_time      = delta_max_time
        self.alpha               = alpha
        self.beta                = beta
        self.c_elo               = c_elo
        self.r_exito             = r_exito
        self.r_fracaso           = r_fracaso

        self._rng = random.Random(random_seed)

        # Estado de sesión
        self.rating              : int          = initial_rating
        self.fatigue             : float        = 0.0
        self.time_spent_min      : float        = 0.0
        self.problems_solved     : list[str]    = []
        self.problems_attempted  : list[str]    = []
        self.topics_seen         : set[str]     = set()

        # Maestría por tema: cuántas veces ha resuelto problemas de cada tema
        # Se normaliza al calcular δ_topics
        self._topic_solves       : dict[str, int] = {t: 0 for t in CANONICAL_TOPICS}

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    def attempt(
        self,
        problem_rating  : int,
        problem_tags    : list[str],
        problem_id      : str = "",
    ) -> AttemptOutcome:
        """Simula el intento del estudiante en un problema.

        Parameters
        ----------
        problem_rating : int
            Rating del problema (dificultad d_p).
        problem_tags : list[str]
            Lista de temas del problema.
        problem_id : str
            Identificador opcional para registro.

        Returns
        -------
        AttemptOutcome con todos los resultados del intento.
        """
        if self.session_over:
            raise RuntimeError(
                "La sesión ya terminó. Llama a reset() para iniciar una nueva."
            )

        # 1. Calcular δ_topics (afinidad temática)
        delta_topics = self._topic_affinity(problem_tags)

        # 2. Probabilidad de éxito con la fórmula sigmoid
        p_solve = self.probability_of_solving(
            problem_rating=problem_rating,
            problem_tags=problem_tags,
        )

        # 3. Tiempo base de resolución
        t_base = self._solve_time_base(problem_rating, problem_tags)

        # 4. Tiempo real con estocasticidad: T_intento = T_base · (1 + ε)
        epsilon  = self._rng.uniform(-_EPS, _EPS)
        t_intento = t_base * (1.0 + epsilon)

        # 5. Resultado estocástico
        solved = self._rng.random() <= p_solve

        # 6. Tiempo consumido
        time_min = t_intento if solved else self.beta * t_intento
        time_min = round(max(1.0, time_min), 1)

        # 7. Actualizar rating y calcular recompensa
        delta_rating = self._update_rating(p_solve, solved)
        reward       = (self.r_exito + delta_rating) if solved else self.r_fracaso

        # 8. Actualizar estado de sesión
        self.time_spent_min += time_min
        self.fatigue         = min(1.0, self.fatigue + self.fatigue_per_problem)
        self.topics_seen.update(problem_tags)

        if problem_id:
            self.problems_attempted.append(problem_id)
            if solved:
                self.problems_solved.append(problem_id)
                for tag in problem_tags:
                    if tag in self._topic_solves:
                        self._topic_solves[tag] += 1

        outcome = AttemptOutcome(
            solved             = solved,
            time_min           = time_min,
            p_solve            = round(p_solve, 4),
            reward             = round(reward, 2),
            delta_rating       = delta_rating,
            new_rating         = self.rating,
            fatigue            = round(self.fatigue, 4),
            time_remaining_min = round(self.time_remaining_min, 1),
            session_over       = self.session_over,
        )

        logger.debug(
            f"attempt pid={problem_id!r} | rating={self.rating} | "
            f"d_p={problem_rating} | δ_topics={delta_topics:.3f} | "
            f"p_solve={p_solve:.3f} | solved={solved} | "
            f"time={time_min:.1f}m | reward={reward:.1f} | "
            f"Δrating={delta_rating:+d} | fatigue={self.fatigue:.2f}"
        )

        return outcome

    def reset(self) -> None:
        """Reinicia el estado para una nueva sesión (nuevo episodio DQN)."""
        self.rating              = self.initial_rating
        self.fatigue             = 0.0
        self.time_spent_min      = 0.0
        self.problems_solved     = []
        self.problems_attempted  = []
        self.topics_seen         = set()
        self._topic_solves       = {t: 0 for t in CANONICAL_TOPICS}
        logger.debug(f"StudentModel reseteado — rating={self.rating}")

    # ------------------------------------------------------------------
    # Fórmulas públicas (sin efecto de estado — solo consulta)
    # ------------------------------------------------------------------

    def probability_of_solving(
        self,
        problem_rating : int,
        problem_tags   : list[str],
    ) -> float:
        """Calcula P_éxito = σ((R_s - d_p) / θ + δ_topics - λ·F).

        Parameters
        ----------
        problem_rating : int   — dificultad del problema d_p
        problem_tags   : list  — temas del problema

        Returns
        -------
        float en (0, 1)
        """
        delta_topics = self._topic_affinity(problem_tags)
        x = (
            (self.rating - problem_rating) / self.theta
            + delta_topics
            - self.lambda_fatigue * self.fatigue
        )
        return self._sigmoid(x)

    def estimate_solve_time(
        self,
        problem_rating  : int,
        problem_tags    : list[str],
        include_noise   : bool = False,
    ) -> float:
        """Estima el tiempo de resolución sin modificar el estado.

        Parameters
        ----------
        problem_rating : int
        problem_tags   : list[str]
        include_noise  : bool — si True, incluye ε estocástico

        Returns
        -------
        float — minutos estimados si el estudiante resuelve el problema.
        """
        t_base = self._solve_time_base(problem_rating, problem_tags)
        if include_noise:
            eps    = self._rng.uniform(-_EPS, _EPS)
            t_base = t_base * (1.0 + eps)
        return round(t_base, 1)

    def will_fit_in_session(
        self, problem_rating: int, problem_tags: list[str]
    ) -> bool:
        """True si el tiempo estimado cabe en el tiempo restante."""
        return self.estimate_solve_time(problem_rating, problem_tags) <= self.time_remaining_min

    # ------------------------------------------------------------------
    # Propiedades de estado
    # ------------------------------------------------------------------

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
        if self.n_attempted == 0:
            return 0.0
        return self.n_solved / self.n_attempted

    @property
    def state_vector(self) -> list[float]:
        """Vector numérico del estado actual del estudiante para el agente DQN.

        Returns
        -------
        list[float] de 5 componentes, todos en [0, 1]:
            [rating_norm, fatigue, time_used_norm, solve_rate, n_topics_norm]
        """
        rating_norm    = (self.rating - _MIN_RATING) / (_MAX_RATING - _MIN_RATING)
        time_used_norm = self.time_spent_min / self.session_budget_min
        n_topics_norm  = min(1.0, len(self.topics_seen) / 20.0)
        return [
            round(rating_norm,    4),
            round(self.fatigue,   4),
            round(time_used_norm, 4),
            round(self.solve_rate, 4),
            round(n_topics_norm,  4),
        ]

    # ------------------------------------------------------------------
    # Métodos privados — fórmulas internas
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float) -> float:
        """σ(x) = 1 / (1 + e^(-x)), numéricamente estable."""
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _topic_affinity(self, problem_tags: list[str]) -> float:
        """Calcula δ_topics ∈ [-0.5, 0.5] — bono por afinidad temática.

        Basado en cuántos problemas del mismo tema ha resuelto el estudiante.
        Si no ha resuelto ningún problema de los temas del problema → δ = 0.0
        Si domina todos los temas → δ = +0.5
        Si es novato en todos     → δ oscila cerca de 0.0

        Formula:
            maestria_tag = solves_tag / (solves_tag + 3)  ∈ [0, 1)
            δ_topics = mean(maestria por tags canónicos del problema) - 0.25
        """
        canonical = [t for t in problem_tags if t in self._topic_solves]
        if not canonical:
            return 0.0

        masteries = [
            self._topic_solves[t] / (self._topic_solves[t] + 3)
            for t in canonical
        ]
        mean_mastery = sum(masteries) / len(masteries)
        # Centrar en 0: rango [0, 1) → [-0.25, 0.75) pero acotamos a ±0.5
        delta = mean_mastery - 0.25
        return max(-_DELTA_MAX, min(_DELTA_MAX, delta))

    def _solve_time_base(self, problem_rating: int, problem_tags: list[str]) -> float:
        """Calcula T_base = T_min + (T_max - T_min) · f_dificultad · g_temas.

        f_dificultad = clip((d_p - R_s + Δ0) / Δ_max, 0, 1)
        g_temas      = 1 + α · (n_temas - 1)
        """
        # Factor de dificultad
        f_dif = (problem_rating - self.rating + self.delta0) / self.delta_max_time
        f_dif = max(0.0, min(1.0, f_dif))

        # Factor de complejidad temática
        n_temas = max(1, len(problem_tags))
        g_temas = 1.0 + self.alpha * (n_temas - 1)

        t_base = self.t_min + (self.t_max - self.t_min) * f_dif * g_temas
        return round(t_base, 2)

    def _update_rating(self, p_solve: float, solved: bool) -> int:
        """Actualiza el rating del estudiante y devuelve el cambio ΔR.

        Si resuelve : ΔR = C · (1 - P_éxito)  → más puntos si era difícil
        Si fracasa  : ΔR = 0
        """
        if solved:
            delta = int(round(self.c_elo * (1.0 - p_solve)))
            self.rating = min(_MAX_RATING, self.rating + delta)
            return delta
        return 0

    def __repr__(self) -> str:
        return (
            f"StudentModel("
            f"rating={self.rating}, "
            f"fatigue={self.fatigue:.2f}, "
            f"time_spent={self.time_spent_min:.1f}m, "
            f"solved={self.n_solved}/{self.n_attempted})"
        )