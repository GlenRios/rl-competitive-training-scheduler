"""
student_model.py — Módulo 5: Modelo del estudiante virtual

Responsabilidad:
    Simular el comportamiento de un estudiante durante una sesión de
    entrenamiento de programación competitiva.

    El modelo captura tres dinámicas reales:

    1. Tiempo de resolución
       Depende de la brecha entre el rating del problema y el del estudiante,
       y de la fatiga acumulada en la sesión.

    2. Probabilidad de resolver
       Un estudiante no siempre resuelve un problema en su nivel — hay
       incertidumbre. La probabilidad cae cuando el problema está muy por
       encima de su nivel o cuando está muy fatigado.

    3. Aprendizaje dentro de la sesión
       Resolver un problema actualiza el rating del estudiante ligeramente
       (modelo ELO simplificado). Esto hace que problemas antes difíciles
       se vuelvan más accesibles conforme avanza la sesión.

Estado del estudiante
---------------------
    rating          : int    — nivel actual (800–3500), dinámico en la sesión
    fatigue         : float  — [0.0, 1.0], crece con cada problema intentado
    time_spent_min  : float  — minutos consumidos en la sesión actual
    problems_solved : list   — problem_ids resueltos en esta sesión
    topics_seen     : set    — temas encontrados en esta sesión

Uso
---
    from src.environment.student_model import StudentModel

    student = StudentModel(initial_rating=1500, session_budget_min=120)

    # Simular intento de un problema
    outcome = student.attempt(problem_rating=1600, problem_tags=["dp", "graphs"])

    print(outcome["solved"])        # True / False
    print(outcome["time_min"])      # minutos usados
    print(outcome["new_rating"])    # rating actualizado
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración del modelo
# ---------------------------------------------------------------------------

# Tabla base de tiempo de resolución por nivel de rating del problema.
# Calibrada para un estudiante de rating idéntico al problema (gap = 0).
# Unidad: minutos.
_BASE_TIME_TABLE: list[tuple[int, int, float]] = [
    (800,  1099, 12.0),
    (1100, 1299, 18.0),
    (1300, 1499, 25.0),
    (1500, 1699, 35.0),
    (1700, 1899, 48.0),
    (1900, 2099, 65.0),
    (2100, 2399, 85.0),
    (2400, 2799, 110.0),
    (2800, 9999, 140.0),
]

# Multiplicadores de tiempo según brecha (problem_rating - student_rating).
# Tuplas: (gap_min, gap_max, time_multiplier, solve_probability_base)
_GAP_TABLE: list[tuple[int, int, float, float]] = [
    (-9999, -401, 0.40, 0.97),   # problema muy fácil: rápido, casi seguro
    (-400,  -201, 0.60, 0.92),   # fácil
    (-200,   -1,  0.80, 0.85),   # ligeramente fácil
    (0,      199, 1.00, 0.72),   # en su nivel
    (200,    399, 1.60, 0.52),   # difícil
    (400,    599, 2.50, 0.28),   # muy difícil
    (600,   9999, 4.00, 0.08),   # fuera de alcance
]

# Constante de aprendizaje ELO simplificado
_ELO_K = 8

# Penalización de fatiga sobre la probabilidad de resolver
# fatigue=0.0 → penalización=0.0  |  fatigue=1.0 → penalización=MAX_FATIGUE_PENALTY
_MAX_FATIGUE_PENALTY = 0.25

# Cuánto aumenta la fatiga por cada problema intentado
_FATIGUE_PER_PROBLEM = 0.08

# Límites de rating (no puede salir del rango válido de Codeforces)
_MIN_RATING = 800
_MAX_RATING = 3500


# ---------------------------------------------------------------------------
# Dataclass: resultado de un intento
# ---------------------------------------------------------------------------

@dataclass
class AttemptOutcome:
    """Resultado de intentar un problema.

    Attributes
    ----------
    solved : bool
        True si el estudiante lo resolvió, False si no.
    time_min : float
        Minutos reales consumidos en el intento (incluso si no resolvió).
    problem_rating : int
        Rating del problema intentado.
    new_rating : int
        Rating del estudiante tras el intento.
    delta_rating : int
        Cambio de rating producido por este intento.
    fatigue : float
        Nivel de fatiga del estudiante tras el intento [0.0, 1.0].
    time_remaining_min : float
        Minutos restantes en la sesión.
    session_over : bool
        True si la sesión terminó (sin tiempo) tras este intento.
    """
    solved             : bool
    time_min           : float
    problem_rating     : int
    new_rating         : int
    delta_rating       : int
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
    initial_rating : int
        Rating inicial del estudiante (800–3500).
    session_budget_min : float
        Tiempo máximo de la sesión en minutos (default: 120).
    random_seed : Optional[int]
        Semilla para reproducibilidad. None = no determinista.
    fatigue_per_problem : float
        Incremento de fatiga por cada problema intentado (default: 0.08).
    elo_k : int
        Constante de aprendizaje ELO (default: 8).
    """

    def __init__(
        self,
        initial_rating     : int   = 1500,
        session_budget_min : float = 120.0,
        random_seed        : Optional[int] = None,
        fatigue_per_problem: float = _FATIGUE_PER_PROBLEM,
        elo_k              : int   = _ELO_K,
    ) -> None:
        if not (_MIN_RATING <= initial_rating <= _MAX_RATING):
            raise ValueError(
                f"initial_rating debe estar entre {_MIN_RATING} y {_MAX_RATING}. "
                f"Recibido: {initial_rating}"
            )
        if session_budget_min <= 0:
            raise ValueError("session_budget_min debe ser positivo.")

        self.initial_rating      = initial_rating
        self.session_budget_min  = session_budget_min
        self.fatigue_per_problem = fatigue_per_problem
        self.elo_k               = elo_k
        self._rng                = random.Random(random_seed)

        # Estado de sesión — se inicializa con reset()
        self.rating             : int         = initial_rating
        self.fatigue            : float       = 0.0
        self.time_spent_min     : float       = 0.0
        self.problems_solved    : list[str]   = []
        self.problems_attempted : list[str]   = []
        self.topics_seen        : set[str]    = set()

    # ------------------------------------------------------------------
    # API pública principal
    # ------------------------------------------------------------------

    def attempt(
        self,
        problem_rating : int,
        problem_tags   : list[str],
        problem_id     : str = "",
    ) -> AttemptOutcome:
        """Simula el intento del estudiante en un problema.

        Parameters
        ----------
        problem_rating : int
            Rating del problema a intentar.
        problem_tags : list[str]
            Lista de temas del problema (ej. ["dp", "graphs"]).
        problem_id : str
            Identificador del problema (opcional, para registro).

        Returns
        -------
        AttemptOutcome con todos los resultados del intento.

        Raises
        ------
        RuntimeError
            Si la sesión ya terminó (sin tiempo restante).
        """
        if self.session_over:
            raise RuntimeError(
                "La sesión ya terminó. Llama a reset() para iniciar una nueva."
            )

        # 1. Calcular tiempo y probabilidad según la brecha
        gap           = problem_rating - self.rating
        time_mult, p_solve_base = self._lookup_gap(gap)
        base_time     = self._base_time(problem_rating)
        raw_time      = base_time * time_mult

        # 2. Aplicar penalización de fatiga
        fatigue_penalty = self.fatigue * _MAX_FATIGUE_PENALTY
        p_solve         = max(0.0, p_solve_base - fatigue_penalty)
        time_min        = raw_time * (1.0 + self.fatigue * 0.3)  # fatiga alarga el tiempo

        # 3. Decidir si resuelve (estocastico)
        solved = self._rng.random() < p_solve

        # 4. Si no resuelve, solo gasta una fracción del tiempo
        if not solved:
            time_min *= 0.6   # intentó pero no terminó

        time_min = round(time_min, 1)

        # 5. Actualizar estado
        self.time_spent_min  += time_min
        self.fatigue          = min(1.0, self.fatigue + self.fatigue_per_problem)
        self.topics_seen.update(problem_tags)

        if problem_id:
            self.problems_attempted.append(problem_id)
            if solved:
                self.problems_solved.append(problem_id)

        # 6. Actualizar rating (ELO simplificado)
        old_rating   = self.rating
        delta_rating = self._update_rating(problem_rating, solved)

        outcome = AttemptOutcome(
            solved             = solved,
            time_min           = time_min,
            problem_rating     = problem_rating,
            new_rating         = self.rating,
            delta_rating       = delta_rating,
            fatigue            = self.fatigue,
            time_remaining_min = self.time_remaining_min,
            session_over       = self.session_over,
        )

        logger.debug(
            f"  attempt pid={problem_id or '?'!r} | rating={old_rating} | "
            f"gap={gap:+d} | p_solve={p_solve:.2f} | "
            f"solved={solved} | time={time_min:.1f}m | "
            f"Δrating={delta_rating:+d} | fatigue={self.fatigue:.2f} | "
            f"time_left={self.time_remaining_min:.1f}m"
        )

        return outcome

    def reset(self) -> None:
        """Reinicia el estado para una nueva sesión (nuevo episodio)."""
        self.rating              = self.initial_rating
        self.fatigue             = 0.0
        self.time_spent_min      = 0.0
        self.problems_solved     = []
        self.problems_attempted  = []
        self.topics_seen         = set()
        logger.debug(f"StudentModel reseteado — rating={self.rating}")

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
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
        """Vector numérico del estado actual del estudiante.

        Usado como parte de la observación del agente DQN.
        Todos los valores están normalizados a [0, 1].

        Returns
        -------
        list[float] con 5 componentes:
            [rating_norm, fatigue, time_used_norm,
             solve_rate, n_topics_norm]
        """
        rating_norm    = (self.rating - _MIN_RATING) / (_MAX_RATING - _MIN_RATING)
        time_used_norm = self.time_spent_min / self.session_budget_min
        n_topics_norm  = min(1.0, len(self.topics_seen) / 20.0)   # 20 temas max

        return [
            round(rating_norm,    4),
            round(self.fatigue,   4),
            round(time_used_norm, 4),
            round(self.solve_rate, 4),
            round(n_topics_norm,  4),
        ]

    # ------------------------------------------------------------------
    # Métodos de estimación (sin efectos de estado — solo consulta)
    # ------------------------------------------------------------------

    def estimate_solve_time(
        self, problem_rating: int, include_fatigue: bool = True
    ) -> float:
        """Estima cuántos minutos tardaría en resolver este problema.

        No modifica el estado del estudiante — solo calcula.

        Parameters
        ----------
        problem_rating : int
            Rating del problema.
        include_fatigue : bool
            Si True, incluye el efecto de la fatiga actual en la estimación.

        Returns
        -------
        float — minutos estimados si resuelve.
        """
        gap       = problem_rating - self.rating
        mult, _   = self._lookup_gap(gap)
        base_time = self._base_time(problem_rating)
        time_min  = base_time * mult

        if include_fatigue:
            time_min *= (1.0 + self.fatigue * 0.3)

        return round(time_min, 1)

    def probability_of_solving(self, problem_rating: int) -> float:
        """Calcula la probabilidad de resolver el problema dado el estado actual.

        No modifica el estado del estudiante — solo calcula.

        Parameters
        ----------
        problem_rating : int
            Rating del problema.

        Returns
        -------
        float en [0.0, 1.0]
        """
        gap           = problem_rating - self.rating
        _, p_base     = self._lookup_gap(gap)
        fatigue_pen   = self.fatigue * _MAX_FATIGUE_PENALTY
        return max(0.0, round(p_base - fatigue_pen, 4))

    def will_fit_in_session(self, problem_rating: int) -> bool:
        """Indica si el tiempo estimado cabe en el tiempo restante de sesión."""
        return self.estimate_solve_time(problem_rating) <= self.time_remaining_min

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _lookup_gap(self, gap: int) -> tuple[float, float]:
        """Devuelve (time_multiplier, p_solve_base) para la brecha dada."""
        for lo, hi, mult, p in _GAP_TABLE:
            if lo <= gap <= hi:
                return mult, p
        # Fallback al último rango (fuera de alcance)
        return _GAP_TABLE[-1][2], _GAP_TABLE[-1][3]

    def _base_time(self, problem_rating: int) -> float:
        """Tiempo base (minutos) para un estudiante de igual nivel que el problema."""
        for lo, hi, minutes in _BASE_TIME_TABLE:
            if lo <= problem_rating <= hi:
                return minutes
        return _BASE_TIME_TABLE[-1][2]

    def _update_rating(self, problem_rating: int, solved: bool) -> int:
        """Actualiza el rating con un modelo ELO simplificado.

        La puntuación esperada (E) se calcula con la fórmula ELO estándar.
        El delta es K * (resultado - E), donde resultado=1 si resolvió, 0 si no.

        Returns
        -------
        int — cambio de rating aplicado (puede ser negativo).
        """
        # Puntuación esperada ELO
        expected = 1.0 / (1.0 + math.pow(10, (problem_rating - self.rating) / 400.0))
        actual   = 1.0 if solved else 0.0
        delta    = int(round(self.elo_k * (actual - expected)))

        self.rating = max(_MIN_RATING, min(_MAX_RATING, self.rating + delta))
        return delta

    def __repr__(self) -> str:
        return (
            f"StudentModel("
            f"rating={self.rating}, "
            f"fatigue={self.fatigue:.2f}, "
            f"time_spent={self.time_spent_min:.1f}m, "
            f"solved={self.n_solved}/{self.n_attempted})"
        )