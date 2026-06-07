"""
selector.py -- Interfaz comun para todos los algoritmos de seleccion

Exporta:
    ProblemSelector  -- ABC que todos los algoritmos deben implementar
    SessionState     -- objeto de estado que se pasa a select_action
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.environment.problem import CANONICAL_TOPICS, N_TOPICS, Problem
from src.environment.student_model import StudentModel


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Estado completo de la sesion en un instante dado.

    Es el objeto que se pasa a select_action() en cada paso.
    Centraliza toda la informacion que un selector puede necesitar.

    Attributes
    ----------
    student            : StudentModel  -- estado actual del estudiante
    problems           : list[Problem] -- lista completa de problemas (fija)
    available_mask     : np.ndarray    -- shape (N,) bool, True si disponible
    time_left_norm     : float         -- tiempo restante / budget [0, 1]
    fatigue            : float         -- fatiga actual [0, 1]
    elos_por_tema      : np.ndarray    -- shape (N_TOPICS,) ratings normalizados
    session_budget_min : float         -- duracion total de la sesion
    """

    student            : StudentModel
    problems           : list[Problem]
    available_mask     : np.ndarray
    session_budget_min : float

    # Campos derivados calculados automaticamente en __post_init__
    time_left_norm : float          = field(init=False)
    fatigue        : float          = field(init=False)
    elos_por_tema  : np.ndarray     = field(init=False)

    def __post_init__(self) -> None:
        self.time_left_norm = (
            self.student.time_remaining_min / self.session_budget_min
            if self.session_budget_min > 0 else 0.0
        )
        self.fatigue = self.student.fatigue

        _MIN = 800
        _MAX = 3500
        self.elos_por_tema = np.array(
            [
                (self.student.topic_ratings.get(t, 1200) - _MIN) / (_MAX - _MIN)
                for t in CANONICAL_TOPICS
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Utilidades para los baselines
    # ------------------------------------------------------------------

    @property
    def n_problems(self) -> int:
        return len(self.problems)

    @property
    def time_remaining_min(self) -> float:
        return self.student.time_remaining_min

    def available_problems(self) -> list[tuple[int, Problem]]:
        """Lista de (indice, Problem) para problemas disponibles."""
        return [
            (i, p)
            for i, p in enumerate(self.problems)
            if self.available_mask[i]
        ]

    def p_exito(self, problem: Problem) -> float:
        """P_exito usando el StudentModel real."""
        return self.student.probability_of_solving(
            problem.rating, problem.tags_list
        )

    def delta_elo_esperado(self, problem: Problem) -> float:
        """Suma de delta_ELO esperado sobre los tags del problema.

        delta_ELO_t = C * (1 - P_t) * challenge_t
        donde P_t y challenge_t usan elo_t del estudiante.
        """
        total = 0.0
        theta = self.student.theta
        c_elo = self.student.c_elo
        for tag in problem.tags_list:
            if tag not in self.student.topic_ratings:
                continue
            elo_t       = self.student.topic_ratings[tag]
            gap_t       = problem.rating - elo_t
            p_t         = _sigmoid((elo_t - problem.rating) / theta)
            challenge_t = _sigmoid(gap_t / theta)
            total      += c_elo * (1.0 - p_t) * challenge_t
        return total

    def t_estimado(self, problem: Problem) -> float:
        """Tiempo estimado de resolucion usando el StudentModel."""
        return max(1.0, self.student.estimate_solve_time(
            problem.rating, problem.tags_list
        ))


# ---------------------------------------------------------------------------
# ProblemSelector -- interfaz comun
# ---------------------------------------------------------------------------

class ProblemSelector(ABC):
    """Interfaz para cualquier algoritmo de seleccion de problemas.

    Implementaciones:
        GreedySelector    -- heuristico greedy por score/tiempo
        KnapsackSelector  -- mochila 0/1 estatica
        RolloutSelector   -- rollout de horizonte 2
        DQNSelector       -- wrapper del agente DQN entrenado
    """

    @abstractmethod
    def select_action(
        self,
        state         : SessionState,
        available_mask: np.ndarray,
        **kwargs,
    ) -> int:
        """Devuelve el indice del problema a seleccionar.

        Parameters
        ----------
        state          : SessionState -- estado actual de la sesion
        available_mask : np.ndarray (N,) bool

        Returns
        -------
        int -- indice del problema elegido en [0, N-1],
               o N para indicar "terminar sesion".
        """

    def reset(self) -> None:
        """Reinicia el estado interno del selector entre episodios."""


# ---------------------------------------------------------------------------
# Utilidad compartida
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)