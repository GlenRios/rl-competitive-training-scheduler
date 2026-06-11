"""
observation_builder.py — Construcción de observaciones para el agente DQN

Responsabilidad única:
    Transformar el estado actual (StudentModel + lista de Problems) en
    arrays numpy listos para ser consumidos por la red neuronal del agente.

Estructura de la observación
-----------------------------
    La observación se compone de dos partes separadas:

    1. student_obs  — shape (STUDENT_OBS_DIM,) = (5,)
       Estado actual del estudiante, normalizado en [0, 1].
       [rating_norm, fatigue, time_used_norm, solve_rate, topics_norm]

    2. problem_matrix — shape (N_PROBLEMS, PROBLEM_OBS_DIM) = (N, 24)
       Una fila por problema: sus características vistas desde el estado
       actual del estudiante (gap, p_solve, time_norm, one-hot de tags).

    El agente DQN concatena student_obs con cada fila de problem_matrix
    para obtener el input completo de Q(s, a_i):
        input_i = [student_obs | problem_matrix[i]]   shape: (5+24,) = (29,)

    Esta separación permite:
    - Reutilizar student_obs una sola vez por step.
    - Recalcular problem_matrix solo cuando cambia el estado del estudiante.

Constantes exportadas
---------------------
    STUDENT_OBS_DIM  = 5
    PROBLEM_OBS_DIM  = 24   (4 numéricas + 20 one-hot de tags)
    FULL_OBS_DIM     = 29   (input final de la red Q por acción)
"""

import logging

import numpy as np

from src.environment.problem import Problem
from src.environment.student_model import StudentModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dimensiones exportadas — el agente DQN las usa para construir su red
# ---------------------------------------------------------------------------

STUDENT_OBS_DIM = 24   # longitud de state_vector del StudentModel (4 + 20 topics)
PROBLEM_OBS_DIM = 24   # longitud de to_observation_vector de Problem
FULL_OBS_DIM    = STUDENT_OBS_DIM + PROBLEM_OBS_DIM   # = 48


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """Construye las observaciones numpy a partir del estado del entorno.

    Parameters
    ----------
    problems : list[Problem]
        Lista completa de problemas del episodio (fija durante el episodio).
    session_budget_min : float
        Duración total de la sesión en minutos (necesaria para normalizar
        el tiempo en la observación del problema).
    """

    def __init__(
        self,
        problems           : list[Problem],
        session_budget_min : float,
    ) -> None:
        if not problems:
            raise ValueError("La lista de problemas no puede estar vacía.")
        if session_budget_min <= 0:
            raise ValueError("session_budget_min debe ser positivo.")

        self.problems            = problems
        self.session_budget_min  = session_budget_min
        self.n_problems          = len(problems)

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def student_obs(self, student: StudentModel) -> np.ndarray:
        """Observación del estado del estudiante.

        Returns
        -------
        np.ndarray shape (STUDENT_OBS_DIM,) = (5,), dtype float32
            [rating_norm, fatigue, time_used_norm, solve_rate, topics_norm]
        """
        return np.array(student.state_vector, dtype=np.float32)

    def problem_matrix(self, student: StudentModel) -> np.ndarray:
        """Matriz de observaciones de todos los problemas.

        Cada fila representa un problema visto desde el estado actual
        del estudiante (gap, p_solve, time_norm cambian con el estado).

        Returns
        -------
        np.ndarray shape (N_PROBLEMS, PROBLEM_OBS_DIM) = (N, 24), dtype float32
        """
        rows = []
        for problem in self.problems:
            p_solve = student.probability_of_solving(
                problem.rating, problem.tags_list
            )
            vec = problem.to_observation_vector(
                student_rating     = student.rating,
                student_fatigue    = student.fatigue,
                session_budget_min = self.session_budget_min,
                time_spent_min     = student.time_spent_min,
                p_solve            = p_solve,
            )
            rows.append(vec)

        return np.array(rows, dtype=np.float32)

    def full_input_matrix(self, student: StudentModel) -> np.ndarray:
        """Concatena student_obs con cada fila de problem_matrix.

        Produce la matriz de inputs completa para la red Q,
        donde cada fila i es el input para calcular Q(s, a_i).

        Returns
        -------
        np.ndarray shape (N_PROBLEMS, FULL_OBS_DIM) = (N, 29), dtype float32
        """
        s_obs   = self.student_obs(student)                  # (5,)
        p_mat   = self.problem_matrix(student)               # (N, 24)
        s_tiled = np.tile(s_obs, (self.n_problems, 1))       # (N, 5)
        return np.concatenate([s_tiled, p_mat], axis=1)      # (N, 29)

    def single_problem_input(
        self, student: StudentModel, problem_idx: int
    ) -> np.ndarray:
        """Input de la red Q para un único problema.

        Útil para calcular Q(s, a_i) durante la selección de acción.

        Returns
        -------
        np.ndarray shape (FULL_OBS_DIM,) = (29,), dtype float32
        """
        if not (0 <= problem_idx < self.n_problems):
            raise IndexError(
                f"problem_idx={problem_idx} fuera de rango [0, {self.n_problems})."
            )
        problem = self.problems[problem_idx]
        p_solve = student.probability_of_solving(
            problem.rating, problem.tags_list
        )
        p_vec = problem.to_observation_vector(
            student_rating     = student.rating,
            student_fatigue    = student.fatigue,
            session_budget_min = self.session_budget_min,
            time_spent_min     = student.time_spent_min,
            p_solve            = p_solve,
        )
        s_vec = student.state_vector
        return np.array(s_vec + p_vec, dtype=np.float32)