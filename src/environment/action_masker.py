"""
action_masker.py — Máscara de acciones válidas

Responsabilidad única:
    Determinar qué problemas puede seleccionar el agente en cada paso,
    y exponer esa información como una máscara booleana numpy.

Un problema es una acción INVÁLIDA si:
    1. Ya fue intentado en esta sesión (no repetir problemas).
    2. Su tiempo estimado de resolución supera el tiempo restante de sesión
       (no tiene sentido seleccionarlo si no cabe en el tiempo).

La máscara se actualiza tras cada `step()` del entorno.
El agente DQN la usa para filtrar Q-values antes de seleccionar la acción:
    q_values[~mask] = -inf  →  el agente nunca elige acciones inválidas.

Uso
---
    from src.environment.action_masker import ActionMasker

    masker = ActionMasker(problems)
    mask   = masker.get_mask(student)   # np.ndarray bool (N,)

    masker.mark_attempted(problem_idx=42)
    mask   = masker.get_mask(student)   # el problema 42 ya aparece como False
"""

import logging

import numpy as np

from src.environment.problem import Problem
from src.environment.student_model import StudentModel

logger = logging.getLogger(__name__)


class ActionMasker:
    """Gestiona la máscara de acciones válidas durante un episodio.

    Parameters
    ----------
    problems : list[Problem]
        Lista completa de problemas del episodio (fija durante el episodio).
    """

    def __init__(self, problems: list[Problem]) -> None:
        if not problems:
            raise ValueError("La lista de problemas no puede estar vacía.")
        self.problems    = problems
        self.n_problems  = len(problems)
        self._attempted  : set[int] = set()   # índices ya intentados

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def get_mask(self, student: StudentModel) -> np.ndarray:
        """Calcula la máscara booleana de acciones válidas.

        Parameters
        ----------
        student : StudentModel
            Estado actual del estudiante (se usa time_remaining_min).

        Returns
        -------
        np.ndarray shape (N_PROBLEMS,), dtype bool
            True  → acción válida (problema disponible y cabe en tiempo)
            False → acción inválida
        """
        mask = np.ones(self.n_problems, dtype=bool)

        for idx, problem in enumerate(self.problems):
            if idx in self._attempted:
                mask[idx] = False
                continue

            # Comprobar si el tiempo estimado cabe en el tiempo restante
            est_time = student.estimate_solve_time(
                problem.rating, problem.tags_list
            )
            if est_time > student.time_remaining_min:
                mask[idx] = False

        return mask

    def mark_attempted(self, problem_idx: int) -> None:
        """Marca un problema como intentado — ya no será acción válida.

        Parameters
        ----------
        problem_idx : int
            Índice del problema en la lista self.problems.
        """
        if not (0 <= problem_idx < self.n_problems):
            raise IndexError(
                f"problem_idx={problem_idx} fuera de rango [0, {self.n_problems})."
            )
        self._attempted.add(problem_idx)

    def reset(self) -> None:
        """Limpia el historial de intentos para un nuevo episodio."""
        self._attempted.clear()

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def n_attempted(self) -> int:
        return len(self._attempted)

    @property
    def n_available(self) -> int:
        """Número de problemas aún no intentados (sin considerar tiempo)."""
        return self.n_problems - self.n_attempted

    def any_valid(self, student: StudentModel) -> bool:
        """True si hay al menos una acción válida disponible."""
        return self.get_mask(student).any()

    def valid_indices(self, student: StudentModel) -> list[int]:
        """Lista de índices de acciones válidas."""
        return list(np.where(self.get_mask(student))[0])