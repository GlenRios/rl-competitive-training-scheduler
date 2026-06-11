"""
greedy.py -- Baseline greedy heuristico

Formula
-------
    score(p) = P_exito(p) * delta_ELO_esperado(p) / t_estimado(p)

    - P_exito(p)          : probabilidad de exito con estado actual del estudiante
    - delta_ELO_esperado  : suma de C*(1-P_t)*challenge_t por tag canonico del problema
    - t_estimado(p)       : tiempo estimado de resolucion

Comportamiento
--------------
    En cada paso elige el problema disponible con mayor score.
    No explora -- siempre es greedy puro.
    Si ningun problema disponible tiene score > 0, elige el de mayor P_exito.
"""

import logging

import numpy as np

from src.baselines.selector import ProblemSelector, SessionState

logger = logging.getLogger(__name__)


class GreedySelector(ProblemSelector):
    """Selecciona en cada paso el problema con mejor score heuristico.

    Parameters
    ----------
    eps : float
        Pequeño valor para evitar division por cero en t_estimado.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def select_action(
        self,
        state         : SessionState,
        available_mask: np.ndarray,
        **kwargs,
    ) -> int:
        available = [
            (i, p) for i, p in enumerate(state.problems)
            if available_mask[i]
        ]
        if not available:
            return state.n_problems

        best_idx   = -1
        best_score = float("-inf")

        for idx, problem in available:
            p       = state.p_exito(problem)
            delta   = state.delta_elo_esperado(problem)
            t       = state.t_estimado(problem) + self.eps
            score   = (p * delta) / t

            logger.debug(
                f"  greedy pid={problem.problem_id} "
                f"p={p:.3f} delta={delta:.2f} t={t:.1f} score={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_idx   = idx

        return best_idx

    def score(self, state: SessionState, problem_idx: int) -> float:
        """Calcula el score greedy para un problema especifico.

        Util para el RolloutSelector que reutiliza esta logica.
        """
        problem = state.problems[problem_idx]
        p       = state.p_exito(problem)
        delta   = state.delta_elo_esperado(problem)
        t       = state.t_estimado(problem) + self.eps
        return (p * delta) / t