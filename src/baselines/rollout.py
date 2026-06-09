"""
rollout.py -- Baseline rollout de horizonte 2

Algoritmo
---------
    Para cada problema candidato p disponible:

        1. Calcula la recompensa esperada de intentar p (sin muestrear):
               r_p = P_exito(p) * (r_exito + delta_ELO_esperado(p))
                   + (1 - P_exito(p)) * r_fracaso

        2. Simula el estado resultante de intentar p (de forma determinista):
               - tiempo consumido esperado:
                     t_p = P*t_exito + (1-P)*beta*t_exito
                         = t_exito * (P + (1-P)*beta)
               - fatiga esperada: fatigue + fatigue_per_problem
               - ELO esperado por tema (solo en caso de exito):
                     elo_t' = elo_t + P_exito * delta_ELO_t

        3. Desde el estado simulado, aplica el greedy para elegir
           el mejor segundo problema q y calcula su recompensa esperada r_q.

        4. Recompensa total a 2 pasos:
               total(p) = r_p + r_q

    Elige el p que maximiza total(p).
    Si solo hay un problema disponible, se comporta como el greedy.

Nota
----
    Usa valores esperados en todos los calculos internos (no muestrea
    aleatoriedad) para hacer el rollout determinista y estable.
"""

import copy
import logging
import math

import numpy as np

from src.baselines.greedy import GreedySelector
from src.baselines.selector import ProblemSelector, SessionState, _sigmoid

logger = logging.getLogger(__name__)


class RolloutSelector(ProblemSelector):
    """Rollout de horizonte 2 con valores esperados deterministas.

    Parameters
    ----------
    greedy : GreedySelector
        Politica de rollout para el segundo paso.
        Si None, crea uno internamente.
    """

    def __init__(self, greedy: GreedySelector = None) -> None:
        self._greedy = greedy or GreedySelector()

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

        if len(available) == 1:
            return available[0][0]

        best_idx   = -1
        best_total = float("-inf")

        for idx, problem in available:
            total = self._evaluate_two_steps(state, idx, available_mask)
            logger.debug(
                f"  rollout pid={problem.problem_id} total_reward={total:.3f}"
            )
            if total > best_total:
                best_total = total
                best_idx   = idx

        return best_idx

    # ------------------------------------------------------------------
    # Evaluacion de dos pasos
    # ------------------------------------------------------------------

    def _evaluate_two_steps(
        self,
        state         : SessionState,
        first_idx     : int,
        available_mask: np.ndarray,
    ) -> float:
        """Calcula la recompensa esperada total a 2 pasos si se elige first_idx."""
        problem = state.problems[first_idx]
        student = state.student

        p_exito = state.p_exito(problem)
        delta   = state.delta_elo_esperado(problem)
        t_exito = state.t_estimado(problem)
        t_used  = t_exito * (p_exito + (1.0 - p_exito) * student.beta)

        # Recompensa esperada del primer paso
        r_exito  = student.r_exito + delta
        r_fracaso= student.r_fracaso
        r_first  = p_exito * r_exito + (1.0 - p_exito) * r_fracaso

        # Estado simulado tras el primer intento
        sim_state = self._simulate_step(state, first_idx, available_mask)

        if sim_state is None:
            return r_first   # sin tiempo para segundo paso

        # Recompensa esperada del segundo paso con greedy
        second_idx = self._greedy.select_action(sim_state, sim_state.available_mask)

        if second_idx >= state.n_problems:
            return r_first   # greedy decide terminar

        second_problem = state.problems[second_idx]
        p2      = sim_state.p_exito(second_problem)
        delta2  = sim_state.delta_elo_esperado(second_problem)
        r_second = p2 * (student.r_exito + delta2) + (1.0 - p2) * student.r_fracaso

        return r_first + r_second

    def _simulate_step(
        self,
        state         : SessionState,
        first_idx     : int,
        available_mask: np.ndarray,
    ) -> SessionState | None:
        """Simula el estado esperado tras intentar first_idx.

        Usa valores esperados en lugar de muestrear.
        Devuelve None si no queda tiempo para un segundo problema.
        """
        from src.environment.student_model import StudentModel
        from src.environment.problem import CANONICAL_TOPICS

        problem = state.problems[first_idx]
        student = state.student

        p_exito = state.p_exito(problem)
        t_exito = state.t_estimado(problem)
        t_used  = t_exito * (p_exito + (1.0 - p_exito) * student.beta)
        new_time_spent = student.time_spent_min + t_used

        if new_time_spent >= student.session_budget_min:
            return None   # sin tiempo

        # Construir topic_ratings simulados
        # Solo actualiza tags del problema en proporcion a P_exito
        theta  = student.theta
        c_elo  = student.c_elo
        new_topic_ratings = dict(student.topic_ratings)

        for tag in problem.tags_list:
            if tag not in new_topic_ratings:
                continue
            elo_t       = new_topic_ratings[tag]
            gap_t       = problem.rating - elo_t
            p_t         = _sigmoid((elo_t - problem.rating) / theta)
            challenge_t = _sigmoid(gap_t / theta)
            delta_t     = c_elo * (1.0 - p_t) * challenge_t
            # Aplicar en proporcion a P_exito (valor esperado)
            new_topic_ratings[tag] = min(
                3500, elo_t + p_exito * delta_t
            )

        # Construir nuevo StudentModel simulado
        sim_student = StudentModel(
            topic_ratings      = new_topic_ratings,
            session_budget_min = student.session_budget_min,
            random_seed        = None,
            theta              = student.theta,
            lambda_fatigue     = student.lambda_fatigue,
            # Compatible con fatigue_per_minute (nuevo) y fatigue_per_problem (antiguo)
            **({"fatigue_per_minute": student.fatigue_per_minute}
               if hasattr(student, "fatigue_per_minute")
               else {"fatigue_per_problem": student.fatigue_per_problem}),
            t_min              = student.t_min,
            t_max              = student.t_max,
            delta0             = student.delta0,
            delta_max_time     = student.delta_max_time,
            alpha              = student.alpha,
            beta               = student.beta,
            c_elo              = student.c_elo,
            r_exito            = student.r_exito,
            r_fracaso          = student.r_fracaso,
        )
        # Ajustar estado de sesion del estudiante simulado
        sim_student.time_spent_min = new_time_spent
        # Fatiga: proporcional al tiempo (nuevo) o por problema (antiguo)
        if hasattr(student, "fatigue_per_minute"):
            fatigue_inc = t_used * student.fatigue_per_minute
        else:
            fatigue_inc = student.fatigue_per_problem
        sim_student.fatigue = min(1.0, student.fatigue + fatigue_inc)
        sim_student.topics_seen    = student.topics_seen | set(problem.tags_list)

        # Mascara: marcar first_idx como usado
        new_mask = available_mask.copy()
        new_mask[first_idx] = False

        return SessionState(
            student            = sim_student,
            problems           = state.problems,
            available_mask     = new_mask,
            session_budget_min = student.session_budget_min,
        )