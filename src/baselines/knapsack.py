"""
knapsack.py -- Baseline mochila 0/1

Modelo
------
    Selecciona al inicio del episodio el conjunto de problemas que
    maximiza el valor total sin exceder el tiempo disponible.

    Valor de un problema:
        V(p) = rating_p + w * n_temas_nuevos(p)

        - rating_p         : dificultad del problema (proxy de aprendizaje)
        - n_temas_nuevos   : numero de tags canonicos del problema que
                             no estan en topics_seen del estudiante
        - w                : peso del bonus de tema nuevo (default 50)

    Peso (costo temporal):
        peso(p) = round(t_estimado(p))   en minutos enteros

    Restriccion:
        sum(peso(p) for p in S) <= time_remaining_min

    Algoritmo: DP 0/1 clasico O(N * W).

    Una vez elegido el conjunto, los problemas se ordenan por dificultad
    ascendente para la simulacion (de facil a dificil).

Nota
----
    Este baseline es el unico NO secuencial: toma todas las decisiones
    al inicio del episodio y las ejecuta en orden fijo. No reacciona
    a fatiga ni a resultados intermedios.
"""

import logging

import numpy as np

from src.baselines.selector import ProblemSelector, SessionState

logger = logging.getLogger(__name__)


class KnapsackSelector(ProblemSelector):
    """Baseline de mochila 0/1 estatica.

    Parameters
    ----------
    w          : float -- peso del bonus por tema nuevo (default 50)
    max_weight : int   -- capacidad maxima de la mochila en minutos.
                          Si None, usa time_remaining_min del estado.
    """

    def __init__(self, w: float = 50.0, max_weight: int = None) -> None:
        self.w          = w
        self.max_weight = max_weight
        self._plan: list[int] = []   # secuencia pre-calculada de indices
        self._plan_pos : int  = 0    # posicion actual en el plan

    def select_action(
        self,
        state         : SessionState,
        available_mask: np.ndarray,
        **kwargs,
    ) -> int:
        # Si no hay plan o quedan problemas del plan que siguen disponibles
        remaining_plan = [
            idx for idx in self._plan[self._plan_pos:]
            if idx < len(available_mask) and available_mask[idx]
        ]

        if remaining_plan:
            next_idx         = remaining_plan[0]
            self._plan_pos   = self._plan.index(next_idx) + 1
            return next_idx

        # Recalcular plan con available_mask actual
        self._plan     = self._solve_knapsack(state, available_mask)
        self._plan_pos = 0

        if not self._plan:
            return state.n_problems   # sin solucion, terminar

        next_idx       = self._plan[0]
        self._plan_pos = 1
        return next_idx

    def reset(self) -> None:
        self._plan     = []
        self._plan_pos = 0

    # ------------------------------------------------------------------
    # Algoritmo DP
    # ------------------------------------------------------------------

    def _solve_knapsack(
        self, state: SessionState, available_mask: np.ndarray
    ) -> list[int]:
        """Resuelve la mochila 0/1 y devuelve los indices ordenados por rating."""
        available = [
            (i, p) for i, p in enumerate(state.problems)
            if available_mask[i]
        ]
        if not available:
            return []

        W = int(self.max_weight or state.time_remaining_min)
        if W <= 0:
            return []

        # Construir listas de items
        indices = []
        weights = []
        values  = []

        for idx, problem in available:
            t      = max(1, round(state.t_estimado(problem)))
            n_new  = self._count_new_topics(problem, state)
            v      = problem.rating + self.w * n_new

            indices.append(idx)
            weights.append(t)
            values.append(v)

        n = len(indices)

        # Tabla DP: dp[i][w] = valor maximo usando primeros i items con peso <= w
        # Optimizacion de memoria: solo guardamos la fila actual y la anterior
        dp = [0.0] * (W + 1)

        for i in range(n):
            wi = weights[i]
            vi = values[i]
            # Recorrer de derecha a izquierda para 0/1
            for w in range(W, wi - 1, -1):
                dp[w] = max(dp[w], dp[w - wi] + vi)

        # Reconstruir solucion
        selected = []
        w = W
        for i in range(n - 1, -1, -1):
            wi = weights[i]
            vi = values[i]
            if w >= wi and dp[w] == dp[w - wi] + vi:
                selected.append(indices[i])
                w -= wi

        # Ordenar por dificultad ascendente (facil a dificil)
        selected.sort(key=lambda idx: state.problems[idx].rating)

        logger.debug(
            f"  knapsack: {len(selected)} problemas seleccionados "
            f"de {n} disponibles | W={W}min"
        )
        return selected

    def _count_new_topics(self, problem, state: SessionState) -> int:
        """Numero de tags canonicos del problema no vistos aun por el estudiante."""
        from src.environment.problem import _TOPIC_IDX
        return sum(
            1 for t in problem.tags_list
            if t in _TOPIC_IDX and t not in state.student.topics_seen
        )