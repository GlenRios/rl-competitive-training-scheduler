"""
metrics.py -- Calculo de metricas por episodio

Responsabilidad unica:
    Dado el historial de un episodio y el estado final del estudiante,
    calcula todas las metricas definidas para la comparacion de selectores.

Metricas implementadas
-----------------------
    total_reward          : suma de recompensas inmediatas del episodio
    spearman_progression  : correlacion de Spearman entre orden de resolucion
                            y rating del problema (1.0 = progresion perfecta)
    topic_coverage        : numero de tags canonicos distintos cubiertos
    success_rate          : problemas resueltos / intentados
    time_used_min         : tiempo total consumido en la sesion
    rating_improvement    : global_rating final - global_rating inicial
    compute_time_s        : segundos de computo del episodio (opcional)

Uso
---
    from src.evaluation.metrics import compute_episode_metrics

    metrics = compute_episode_metrics(
        history        = episode_history,    # list[dict] del env
        initial_rating = profile.global_rating,
        final_rating   = student.global_rating,
        compute_time_s = elapsed,
    )
"""

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Importacion lazy de scipy para no fallar si no esta instalado
try:
    from scipy.stats import spearmanr
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy no disponible -- spearman_progression usara implementacion manual")

from src.environment.problem import CANONICAL_TOPICS


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def compute_episode_metrics(
    history        : list[dict],
    initial_rating : float,
    final_rating   : float,
    compute_time_s : float = 0.0,
) -> dict:
    """Calcula todas las metricas de un episodio.

    Parameters
    ----------
    history        : list[dict]
        Lista de step_records del episodio. Cada dict contiene al menos:
        {solved, problem_rating, reward, time_min, topic_deltas, p_solve, ...}
    initial_rating : float
        Rating global del estudiante al inicio del episodio.
    final_rating   : float
        Rating global del estudiante al final del episodio.
    compute_time_s : float
        Segundos de computo del episodio completo.

    Returns
    -------
    dict con todas las metricas, listas para ser convertidas a DataFrame.
    """
    if not history:
        return _empty_metrics(compute_time_s)

    total_reward         = _total_reward(history)
    spearman             = _spearman_progression(history)
    topic_coverage       = _topic_coverage(history)
    success_rate         = _success_rate(history)
    time_used            = _time_used(history)
    rating_improvement   = round(final_rating - initial_rating, 2)
    n_attempted          = len(history)
    n_solved             = sum(1 for s in history if s.get("solved", False))

    return {
        "total_reward"         : round(total_reward, 2),
        "spearman_progression" : round(spearman, 4),
        "topic_coverage"       : topic_coverage,
        "success_rate"         : round(success_rate, 4),
        "time_used_min"        : round(time_used, 2),
        "rating_improvement"   : rating_improvement,
        "n_attempted"          : n_attempted,
        "n_solved"             : n_solved,
        "compute_time_s"       : round(compute_time_s, 4),
    }


# ---------------------------------------------------------------------------
# Metricas individuales
# ---------------------------------------------------------------------------

def _total_reward(history: list[dict]) -> float:
    """Suma de todas las recompensas inmediatas del episodio."""
    return sum(s.get("reward", 0.0) for s in history)


def _spearman_progression(history: list[dict]) -> float:
    """Correlacion de Spearman entre orden de intento y rating del problema.

    Valores:
        +1.0  -> progresion perfecta (siempre de facil a dificil)
         0.0  -> orden aleatorio
        -1.0  -> progresion inversa (siempre de dificil a facil)

    Si hay menos de 2 problemas, devuelve 0.0 (no definida).
    """
    ratings = [s.get("problem_rating", 0) for s in history]
    if len(ratings) < 2:
        return 0.0

    n      = len(ratings)
    orders = list(range(1, n + 1))

    if _SCIPY_AVAILABLE:
        corr, _ = spearmanr(orders, ratings)
        return float(corr) if not math.isnan(corr) else 0.0

    # Implementacion manual de Spearman
    return _spearman_manual(orders, ratings)


def _spearman_manual(x: list, y: list) -> float:
    """Correlacion de Spearman sin scipy."""
    n      = len(x)
    rank_x = _ranks(x)
    rank_y = _ranks(y)
    d2     = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    return 1.0 - (6.0 * d2) / (n * (n ** 2 - 1))


def _ranks(values: list) -> list[float]:
    """Calcula los rangos de una lista (con manejo de empates por media)."""
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks       = [0.0] * len(values)
    i           = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j+1][1] == sorted_vals[i][1]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = rank
        i = j + 1
    return ranks


def _topic_coverage(history: list[dict]) -> int:
    """Numero de tags canonicos unicos cubiertos durante el episodio."""
    canonical_set = set(CANONICAL_TOPICS)
    seen: set[str] = set()
    for step in history:
        deltas = step.get("topic_deltas", {})
        if isinstance(deltas, dict):
            seen.update(t for t in deltas.keys() if t in canonical_set)
        # Tambien considerar problemas fallados (topics_seen en el estudiante)
        # Se infiere del step si tiene el campo problem_tags
        tags = step.get("problem_tags", [])
        seen.update(t for t in tags if t in canonical_set)
    return len(seen)


def _success_rate(history: list[dict]) -> float:
    """Problemas resueltos / intentados."""
    n = len(history)
    if n == 0:
        return 0.0
    solved = sum(1 for s in history if s.get("solved", False))
    return solved / n


def _time_used(history: list[dict]) -> float:
    """Tiempo total consumido en la sesion."""
    return sum(s.get("time_min", 0.0) for s in history)


def _empty_metrics(compute_time_s: float = 0.0) -> dict:
    """Metricas vacias para episodios sin pasos."""
    return {
        "total_reward"         : 0.0,
        "spearman_progression" : 0.0,
        "topic_coverage"       : 0,
        "success_rate"         : 0.0,
        "time_used_min"        : 0.0,
        "rating_improvement"   : 0.0,
        "n_attempted"          : 0,
        "n_solved"             : 0,
        "compute_time_s"       : round(compute_time_s, 4),
    }


# ---------------------------------------------------------------------------
# Agregacion de multiples episodios
# ---------------------------------------------------------------------------

def aggregate_metrics(episodes: list[dict]) -> dict:
    """Agrega metricas de multiples episodios (media y desviacion estandar).

    Parameters
    ----------
    episodes : list[dict]
        Lista de dicts devueltos por compute_episode_metrics.

    Returns
    -------
    dict con {metrica_mean, metrica_std} para cada metrica numerica.
    """
    if not episodes:
        return {}

    numeric_keys = [
        "total_reward", "spearman_progression", "topic_coverage",
        "success_rate", "time_used_min", "rating_improvement",
        "n_attempted", "n_solved", "compute_time_s",
    ]

    result = {}
    for key in numeric_keys:
        values = [ep[key] for ep in episodes if key in ep]
        if not values:
            continue
        mean = sum(values) / len(values)
        std  = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        result[f"{key}_mean"] = round(mean, 4)
        result[f"{key}_std"]  = round(std, 4)

    return result