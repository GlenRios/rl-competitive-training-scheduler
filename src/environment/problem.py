"""
problem.py — Clase Problem

Responsabilidad:
    Representar un problema de programación competitiva como entidad
    del sistema. Centraliza los datos del problema y produce el vector
    de observación que el agente DQN recibe para tomar decisiones.

Vector de observación por problema
------------------------------------
    El agente necesita una representación numérica fija de cada problema
    candidato. Este vector tiene dos partes:

    Parte 1 — Características del problema (4 valores):
        [0] rating_norm    — dificultad normalizada en [0, 1]
        [1] gap_norm       — (problem_rating - student_rating) / 2700, en [-1, 1]
        [2] p_solve        — probabilidad de resolverlo dado el estado actual
        [3] time_norm      — solve_time_min estimado / session_budget

    Parte 2 — Tags en one-hot (N_TOPICS = 20 valores):
        Un 1 en la posición i indica que el problema involucra el tema i.

    Total: 4 + 20 = 24 valores por problema.

Temas canónicos (CANONICAL_TOPICS)
------------------------------------
    Lista fija de los 20 temas más frecuentes en Codeforces.
    Cualquier tag fuera de esta lista se ignora en el one-hot.
    El orden importa — no modificar sin regenerar el dataset.

Uso
---
    from src.environment.problem import Problem

    # Desde una fila del DataFrame procesado
    problem = Problem.from_row(df.iloc[0])

    # Lista completa desde el DataFrame
    problems = Problem.from_dataframe(df)

    # Vector para el agente dado el estado actual del estudiante
    vec = problem.to_observation_vector(
        student_rating=1500,
        student_fatigue=0.2,
        session_budget_min=120.0,
        time_spent_min=30.0,
    )
"""

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temas canónicos
#
# Los 20 tags más frecuentes en Codeforces, en orden de frecuencia
# aproximada. Este orden define el índice en el vector one-hot.
# ---------------------------------------------------------------------------

CANONICAL_TOPICS: list[str] = [
    "implementation",        # 0
    "math",                  # 1
    "greedy",                # 2
    "dp",                    # 3
    "data structures",       # 4
    "constructive algorithms", # 5
    "graphs",                # 6
    "strings",               # 7
    "sortings",              # 8
    "binary search",         # 9
    "brute force",           # 10
    "number theory",         # 11
    "combinatorics",         # 12
    "trees",                 # 13
    "dfs and similar",       # 14
    "two pointers",          # 15
    "bitmasks",              # 16
    "geometry",              # 17
    "shortest paths",        # 18
    "hashing",               # 19
]

N_TOPICS   = len(CANONICAL_TOPICS)           # 20
_TOPIC_IDX = {t: i for i, t in enumerate(CANONICAL_TOPICS)}

# Rango válido de rating en Codeforces
_MIN_RATING = 800
_MAX_RATING = 3500
_RATING_RANGE = _MAX_RATING - _MIN_RATING    # 2700


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """Representa un problema de programación competitiva.

    Attributes
    ----------
    problem_id : str
        Identificador único (ej. "325A").
    name : str
        Título del problema.
    rating : int
        Dificultad numérica (800–3500).
    tags_list : list[str]
        Lista de temas del problema (ej. ["dp", "graphs"]).
    statement : str
        Enunciado textual (usado para contexto, no para el agente).
    difficulty_band : str
        Banda de dificultad asignada por el Preprocessor (ej. "1200-1599").
    time_limit_ms : int | None
        Límite de tiempo del juez en milisegundos. None si no disponible.
    solved_count : int | None
        Número de usuarios que resolvieron el problema. None si no disponible.
    """

    problem_id      : str
    name            : str
    rating          : int
    tags_list       : list[str]
    statement       : str
    difficulty_band : str             = "unknown"
    time_limit_ms   : Optional[int]   = None
    solved_count    : Optional[int]   = None

    # ------------------------------------------------------------------
    # Construcción desde DataFrame
    # ------------------------------------------------------------------

    @classmethod
    def from_row(cls, row: pd.Series) -> "Problem":
        """Construye un Problem desde una fila del DataFrame procesado.

        Acepta tanto el formato de `tags_list` como columna Python list,
        o como string representación de lista.
        """
        tags_list = row.get("tags_list", [])
        if isinstance(tags_list, str):
            try:
                tags_list = ast.literal_eval(tags_list)
            except (ValueError, SyntaxError):
                tags_list = [t.strip() for t in tags_list.split(",") if t.strip()]

        return cls(
            problem_id      = str(row["problem_id"]),
            name            = str(row.get("name", "")) if pd.notna(row.get("name")) else "",
            rating          = int(row["rating"]),
            tags_list       = [t.lower() for t in tags_list if isinstance(t, str)],
            statement       = str(row.get("statement", "")),
            difficulty_band = str(row.get("difficulty_band", "unknown")),
            time_limit_ms   = int(row["time_limit_ms"])
                              if pd.notna(row.get("time_limit_ms")) else None,
            solved_count    = int(row["solved_count"])
                              if pd.notna(row.get("solved_count")) else None,
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["Problem"]:
        """Construye la lista completa de Problems desde el DataFrame procesado.

        Parameters
        ----------
        df : pd.DataFrame
            Salida del Preprocessor con columnas: problem_id, name, rating,
            tags_list, statement, difficulty_band.

        Returns
        -------
        list[Problem] en el mismo orden que el DataFrame.
        """
        problems = []
        skipped  = 0
        for _, row in df.iterrows():
            try:
                problems.append(cls.from_row(row))
            except (KeyError, ValueError, TypeError) as e:
                skipped += 1
                logger.warning(f"Fila omitida al construir Problem: {e}")

        if skipped:
            logger.warning(f"  {skipped} filas omitidas de {len(df)} totales")

        logger.info(f"Problem.from_dataframe -> {len(problems)} problemas cargados")
        return problems

    # ------------------------------------------------------------------
    # Vector de observación para el agente DQN
    # ------------------------------------------------------------------

    def to_observation_vector(
        self,
        student_rating    : int,
        student_fatigue   : float,
        session_budget_min: float,
        time_spent_min    : float,
        p_solve           : float,
    ) -> list[float]:
        """Construye el vector numérico que el agente DQN recibe para este problema.

        Parameters
        ----------
        student_rating : int
            Rating actual del estudiante.
        student_fatigue : float
            Nivel de fatiga actual [0, 1].
        session_budget_min : float
            Presupuesto total de la sesión en minutos.
        time_spent_min : float
            Minutos ya consumidos en la sesión.
        p_solve : float
            Probabilidad de resolver este problema dado el estado actual
            del estudiante (calculada por StudentModel).

        Returns
        -------
        list[float] de longitud OBS_DIM = 4 + N_TOPICS = 24.
            Todos los valores en [0, 1] o [-1, 1] según la componente.
        """
        time_remaining = max(0.0, session_budget_min - time_spent_min)

        # -- Parte 1: características numéricas (4 valores) -------------

        rating_norm = (self.rating - _MIN_RATING) / _RATING_RANGE

        gap         = self.rating - student_rating
        gap_norm    = max(-1.0, min(1.0, gap / _RATING_RANGE))

        # Tiempo estimado de resolución (basado en gap y fatiga)
        solve_time  = self._estimate_time(gap, student_fatigue)
        time_norm   = min(1.0, solve_time / session_budget_min) \
                      if session_budget_min > 0 else 1.0

        # -- Parte 2: tags one-hot (N_TOPICS valores) -------------------

        tags_onehot = self._tags_to_onehot()

        return [
            round(rating_norm, 4),
            round(gap_norm,    4),
            round(p_solve,     4),
            round(time_norm,   4),
        ] + tags_onehot

    def _estimate_time(self, gap: int, fatigue: float) -> float:
        """Estimación rápida de tiempo basada en gap y fatiga.

        Usa la misma tabla que StudentModel para consistencia.
        No depende de StudentModel directamente para evitar acoplamiento.
        """
        _GAP_MULT = [
            (-9999, -401, 0.40),
            (-400,  -201, 0.60),
            (-200,   -1,  0.80),
            (0,      199, 1.00),
            (200,    399, 1.60),
            (400,    599, 2.50),
            (600,   9999, 4.00),
        ]
        _BASE = [
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
        base = next((t for lo, hi, t in _BASE if lo <= self.rating <= hi), 35.0)
        mult = next((m for lo, hi, m in _GAP_MULT if lo <= gap <= hi), 1.0)
        return base * mult * (1.0 + fatigue * 0.3)

    def _tags_to_onehot(self) -> list[float]:
        """Convierte tags_list a vector one-hot sobre CANONICAL_TOPICS."""
        vec = [0.0] * N_TOPICS
        for tag in self.tags_list:
            idx = _TOPIC_IDX.get(tag)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Dimensión del vector de observación: 4 + N_TOPICS."""
        return 4 + N_TOPICS   # = 24

    @property
    def canonical_tags(self) -> list[str]:
        """Subconjunto de tags_list que están en CANONICAL_TOPICS."""
        return [t for t in self.tags_list if t in _TOPIC_IDX]

    @property
    def n_canonical_tags(self) -> int:
        return len(self.canonical_tags)

    @property
    def is_untagged(self) -> bool:
        """True si el problema no tiene ningún tag canónico."""
        return self.n_canonical_tags == 0

    def __repr__(self) -> str:
        return (
            f"Problem(id={self.problem_id!r}, "
            f"rating={self.rating}, "
            f"tags={self.tags_list}, "
            f"band={self.difficulty_band!r})"
        )