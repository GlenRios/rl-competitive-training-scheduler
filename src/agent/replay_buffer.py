"""
replay_buffer.py — Memoria de experiencia (Experience Replay)

Responsabilidad única:
    Almacenar transiciones (s, a, r, s', done) y proveer muestras
    aleatorias para el entrenamiento del agente DQN.

Qué se almacena por transición
--------------------------------
    state_input      : (29,)   — input completo [student_obs | problem_obs_action]
                                 para la acción tomada en este paso.
    action           : int     — índice del problema seleccionado.
    reward           : float   — recompensa inmediata recibida.
    next_student_obs : (5,)    — estado del estudiante tras el intento.
    next_mask        : (N,)    — máscara de acciones válidas en s'.
    terminated       : bool    — True si el episodio terminó.

Por qué NO almacenamos next_problem_matrix
-------------------------------------------
    La matrix de problemas (N, 24) depende del estado del estudiante y
    pesa N×24 floats por transición. Con N=500 y buffer de 10k entradas,
    eso serían ~240 MB solo en problem matrices.

    En cambio almacenamos next_student_obs (5 floats) y dejamos que el
    Trainer reconstruya la next_full_input_matrix en el momento del
    entrenamiento usando el ObservationBuilder — que tiene acceso a la
    lista de problemas y puede hacer el cálculo en batch eficientemente.

Capacidad y política de descarte
---------------------------------
    Al alcanzar la capacidad máxima, las transiciones más antiguas son
    reemplazadas (política FIFO mediante un índice circular).

Uso
---
    from src.agent.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=10_000)
    buffer.push(Transition(...))

    if buffer.is_ready(min_size=64):
        batch = buffer.sample(batch_size=64)
"""

import logging
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transition — una experiencia almacenada
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Una transición (s, a, r, s', done) del entorno.

    Attributes
    ----------
    state_input      : np.ndarray (29,)  — [student_obs | problem_obs[action]]
    action           : int               — índice del problema seleccionado
    reward           : float             — recompensa inmediata
    next_student_obs : np.ndarray (5,)   — estado del estudiante en s'
    next_mask        : np.ndarray (N,) bool — acciones válidas en s'
    terminated       : bool              — True si el episodio terminó
    """
    state_input      : np.ndarray
    action           : int
    reward           : float
    next_student_obs : np.ndarray
    next_mask        : np.ndarray
    terminated       : bool


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Buffer circular de experiencias para DQN.

    Parameters
    ----------
    capacity : int
        Número máximo de transiciones almacenadas.
        Al llenarse, las más antiguas se descartan (FIFO).
    """

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity debe ser positivo.")
        self.capacity = capacity
        self._buffer  = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Añade una transición al buffer.

        Si el buffer está lleno, la transición más antigua se descarta.
        """
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Devuelve una muestra aleatoria de transiciones sin reemplazo.

        Parameters
        ----------
        batch_size : int
            Número de transiciones a muestrear.

        Returns
        -------
        list[Transition] de longitud batch_size.

        Raises
        ------
        ValueError si batch_size > len(buffer).
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size={batch_size} mayor que el buffer actual "
                f"({len(self._buffer)} transiciones)."
            )
        return random.sample(self._buffer, batch_size)

    def is_ready(self, min_size: int) -> bool:
        """True si hay suficientes transiciones para un batch de entrenamiento."""
        return len(self._buffer) >= min_size

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) == self.capacity

    @property
    def fill_ratio(self) -> float:
        """Fracción del buffer ocupada [0.0, 1.0]."""
        return len(self._buffer) / self.capacity

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={len(self._buffer)}, "
            f"capacity={self.capacity}, "
            f"fill={self.fill_ratio:.1%})"
        )