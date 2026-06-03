"""
env.py — Módulo 6: Entorno Gymnasium

Responsabilidad única:
    Implementar la interfaz estándar de Gymnasium (reset / step / render)
    coordinando StudentModel, ActionMasker y ObservationBuilder.

    Este archivo es deliberadamente delgado — toda la lógica compleja
    vive en los módulos de soporte.

Espacios
--------
    action_space      : Discrete(N_PROBLEMS)
                        El agente elige el índice del problema a resolver.

    observation_space : Box(low=0, high=1, shape=(STUDENT_OBS_DIM,))
                        Estado del estudiante normalizado.
                        La matriz de problemas se pasa en `info` para que
                        el agente DQN construya el input completo.

Flujo de un episodio
--------------------
    obs, info = env.reset()
    while True:
        mask   = info["action_mask"]          # (N,) bool
        p_mat  = info["problem_matrix"]       # (N, 24) float32
        action = agent.select(obs, p_mat, mask)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Uso
---
    from src.environment.env import TrainingEnv
    from src.environment.problem import Problem

    problems = Problem.from_dataframe(df)
    env      = TrainingEnv(problems, initial_rating=1500, session_budget_min=120)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action=42)
"""

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from src.environment.action_masker import ActionMasker
from src.environment.observation_builder import (
    FULL_OBS_DIM,
    PROBLEM_OBS_DIM,
    STUDENT_OBS_DIM,
    ObservationBuilder,
)
from src.environment.problem import Problem
from src.environment.student_model import StudentModel

logger = logging.getLogger(__name__)


class TrainingEnv(gym.Env):
    """Entorno de simulación de sesión de entrenamiento competitivo.

    Parameters
    ----------
    problems : list[Problem]
        Lista completa de problemas disponibles para la sesión.
    initial_rating : int
        Rating inicial del estudiante simulado.
    session_budget_min : float
        Duración máxima de la sesión en minutos (default: 120).
    random_seed : int | None
        Semilla para reproducibilidad del StudentModel.
    student_kwargs : dict | None
        Parámetros adicionales para el StudentModel
        (theta, lambda_fatigue, t_min, t_max, etc.).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        problems           : list[Problem],
        initial_rating     : int   = 1500,
        session_budget_min : float = 120.0,
        random_seed        : Optional[int] = None,
        student_kwargs     : Optional[dict] = None,
    ) -> None:
        super().__init__()

        if not problems:
            raise ValueError("La lista de problemas no puede estar vacía.")

        self.problems            = problems
        self.n_problems          = len(problems)
        self.initial_rating      = initial_rating
        self.session_budget_min  = session_budget_min
        self.random_seed         = random_seed

        # -- Espacios de Gymnasium --------------------------------------
        self.action_space = gym.spaces.Discrete(self.n_problems)

        self.observation_space = gym.spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (STUDENT_OBS_DIM,),
            dtype = np.float32,
        )

        # -- Módulos de soporte -----------------------------------------
        student_kwargs = student_kwargs or {}
        self._student  = StudentModel(
            initial_rating     = initial_rating,
            session_budget_min = session_budget_min,
            random_seed        = random_seed,
            **student_kwargs,
        )
        self._obs_builder = ObservationBuilder(problems, session_budget_min)
        self._masker      = ActionMasker(problems)

        # -- Contadores de episodio -------------------------------------
        self._episode_reward  : float        = 0.0
        self._step_count      : int          = 0
        self._episode_history : list[dict]   = []

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed    : Optional[int] = None,
        options : Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """Inicia un nuevo episodio.

        Returns
        -------
        obs  : np.ndarray shape (STUDENT_OBS_DIM,)
        info : dict con action_mask y problem_matrix
        """
        super().reset(seed=seed)

        self._student.reset()
        self._masker.reset()
        self._episode_reward  = 0.0
        self._step_count      = 0
        self._episode_history = []

        obs  = self._obs_builder.student_obs(self._student)
        info = self._build_info()

        logger.debug(
            f"reset() — rating={self._student.rating} | "
            f"budget={self.session_budget_min}min | "
            f"n_problems={self.n_problems}"
        )
        return obs, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Ejecuta un paso: el agente selecciona el problema `action`.

        Parameters
        ----------
        action : int
            Índice del problema seleccionado (0 … N_PROBLEMS-1).

        Returns
        -------
        obs        : np.ndarray shape (STUDENT_OBS_DIM,)
        reward     : float
        terminated : bool — True si la sesión terminó
        truncated  : bool — siempre False (no usamos límite de pasos)
        info       : dict con action_mask, problem_matrix y detalles del intento
        """
        # -- Validar acción ---------------------------------------------
        mask = self._masker.get_mask(self._student)
        if not (0 <= action < self.n_problems):
            raise ValueError(
                f"Acción {action} fuera del rango [0, {self.n_problems})."
            )
        if not mask[action]:
            raise ValueError(
                f"Acción {action} no válida — problema ya intentado "
                "o tiempo insuficiente."
            )

        # -- Ejecutar intento -------------------------------------------
        problem = self.problems[action]
        outcome = self._student.attempt(
            problem_rating = problem.rating,
            problem_tags   = problem.tags_list,
            problem_id     = problem.problem_id,
        )

        # -- Actualizar máscara -----------------------------------------
        self._masker.mark_attempted(action)

        # -- Acumuladores ----------------------------------------------
        self._episode_reward += outcome.reward
        self._step_count     += 1

        step_record = {
            "step"          : self._step_count,
            "problem_id"    : problem.problem_id,
            "problem_rating": problem.rating,
            "solved"        : outcome.solved,
            "p_solve"       : outcome.p_solve,
            "time_min"      : outcome.time_min,
            "reward"        : outcome.reward,
            "delta_rating"  : outcome.delta_rating,
            "new_rating"    : outcome.new_rating,
            "fatigue"       : outcome.fatigue,
            "time_remaining": outcome.time_remaining_min,
        }
        self._episode_history.append(step_record)

        logger.debug(
            f"step={self._step_count} | pid={problem.problem_id} | "
            f"solved={outcome.solved} | reward={outcome.reward:.1f} | "
            f"time_left={outcome.time_remaining_min:.1f}m"
        )

        # -- Condición de terminación -----------------------------------
        terminated = (
            outcome.session_over
            or not self._masker.any_valid(self._student)
        )
        truncated  = False

        obs  = self._obs_builder.student_obs(self._student)
        info = self._build_info(step_record=step_record)

        return obs, outcome.reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # render()
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Imprime el estado actual del episodio en consola."""
        sep = "-" * 55
        print(sep)
        print(f"  Paso          : {self._step_count}")
        print(f"  Rating        : {self._student.rating}")
        print(f"  Fatiga        : {self._student.fatigue:.2f}")
        print(f"  Tiempo usado  : {self._student.time_spent_min:.1f} / "
              f"{self.session_budget_min:.0f} min")
        print(f"  Resueltos     : {self._student.n_solved} / "
              f"{self._student.n_attempted}")
        print(f"  Recompensa ep.: {self._episode_reward:.1f}")
        print(f"  Disponibles   : {self._masker.n_available}")
        print(sep)

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    @property
    def episode_history(self) -> list[dict]:
        return self._episode_history

    @property
    def student(self) -> StudentModel:
        return self._student

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _build_info(self, step_record: Optional[dict] = None) -> dict:
        """Construye el diccionario `info` retornado en reset/step."""
        info: dict[str, Any] = {
            "action_mask"    : self._masker.get_mask(self._student),
            "problem_matrix" : self._obs_builder.problem_matrix(self._student),
            "n_available"    : self._masker.n_available,
            "episode_reward" : self._episode_reward,
            "step_count"     : self._step_count,
        }
        if step_record:
            info["step"] = step_record
        return info