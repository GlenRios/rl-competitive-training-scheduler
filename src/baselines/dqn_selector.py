"""
dqn_selector.py -- Adapter que envuelve DQNAgent en la interfaz ProblemSelector

Coloca este archivo en src/baselines/dqn_selector.py
"""

import numpy as np

from src.baselines.selector import ProblemSelector, SessionState
from src.agent.dqn import DQNAgent
from src.environment.observation_builder import ObservationBuilder


class DQNSelectorAdapter(ProblemSelector):
    """Envuelve un DQNAgent entrenado como ProblemSelector.

    Permite usar el agente DQN en cualquier contexto que espere
    un ProblemSelector: Evaluator, páginas de Streamlit, etc.

    Parameters
    ----------
    agent       : DQNAgent        -- agente con pesos cargados
    obs_builder : ObservationBuilder -- construye las observaciones del entorno
    """

    def __init__(self, agent: DQNAgent, obs_builder: ObservationBuilder) -> None:
        self.agent       = agent
        self.obs_builder = obs_builder

    def select_action(
        self,
        state         : SessionState,
        available_mask: np.ndarray,
        **kwargs,
    ) -> int:
        """Selecciona el problema con mayor Q-valor entre los disponibles.

        Parameters
        ----------
        state          : SessionState
        available_mask : np.ndarray (N,) bool

        Returns
        -------
        int — índice del problema elegido.
        """
        student_obs    = self.obs_builder.build_student_obs(state.student)
        problem_matrix = self.obs_builder.build_problem_matrix(state.student)

        action = self.agent.select_action(
            student_obs    = student_obs,
            problem_matrix = problem_matrix,
            mask           = available_mask,
            epsilon        = 0.0,   # modo explotación pura (sin exploración)
        )
        return action

    def reset(self) -> None:
        """Sin estado interno — nada que reiniciar."""