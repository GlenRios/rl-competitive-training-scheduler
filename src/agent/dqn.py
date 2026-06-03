"""
dqn.py — Red neuronal Q y agente DQN

Responsabilidad única:
    Definir la arquitectura de la red Q y la lógica de selección de acción.
    NO gestiona el entrenamiento ni el buffer — eso es responsabilidad de trainer.py.

Arquitectura de la red Q
-------------------------
    Input  : (FULL_OBS_DIM,) = (29,)
             Concatenación de student_obs (5) + problem_obs_i (24)
             para un único problema candidato.

    Hidden : capas configurables con activación ReLU.
             Default: [128, 64]

    Output : (1,) — valor Q estimado para ese (estado, acción).

Selección de acción
-------------------
    El agente calcula Q(s, a_i) para todos los problemas disponibles,
    aplica la máscara de acciones válidas, y elige:

        - Con probabilidad ε  → acción aleatoria válida  (exploración)
        - Con probabilidad 1-ε → argmax Q válido          (explotación)

    Esto implementa la política ε-greedy estándar de DQN.

Dos redes: online y target
--------------------------
    Se mantienen dos instancias de QNetwork con la misma arquitectura:
    - online_net  : se actualiza en cada paso de entrenamiento.
    - target_net  : se sincroniza periódicamente con online_net.
                    Se usa para calcular los Q-values objetivo en la loss,
                    lo que estabiliza el entrenamiento.

Uso
---
    from src.agent.dqn import DQNAgent
    from src.environment.observation_builder import FULL_OBS_DIM

    agent = DQNAgent(obs_dim=FULL_OBS_DIM, n_actions=500)

    # Seleccionar acción dado el estado actual
    action = agent.select_action(
        student_obs    = obs,          # (5,)
        problem_matrix = p_mat,        # (N, 24)
        action_mask    = mask,         # (N,) bool
        epsilon        = 0.1,
    )

    # Sincronizar target con online
    agent.sync_target()

    # Guardar y cargar checkpoint
    agent.save("experiments/results/agent.pt")
    agent.load("experiments/results/agent.pt")
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.environment.observation_builder import (
    FULL_OBS_DIM,
    STUDENT_OBS_DIM,
    PROBLEM_OBS_DIM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QNetwork — la red neuronal
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP que estima Q(s, a_i) dado el input concatenado [student_obs | problem_obs_i].

    Parameters
    ----------
    obs_dim      : int   — dimensión del input (default FULL_OBS_DIM = 29)
    hidden_sizes : list  — neuronas por capa oculta (default [128, 64])
    """

    def __init__(
        self,
        obs_dim      : int       = FULL_OBS_DIM,
        hidden_sizes : list[int] = None,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64]

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim  = h
        layers.append(nn.Linear(in_dim, 1))   # output: Q(s, a_i) escalar

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (batch, obs_dim)

        Returns
        -------
        Tensor shape (batch, 1)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# DQNAgent — selección de acción + gestión de redes
# ---------------------------------------------------------------------------

class DQNAgent:
    """Agente DQN con política ε-greedy y red target para estabilidad.

    Parameters
    ----------
    obs_dim      : int        — dimensión del input de la red (default 29)
    n_actions    : int        — número total de problemas / acciones posibles
    hidden_sizes : list[int]  — arquitectura de la red oculta
    lr           : float      — learning rate del optimizador Adam
    gamma        : float      — factor de descuento (default 0.99)
    device       : str        — "cpu" o "cuda" (autodetectado si None)
    """

    def __init__(
        self,
        obs_dim      : int              = FULL_OBS_DIM,
        n_actions    : int              = 500,
        hidden_sizes : list[int]        = None,
        lr           : float            = 1e-3,
        gamma        : float            = 0.99,
        device       : Optional[str]    = None,
    ) -> None:
        self.obs_dim      = obs_dim
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.device       = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Dos redes con la misma arquitectura
        self.online_net = QNetwork(obs_dim, hidden_sizes or [128, 64]).to(self.device)
        self.target_net = QNetwork(obs_dim, hidden_sizes or [128, 64]).to(self.device)
        self.sync_target()   # target empieza igual que online
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

        logger.info(
            f"DQNAgent init — device={self.device} | "
            f"obs_dim={obs_dim} | n_actions={n_actions} | "
            f"hidden={hidden_sizes or [128, 64]} | lr={lr}"
        )

    # ------------------------------------------------------------------
    # Selección de acción
    # ------------------------------------------------------------------

    def select_action(
        self,
        student_obs    : np.ndarray,
        problem_matrix : np.ndarray,
        action_mask    : np.ndarray,
        epsilon        : float = 0.0,
    ) -> int:
        """Selecciona una acción usando la política ε-greedy.

        Parameters
        ----------
        student_obs    : np.ndarray shape (STUDENT_OBS_DIM,) = (5,)
        problem_matrix : np.ndarray shape (N, PROBLEM_OBS_DIM) = (N, 24)
        action_mask    : np.ndarray shape (N,) bool — True = acción válida
        epsilon        : float — probabilidad de exploración [0, 1]

        Returns
        -------
        int — índice del problema seleccionado
        """
        valid_indices = np.where(action_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No hay acciones válidas disponibles.")

        # Exploración: acción aleatoria válida
        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        # Explotación: argmax Q sobre acciones válidas
        q_values = self.compute_q_values(student_obs, problem_matrix)
        q_values[~action_mask] = float("-inf")   # enmascarar inválidas
        return int(np.argmax(q_values))

    def compute_q_values(
        self,
        student_obs    : np.ndarray,
        problem_matrix : np.ndarray,
    ) -> np.ndarray:
        """Calcula Q(s, a_i) para todos los problemas.

        Construye el input completo [student_obs | problem_obs_i] para
        cada problema y pasa el batch por la red online.

        Parameters
        ----------
        student_obs    : np.ndarray shape (5,)
        problem_matrix : np.ndarray shape (N, 24)

        Returns
        -------
        np.ndarray shape (N,) — Q(s, a_i) para i = 0…N-1
        """
        n = len(problem_matrix)
        s_tiled    = np.tile(student_obs, (n, 1))                  # (N, 5)
        full_input = np.concatenate([s_tiled, problem_matrix], axis=1)  # (N, 29)

        tensor = torch.tensor(full_input, dtype=torch.float32).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            q_vals = self.online_net(tensor).squeeze(-1)   # (N,)
        self.online_net.train()

        return q_vals.cpu().numpy()

    # ------------------------------------------------------------------
    # Actualización de la red
    # ------------------------------------------------------------------

    def update(
        self,
        state_inputs     : torch.Tensor,
        actions          : torch.Tensor,
        rewards          : torch.Tensor,
        next_full_inputs : torch.Tensor,
        next_masks       : torch.Tensor,
        terminated       : torch.Tensor,
    ) -> float:
        """Realiza un paso de gradiente sobre un batch de transiciones.

        Implementa la actualización estándar de DQN:
            target = r + γ · max_{a' válido} Q_target(s', a')  si no termina
            target = r                                           si termina

        Parameters
        ----------
        state_inputs     : (B, 29) — input de la red para la acción tomada
        actions          : (B,)    — índices de las acciones (no usados directamente,
                                     ya están codificados en state_inputs)
        rewards          : (B,)    — recompensas recibidas
        next_full_inputs : (B, N, 29) — inputs completos del siguiente estado
        next_masks       : (B, N)  — máscara de acciones válidas en s'
        terminated       : (B,)    — True si el episodio terminó

        Returns
        -------
        float — loss del batch
        """
        B = state_inputs.shape[0]

        # Q(s, a) con la red online
        q_current = self.online_net(state_inputs).squeeze(-1)   # (B,)

        # max Q_target(s', a') sobre acciones válidas
        with torch.no_grad():
            # Reshape para pasar por la red: (B*N, 29)
            B, N, D = next_full_inputs.shape
            flat_next = next_full_inputs.view(B * N, D)
            q_next_flat = self.target_net(flat_next).squeeze(-1)   # (B*N,)
            q_next = q_next_flat.view(B, N)                        # (B, N)

            # Enmascarar acciones inválidas con -inf
            q_next[~next_masks] = float("-inf")
            q_next_max = q_next.max(dim=1).values                  # (B,)

            # Si el episodio terminó, no hay Q futuro
            q_next_max[terminated] = 0.0

            target = rewards + self.gamma * q_next_max             # (B,)

        loss = self.loss_fn(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Sincronización target y checkpoints
    # ------------------------------------------------------------------

    def sync_target(self) -> None:
        """Copia los pesos de online_net a target_net."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        logger.debug("Target network sincronizada con online network")

    def save(self, path: str | Path) -> None:
        """Guarda el estado del agente en disco."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net"  : self.online_net.state_dict(),
            "target_net"  : self.target_net.state_dict(),
            "optimizer"   : self.optimizer.state_dict(),
            "obs_dim"     : self.obs_dim,
            "n_actions"   : self.n_actions,
            "gamma"       : self.gamma,
        }, path)
        logger.info(f"Checkpoint guardado en {path}")

    def load(self, path: str | Path) -> None:
        """Carga el estado del agente desde disco."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Checkpoint cargado desde {path}")

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.online_net.parameters())
        return (
            f"DQNAgent(obs_dim={self.obs_dim}, n_actions={self.n_actions}, "
            f"params={n_params:,}, device={self.device})"
        )