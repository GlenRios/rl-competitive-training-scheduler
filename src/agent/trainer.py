"""
trainer.py — Loop de entrenamiento DQN

Responsabilidad única:
    Orquestar el entrenamiento del agente DQN coordinando:
        - TrainingEnv  (entorno)
        - DQNAgent     (red Q + selección de acción)
        - ReplayBuffer (memoria de experiencia)
        - ObservationBuilder (reconstrucción de inputs para el batch)

Algoritmo implementado
-----------------------
    DQN estándar con las siguientes características:
    - eps-greedy con decaimiento lineal (eps_start -> eps_end en N episodios)
    - Experience replay con muestreo aleatorio uniforme
    - Red target sincronizada cada `target_sync_every` episodios
    - Gradient clipping (dentro de DQNAgent.update)
    - Checkpoint automático del mejor agente

Loop por episodio
-----------------
    1. env.reset() -> obs, info
    2. while not terminated:
        a. agent.select_action(obs, problem_matrix, mask, epsilon)
        b. env.step(action) -> next_obs, reward, terminated, truncated, info
        c. buffer.push(Transition(...))
        d. if buffer.is_ready: agent.update(buffer.sample(...))
    3. if episodio % target_sync_every == 0: agent.sync_target()
    4. Log de métricas

Uso
---
    from src.agent.trainer import DQNTrainer

    trainer = DQNTrainer(env=env, agent=agent, buffer=buffer, obs_builder=obs_builder)
    metrics = trainer.train(n_episodes=500)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.agent.dqn import DQNAgent
from src.agent.replay_buffer import ReplayBuffer, Transition
from src.environment.env import TrainingEnv
from src.environment.observation_builder import ObservationBuilder

logger = logging.getLogger(__name__)


class DQNTrainer:
    """Orquesta el entrenamiento del agente DQN.

    Parameters
    ----------
    env              : TrainingEnv    — entorno de simulación
    agent            : DQNAgent       — agente con red Q y política eps-greedy
    buffer           : ReplayBuffer   — memoria de experiencia
    obs_builder      : ObservationBuilder — para reconstruir inputs del batch
    batch_size       : int            — tamaño del batch de entrenamiento
    epsilon_start    : float          — exploración inicial (default 1.0)
    epsilon_end      : float          — exploración mínima (default 0.05)
    epsilon_decay_ep : int            — episodios hasta alcanzar epsilon_end
    target_sync_every: int            — sincronizar target cada N episodios
    checkpoint_dir   : str | Path     — directorio para guardar checkpoints
    log_every        : int            — loggear métricas cada N episodios
    """

    def __init__(
        self,
        env              : TrainingEnv,
        agent            : DQNAgent,
        buffer           : ReplayBuffer,
        obs_builder      : ObservationBuilder,
        batch_size       : int   = 64,
        epsilon_start    : float = 1.0,
        epsilon_end      : float = 0.05,
        epsilon_decay_ep : int   = 300,
        target_sync_every: int   = 50,
        checkpoint_dir   : str | Path = "experiments/results/",
        log_every        : int   = 10,
        rest_every       : int   = 100,
        rest_seconds     : float = 2.0,
    ) -> None:
        self.env               = env
        self.agent             = agent
        self.buffer            = buffer
        self.obs_builder       = obs_builder
        self.batch_size        = batch_size
        self.epsilon_start     = epsilon_start
        self.epsilon_end       = epsilon_end
        self.epsilon_decay_ep  = epsilon_decay_ep
        self.target_sync_every = target_sync_every
        self.checkpoint_dir    = Path(checkpoint_dir)
        self.log_every         = log_every
        self.rest_every        = rest_every
        self.rest_seconds      = rest_seconds

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Historial de métricas por episodio
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def train(self, n_episodes: int) -> list[dict]:
        """Ejecuta el loop de entrenamiento completo.

        Parameters
        ----------
        n_episodes : int — número de episodios a entrenar

        Returns
        -------
        list[dict] — historial de métricas por episodio:
            episode, epsilon, reward, n_solved, n_attempted,
            final_rating, avg_loss, duration_s
        """
        separator = "-" * 65
        logger.info(separator)
        logger.info(f"  ENTRENAMIENTO DQN")
        logger.info(f"  Episodios      : {n_episodes}")
        logger.info(f"  Batch size     : {self.batch_size}")
        logger.info(f"  Buffer capacity: {self.buffer.capacity}")
        logger.info(f"  eps: {self.epsilon_start} -> {self.epsilon_end} en {self.epsilon_decay_ep} ep.")
        logger.info(f"  Target sync    : cada {self.target_sync_every} ep.")
        logger.info(separator)

        best_ckpt = self.checkpoint_dir / "best_agent.pt"
        best_reward = float("-inf")
        if best_ckpt.exists():
            try:
                best_reward = self.agent.load(best_ckpt)
                logger.info(
                    f"  Checkpoint previo detectado — best_reward={best_reward:.2f}. "
                    "Solo se sobreescribirá si se supera."
                )
            except RuntimeError:
                logger.warning(
                    f"  Checkpoint en {best_ckpt} incompatible con la arquitectura "
                    "actual — se ignora y se entrena desde cero."
                )
                best_reward = float("-inf")
        total_steps    = 0

        for ep in range(1, n_episodes + 1):
            epsilon = self._compute_epsilon(ep)
            t_start = time.time()

            ep_reward, ep_losses, ep_steps = self._run_episode(epsilon)
            total_steps += ep_steps

            # Sincronizar target periódicamente
            if ep % self.target_sync_every == 0:
                self.agent.sync_target()

            # Métricas del episodio
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            metrics  = {
                "episode"      : ep,
                "epsilon"      : round(epsilon, 4),
                "reward"       : round(ep_reward, 2),
                "n_solved"     : self.env.student.n_solved,
                "n_attempted"  : self.env.student.n_attempted,
                "solve_rate"   : round(self.env.student.solve_rate, 3),
                "final_rating" : self.env.student.rating,
                "avg_loss"     : round(avg_loss, 5),
                "buffer_size"  : len(self.buffer),
                "steps"        : ep_steps,
                "total_steps"  : total_steps,
                "duration_s"   : round(time.time() - t_start, 2),
            }
            self._history.append(metrics)

            # Guardar mejor checkpoint
            if ep_reward > best_reward:
                best_reward = ep_reward
                self.agent.save(self.checkpoint_dir / "best_agent.pt", best_reward=best_reward)

            # Pausa periodica para no sobrecalentar la maquina
            if self.rest_every > 0 and ep % self.rest_every == 0:
                time.sleep(self.rest_seconds)

            # Log periódico
            if ep % self.log_every == 0 or ep == 1:
                logger.info(
                    f"  Ep {ep:>4}/{n_episodes} | "
                    f"eps={epsilon:.3f} | "
                    f"reward={ep_reward:>7.1f} | "
                    f"solved={self.env.student.n_solved}/{self.env.student.n_attempted} | "
                    f"rating={self.env.student.rating} | "
                    f"loss={avg_loss:.4f} | "
                    f"buf={len(self.buffer)} | "
                    f"{metrics['duration_s']}s"
                )

        # Checkpoint final
        self.agent.save(self.checkpoint_dir / "final_agent.pt")
        logger.info(separator)
        logger.info(f"  Entrenamiento completo — mejor reward: {best_reward:.1f}")
        logger.info(separator)

        return self._history

    # ------------------------------------------------------------------
    # Episodio
    # ------------------------------------------------------------------

    def _run_episode(self, epsilon: float) -> tuple[float, list[float], int]:
        """Ejecuta un episodio completo.

        Returns
        -------
        (episode_reward, losses, n_steps)
        """
        obs, info = self.env.reset()
        terminated = False
        losses     = []
        steps      = 0

        while not terminated:
            mask    = info["action_mask"]        # (N,) bool
            p_mat   = info["problem_matrix"]     # (N, 24)
            student = self.env.student

            # 1. Seleccionar acción
            action = self.agent.select_action(
                student_obs    = obs,
                problem_matrix = p_mat,
                action_mask    = mask,
                epsilon        = epsilon,
            )

            # 2. Input de la red para la acción elegida
            state_input = self.obs_builder.single_problem_input(student, action)

            # 3. Ejecutar acción en el entorno
            next_obs, reward, terminated, _, next_info = self.env.step(action)

            # 4. Almacenar transición en el buffer
            transition = Transition(
                state_input      = state_input,
                action           = action,
                reward           = float(reward),
                next_student_obs = next_obs,
                next_mask        = next_info["action_mask"],
                terminated       = terminated,
            )
            self.buffer.push(transition)

            # 5. Entrenar si hay suficientes experiencias
            if self.buffer.is_ready(self.batch_size):
                loss = self._update(self.buffer.sample(self.batch_size))
                losses.append(loss)

            obs   = next_obs
            info  = next_info
            steps += 1

        return self.env.episode_reward, losses, steps

    # ------------------------------------------------------------------
    # Actualización de la red
    # ------------------------------------------------------------------

    def _update(self, batch: list[Transition]) -> float:
        """Prepara el batch y llama a agent.update().

        Aquí se reconstruye next_full_input_matrix para cada transición
        usando el ObservationBuilder, que tiene acceso a la lista de
        problemas y al estado del siguiente estudiante.
        """
        device = self.agent.device
        N      = self.env.n_problems

        state_inputs     = torch.tensor(
            np.stack([t.state_input for t in batch]),
            dtype=torch.float32
        ).to(device)

        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32
        ).to(device)

        terminated = torch.tensor(
            [t.terminated for t in batch], dtype=torch.bool
        ).to(device)

        next_masks = torch.tensor(
            np.stack([t.next_mask for t in batch]),
            dtype=torch.bool
        ).to(device)

        # Reconstruir next_full_input_matrix para cada transición del batch
        # Forma final: (B, N, FULL_OBS_DIM)
        next_full_inputs = self._build_next_full_inputs(batch)
        next_full_inputs = torch.tensor(
            next_full_inputs, dtype=torch.float32
        ).to(device)

        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long
        ).to(device)

        return self.agent.update(
            state_inputs     = state_inputs,
            actions          = actions,
            rewards          = rewards,
            next_full_inputs = next_full_inputs,
            next_masks       = next_masks,
            terminated       = terminated,
        )

    def _build_next_full_inputs(
        self, batch: list[Transition]
    ) -> np.ndarray:
        """Reconstruye la matriz (B, N, 29) para los siguientes estados.

        Para cada transición, necesitamos [next_student_obs | problem_obs_i]
        para cada uno de los N problemas. Los problem_obs_i dependen del
        estado del estudiante, por lo que hay que recalcularlos.

        Returns
        -------
        np.ndarray shape (B, N, FULL_OBS_DIM)
        """
        B    = len(batch)
        N    = self.obs_builder.n_problems
        D    = self.obs_builder.session_budget_min
        result = np.zeros((B, N, self.agent.obs_dim), dtype=np.float32)

        for b_idx, transition in enumerate(batch):
            # Reconstruir estado temporal del estudiante para obtener
            # los problem obs con los valores correctos de gap y p_solve.
            # Usamos el state_vector almacenado (next_student_obs).
            # Como aproximación eficiente, usamos los problem features
            # pre-calculados en info["problem_matrix"] que se guardaron
            # indirectamente via obs_builder en el siguiente step.

            # NOTA: Esta es una aproximación. Los problem_obs dependen
            # de (student_rating, student_fatigue, time_spent) que
            # están codificados en next_student_obs.
            # Para la arquitectura actual (29 dims), recalcular exacto
            # requeriría el StudentModel completo; usamos la info
            # almacenada en next_student_obs como proxy.

            s_obs = transition.next_student_obs   # (5,)
            s_tiled = np.tile(s_obs, (N, 1))      # (N, 5)

            # Usar problem_matrix del snapshot más reciente del obs_builder
            # (calculado con el estado del estudiante en ese momento)
            # Como proxy: usamos los features del problema sin dependencia
            # del estado (rating_norm, gap basado en student obs, onehot)
            p_mat = self._approx_problem_matrix(s_obs)  # (N, 24)

            full = np.concatenate([s_tiled, p_mat], axis=1)  # (N, 29)
            result[b_idx] = full

        return result

    def _approx_problem_matrix(self, student_obs: np.ndarray) -> np.ndarray:
        """Aproxima problem_matrix desde student_obs sin StudentModel completo.

        Usa los Problem objects del obs_builder para reconstruir los
        features independientes del estado (rating_norm, onehot) y
        aproxima gap_norm, p_solve y time_norm desde student_obs.

        student_obs = [rating_norm, fatigue, time_used_norm, solve_rate, topics_norm]
        """
        from src.environment.observation_builder import STUDENT_OBS_DIM, PROBLEM_OBS_DIM
        from src.environment.problem import _MIN_RATING, _MAX_RATING, N_TOPICS

        _RATING_RANGE = _MAX_RATING - _MIN_RATING
        N             = self.obs_builder.n_problems

        # Extraer info del student_obs
        student_rating_norm = float(student_obs[0])
        student_fatigue     = float(student_obs[1])
        student_rating      = int(student_rating_norm * _RATING_RANGE + _MIN_RATING)
        budget              = self.obs_builder.session_budget_min
        time_spent          = float(student_obs[2]) * budget

        rows = []
        for problem in self.obs_builder.problems:
            # Aproximar p_solve con la fórmula sigmoid simple
            import math
            gap      = problem.rating - student_rating
            x        = (student_rating - problem.rating) / 400.0 - 0.5 * student_fatigue
            p_solve  = 1.0 / (1.0 + math.exp(-x))
            p_solve  = max(0.0, min(1.0, p_solve))

            vec = problem.to_observation_vector(
                student_rating     = student_rating,
                student_fatigue    = student_fatigue,
                session_budget_min = budget,
                time_spent_min     = time_spent,
                p_solve            = p_solve,
            )
            rows.append(vec)

        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------

    def _compute_epsilon(self, episode: int) -> float:
        """Decaimiento lineal de eps desde epsilon_start hasta epsilon_end."""
        ratio = min(1.0, (episode - 1) / max(1, self.epsilon_decay_ep - 1))
        return self.epsilon_start + ratio * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def best_reward(self) -> float:
        if not self._history:
            return float("-inf")
        return max(m["reward"] for m in self._history)