"""
tests/test_agent.py

Tests para los tres módulos del agente DQN:
    - QNetwork / DQNAgent  (dqn.py)
    - ReplayBuffer         (replay_buffer.py)
    - DQNTrainer           (trainer.py)

Ejecutar con:
    pytest tests/test_agent.py -v
"""

import numpy as np
import pytest
import torch

from src.agent.dqn import DQNAgent, QNetwork
from src.agent.replay_buffer import ReplayBuffer, Transition
from src.agent.trainer import DQNTrainer
from src.environment.env import TrainingEnv
from src.environment.observation_builder import (
    FULL_OBS_DIM, STUDENT_OBS_DIM, PROBLEM_OBS_DIM, ObservationBuilder
)
from src.environment.problem import Problem
from src.environment.student_model import StudentModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PROBLEMS = 8
N_ACTIONS  = N_PROBLEMS

def make_problems(n: int = N_PROBLEMS) -> list[Problem]:
    return [
        Problem(
            problem_id      = f"{i+1}A",
            name            = f"Problem {i+1}",
            rating          = 800 + i * 150,
            tags_list       = ["math", "dp"] if i % 2 == 0 else ["greedy"],
            statement       = f"Statement {i+1} " * 10,
            difficulty_band = "800-1199",
        )
        for i in range(n)
    ]


def make_student_obs() -> np.ndarray:
    # STUDENT_OBS_DIM = 24 (4 scalars + 20 topic ratings)
    return np.random.rand(STUDENT_OBS_DIM).astype(np.float32)


def make_problem_matrix(n: int = N_PROBLEMS) -> np.ndarray:
    return np.random.rand(n, PROBLEM_OBS_DIM).astype(np.float32)


def make_transition(n_actions: int = N_ACTIONS) -> Transition:
    return Transition(
        state_input      = np.random.rand(FULL_OBS_DIM).astype(np.float32),  # 48
        action           = 0,
        reward           = 1.0,
        next_student_obs = np.random.rand(STUDENT_OBS_DIM).astype(np.float32),  # 24
        next_mask        = np.ones(n_actions, dtype=bool),
        terminated       = False,
    )


@pytest.fixture
def problems():
    return make_problems()


@pytest.fixture
def agent():
    return DQNAgent(obs_dim=FULL_OBS_DIM, n_actions=N_ACTIONS, hidden_sizes=[32, 16])


@pytest.fixture
def buffer():
    return ReplayBuffer(capacity=100)


@pytest.fixture
def env(problems):
    return TrainingEnv(problems, session_budget_min=120, random_seed=0)


@pytest.fixture
def obs_builder(problems):
    return ObservationBuilder(problems, session_budget_min=120.0)


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------

class TestQNetwork:
    def test_forward_output_shape(self):
        net   = QNetwork(obs_dim=29, hidden_sizes=[32, 16])
        x     = torch.rand(4, 29)
        out   = net(x)
        assert out.shape == (4, 1)

    def test_single_input_output_shape(self):
        net = QNetwork(obs_dim=29, hidden_sizes=[32])
        x   = torch.rand(1, 29)
        assert net(x).shape == (1, 1)

    def test_output_is_float(self):
        net = QNetwork(obs_dim=29)
        x   = torch.rand(2, 29)
        assert net(x).dtype == torch.float32

    def test_default_hidden_sizes(self):
        net    = QNetwork(obs_dim=29)
        params = sum(p.numel() for p in net.parameters())
        assert params > 0


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class TestDQNAgent:
    def test_repr_contains_n_actions(self, agent):
        assert str(N_ACTIONS) in repr(agent)

    def test_compute_q_values_shape(self, agent):
        s_obs = make_student_obs()
        p_mat = make_problem_matrix()
        q     = agent.compute_q_values(s_obs, p_mat)
        assert q.shape == (N_PROBLEMS,)

    def test_compute_q_values_dtype(self, agent):
        q = agent.compute_q_values(make_student_obs(), make_problem_matrix())
        assert q.dtype == np.float32

    def test_select_action_returns_valid_index(self, agent):
        s_obs = make_student_obs()
        p_mat = make_problem_matrix()
        mask  = np.ones(N_ACTIONS, dtype=bool)
        action = agent.select_action(s_obs, p_mat, mask, epsilon=0.0)
        assert 0 <= action < N_ACTIONS

    def test_select_action_respects_mask(self, agent):
        s_obs = make_student_obs()
        p_mat = make_problem_matrix()
        # Solo la acción 3 es válida
        mask  = np.zeros(N_ACTIONS, dtype=bool)
        mask[3] = True
        for _ in range(10):
            action = agent.select_action(s_obs, p_mat, mask, epsilon=1.0)
            assert action == 3

    def test_select_action_raises_on_empty_mask(self, agent):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        with pytest.raises(ValueError, match="No hay acciones"):
            agent.select_action(make_student_obs(), make_problem_matrix(), mask)

    def test_sync_target_copies_weights(self, agent):
        # Modificar pesos de online_net
        with torch.no_grad():
            for p in agent.online_net.parameters():
                p.fill_(1.0)
        agent.sync_target()
        for po, pt in zip(
            agent.online_net.parameters(), agent.target_net.parameters()
        ):
            assert torch.allclose(po, pt)

    def test_update_returns_float_loss(self, agent):
        B = 4
        state_inputs     = torch.rand(B, FULL_OBS_DIM)
        actions          = torch.zeros(B, dtype=torch.long)
        rewards          = torch.rand(B)
        next_full_inputs = torch.rand(B, N_PROBLEMS, FULL_OBS_DIM)
        next_masks       = torch.ones(B, N_PROBLEMS, dtype=torch.bool)
        terminated       = torch.zeros(B, dtype=torch.bool)
        loss = agent.update(
            state_inputs, actions, rewards,
            next_full_inputs, next_masks, terminated,
        )
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_update_changes_online_weights(self, agent):
        before = [p.clone() for p in agent.online_net.parameters()]
        B = 4
        agent.update(
            state_inputs     = torch.rand(B, FULL_OBS_DIM),
            actions          = torch.zeros(B, dtype=torch.long),
            rewards          = torch.rand(B),
            next_full_inputs = torch.rand(B, N_PROBLEMS, FULL_OBS_DIM),
            next_masks       = torch.ones(B, N_PROBLEMS, dtype=torch.bool),
            terminated       = torch.zeros(B, dtype=torch.bool),
        )
        after = list(agent.online_net.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed

    def test_save_and_load(self, agent, tmp_path):
        path = tmp_path / "agent.pt"
        agent.save(path)
        assert path.exists()
        # Modificar pesos y luego cargar
        with torch.no_grad():
            for p in agent.online_net.parameters():
                p.fill_(99.0)
        agent.load(path)
        # Los pesos deben haberse restaurado (no son 99)
        any_restored = any(
            not torch.all(p == 99.0)
            for p in agent.online_net.parameters()
        )
        assert any_restored


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def test_empty_at_start(self, buffer):
        assert len(buffer) == 0

    def test_push_increases_size(self, buffer):
        buffer.push(make_transition())
        assert len(buffer) == 1

    def test_sample_returns_correct_size(self, buffer):
        for _ in range(20):
            buffer.push(make_transition())
        batch = buffer.sample(10)
        assert len(batch) == 10

    def test_sample_raises_when_not_enough(self, buffer):
        buffer.push(make_transition())
        with pytest.raises(ValueError):
            buffer.sample(50)

    def test_is_ready_false_when_empty(self, buffer):
        assert not buffer.is_ready(64)

    def test_is_ready_true_when_enough(self, buffer):
        for _ in range(10):
            buffer.push(make_transition())
        assert buffer.is_ready(5)

    def test_capacity_respected(self):
        buf = ReplayBuffer(capacity=5)
        for _ in range(10):
            buf.push(make_transition())
        assert len(buf) == 5

    def test_fill_ratio_correct(self, buffer):
        for _ in range(50):
            buffer.push(make_transition())
        assert buffer.fill_ratio == pytest.approx(0.5)

    def test_is_full_when_at_capacity(self):
        buf = ReplayBuffer(capacity=3)
        for _ in range(3):
            buf.push(make_transition())
        assert buf.is_full

    def test_raises_on_zero_capacity(self):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0)

    def test_transitions_have_correct_types(self, buffer):
        buffer.push(make_transition())
        t = buffer.sample(1)[0]
        assert isinstance(t, Transition)
        assert isinstance(t.state_input, np.ndarray)
        assert isinstance(t.reward, float)
        assert isinstance(t.terminated, bool)


# ---------------------------------------------------------------------------
# DQNTrainer
# ---------------------------------------------------------------------------

class TestDQNTrainer:
    @pytest.fixture
    def trainer(self, env, agent, buffer, obs_builder, tmp_path):
        return DQNTrainer(
            env               = env,
            agent             = agent,
            buffer            = buffer,
            obs_builder       = obs_builder,
            batch_size        = 8,
            epsilon_start     = 1.0,
            epsilon_end       = 0.5,
            epsilon_decay_ep  = 5,
            target_sync_every = 2,
            log_every         = 2,
            checkpoint_dir    = tmp_path,
        )

    def test_epsilon_at_start(self, trainer):
        assert trainer._compute_epsilon(1) == pytest.approx(1.0)

    def test_epsilon_at_end(self, trainer):
        assert trainer._compute_epsilon(100) == pytest.approx(0.5)

    def test_epsilon_decreases(self, trainer):
        eps = [trainer._compute_epsilon(ep) for ep in range(1, 10)]
        assert all(eps[i] >= eps[i+1] for i in range(len(eps)-1))

    def test_train_returns_history(self, trainer):
        history = trainer.train(n_episodes=3)
        assert isinstance(history, list)
        assert len(history) == 3

    def test_history_has_required_keys(self, trainer):
        history = trainer.train(n_episodes=2)
        required = {"episode", "epsilon", "reward", "n_solved", "avg_loss"}
        assert required.issubset(history[0].keys())

    def test_episode_numbers_sequential(self, trainer):
        history = trainer.train(n_episodes=3)
        assert [m["episode"] for m in history] == [1, 2, 3]

    def test_buffer_grows_during_training(self, trainer):
        assert len(trainer.buffer) == 0
        trainer.train(n_episodes=2)
        assert len(trainer.buffer) > 0

    def test_best_checkpoint_saved(self, trainer, tmp_path):
        trainer.checkpoint_dir = tmp_path
        trainer.train(n_episodes=2)
        assert (tmp_path / "best_agent.pt").exists()

    def test_final_checkpoint_saved(self, trainer, tmp_path):
        trainer.checkpoint_dir = tmp_path
        trainer.train(n_episodes=2)
        assert (tmp_path / "final_agent.pt").exists()

    def test_best_reward_property(self, trainer):
        trainer.train(n_episodes=3)
        assert trainer.best_reward == max(m["reward"] for m in trainer.history)