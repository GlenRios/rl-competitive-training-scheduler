"""
tests/test_environment.py

Tests para los tres módulos del entorno:
    - ObservationBuilder
    - ActionMasker
    - TrainingEnv

Ejecutar con:
    pytest tests/test_environment.py -v
"""

import numpy as np
import pytest

from src.environment.problem import Problem
from src.environment.student_model import StudentModel
from src.environment.observation_builder import (
    ObservationBuilder,
    STUDENT_OBS_DIM,
    PROBLEM_OBS_DIM,
    FULL_OBS_DIM,
)
from src.environment.action_masker import ActionMasker
from src.environment.env import TrainingEnv


# ---------------------------------------------------------------------------
# Fixtures compartidos
# ---------------------------------------------------------------------------

def make_problems(n: int = 10) -> list[Problem]:
    ratings = [800, 900, 1000, 1100, 1200, 1300, 1400, 1600, 1800, 2000]
    tags    = [["math"], ["greedy"], ["dp"], ["graphs"], ["math", "dp"],
               ["greedy", "graphs"], ["strings"], ["trees"], ["dp", "greedy"], ["math"]]
    return [
        Problem(
            problem_id      = f"{i+1}A",
            name            = f"Problem {i+1}",
            rating          = ratings[i % len(ratings)],
            tags_list       = tags[i % len(tags)],
            statement       = f"Statement {i+1} " * 10,
            difficulty_band = "800-1199",
        )
        for i in range(n)
    ]


@pytest.fixture
def problems() -> list[Problem]:
    return make_problems(10)


@pytest.fixture
def student() -> StudentModel:
    from src.environment.problem import CANONICAL_TOPICS
    return StudentModel(
        topic_ratings={t: 1500 for t in CANONICAL_TOPICS},
        session_budget_min=120, random_seed=42,
    )


@pytest.fixture
def builder(problems) -> ObservationBuilder:
    return ObservationBuilder(problems, session_budget_min=120.0)


@pytest.fixture
def masker(problems) -> ActionMasker:
    return ActionMasker(problems)


@pytest.fixture
def env(problems) -> TrainingEnv:
    return TrainingEnv(problems, session_budget_min=120, random_seed=42)


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class TestObservationBuilder:
    def test_student_obs_shape(self, builder, student):
        obs = builder.student_obs(student)
        assert obs.shape == (STUDENT_OBS_DIM,)

    def test_student_obs_dtype(self, builder, student):
        obs = builder.student_obs(student)
        assert obs.dtype == np.float32

    def test_student_obs_values_in_range(self, builder, student):
        obs = builder.student_obs(student)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_problem_matrix_shape(self, builder, student):
        mat = builder.problem_matrix(student)
        assert mat.shape == (10, PROBLEM_OBS_DIM)

    def test_problem_matrix_dtype(self, builder, student):
        mat = builder.problem_matrix(student)
        assert mat.dtype == np.float32

    def test_full_input_matrix_shape(self, builder, student):
        mat = builder.full_input_matrix(student)
        assert mat.shape == (10, FULL_OBS_DIM)

    def test_full_input_matrix_is_concat_of_student_and_problem(self, builder, student):
        full = builder.full_input_matrix(student)
        s_obs = builder.student_obs(student)
        # Las primeras STUDENT_OBS_DIM columnas deben ser iguales en cada fila
        assert np.allclose(full[:, :STUDENT_OBS_DIM], s_obs)

    def test_single_problem_input_shape(self, builder, student):
        vec = builder.single_problem_input(student, problem_idx=0)
        assert vec.shape == (FULL_OBS_DIM,)

    def test_single_problem_input_raises_on_bad_idx(self, builder, student):
        with pytest.raises(IndexError):
            builder.single_problem_input(student, problem_idx=999)

    def test_dims_are_consistent(self):
        assert FULL_OBS_DIM == STUDENT_OBS_DIM + PROBLEM_OBS_DIM

    def test_raises_on_empty_problems(self):
        with pytest.raises(ValueError, match="vacía"):
            ObservationBuilder([], session_budget_min=120.0)

    def test_raises_on_zero_budget(self, problems):
        with pytest.raises(ValueError, match="positivo"):
            ObservationBuilder(problems, session_budget_min=0)


# ---------------------------------------------------------------------------
# ActionMasker
# ---------------------------------------------------------------------------

class TestActionMasker:
    def test_all_valid_at_start(self, masker, student):
        mask = masker.get_mask(student)
        # Estudiante de 1500 vs problemas de 800-1200: todos caben en 120 min
        assert mask.any()

    def test_mask_shape(self, masker, student):
        mask = masker.get_mask(student)
        assert mask.shape == (10,)

    def test_mask_dtype_is_bool(self, masker, student):
        mask = masker.get_mask(student)
        assert mask.dtype == bool

    def test_attempted_problem_becomes_invalid(self, masker, student):
        masker.mark_attempted(0)
        mask = masker.get_mask(student)
        assert not mask[0]

    def test_n_attempted_increases_after_mark(self, masker):
        assert masker.n_attempted == 0
        masker.mark_attempted(3)
        assert masker.n_attempted == 1

    def test_n_available_decreases_after_mark(self, masker):
        before = masker.n_available
        masker.mark_attempted(0)
        assert masker.n_available == before - 1

    def test_reset_clears_attempted(self, masker, student):
        masker.mark_attempted(0)
        masker.mark_attempted(1)
        masker.reset()
        assert masker.n_attempted == 0
        mask = masker.get_mask(student)
        assert mask[0] and mask[1]

    def test_mark_invalid_index_raises(self, masker):
        with pytest.raises(IndexError):
            masker.mark_attempted(999)

    def test_problem_invalid_when_no_time(self, problems):
        # Estudiante con casi sin tiempo → problemas difíciles no caben
        from src.environment.problem import CANONICAL_TOPICS
        s = StudentModel(topic_ratings={t: 1500 for t in CANONICAL_TOPICS},
                         session_budget_min=120, random_seed=0)
        s.time_spent_min = 119.0   # solo 1 minuto restante
        masker = ActionMasker(problems)
        mask = masker.get_mask(s)
        # Con 1 min restante, ningún problema debería caber
        assert not mask.any()

    def test_valid_indices_returns_list(self, masker, student):
        indices = masker.valid_indices(student)
        assert isinstance(indices, list)

    def test_any_valid_true_at_start(self, masker, student):
        assert masker.any_valid(student)


# ---------------------------------------------------------------------------
# TrainingEnv
# ---------------------------------------------------------------------------

class TestTrainingEnvInit:
    def test_action_space_is_discrete(self, env):
        import gymnasium as gym
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def test_action_space_size(self, env):
        assert env.action_space.n == 10

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (STUDENT_OBS_DIM,)

    def test_raises_on_empty_problems(self):
        with pytest.raises(ValueError, match="vacía"):
            TrainingEnv([], session_budget_min=120)


class TestTrainingEnvReset:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs.shape == (STUDENT_OBS_DIM,)

    def test_reset_obs_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_reset_info_has_action_mask(self, env):
        _, info = env.reset()
        assert "action_mask" in info

    def test_reset_info_has_problem_matrix(self, env):
        _, info = env.reset()
        assert "problem_matrix" in info
        assert info["problem_matrix"].shape == (10, PROBLEM_OBS_DIM)

    def test_reset_action_mask_shape(self, env):
        _, info = env.reset()
        assert info["action_mask"].shape == (10,)

    def test_reset_clears_episode_reward(self, env):
        env.reset()
        # Hacer un step y luego resetear
        _, info = env.reset()
        valid = np.where(info["action_mask"])[0]
        env.step(int(valid[0]))
        env.reset()
        assert env.episode_reward == 0.0

    def test_reset_step_count_zero(self, env):
        env.reset()
        assert env._step_count == 0


class TestTrainingEnvStep:
    def test_step_returns_5_tuple(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        result  = env.step(int(valid[0]))
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        obs, *_ = env.step(int(valid[0]))
        assert obs.shape == (STUDENT_OBS_DIM,)

    def test_step_reward_is_float(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        _, reward, *_ = env.step(int(valid[0]))
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        _, _, terminated, truncated, _ = env.step(int(valid[0]))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_truncated_always_false(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        _, _, _, truncated, _ = env.step(int(valid[0]))
        assert not truncated

    def test_step_info_has_action_mask(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        _, _, _, _, info = env.step(int(valid[0]))
        assert "action_mask" in info

    def test_step_marks_problem_as_attempted(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        action  = int(valid[0])
        _, _, _, _, info = env.step(action)
        assert not info["action_mask"][action]

    def test_step_raises_on_invalid_action(self, env):
        _, info = env.reset()
        valid   = np.where(info["action_mask"])[0]
        action  = int(valid[0])
        env.step(action)   # primer intento válido
        with pytest.raises(ValueError, match="no válida"):
            env.step(action)   # segundo intento → ya intentado → inválido

    def test_episode_reward_accumulates(self, env):
        _, info = env.reset()
        total   = 0.0
        for _ in range(3):
            valid = np.where(info["action_mask"])[0]
            if not len(valid):
                break
            _, reward, terminated, _, info = env.step(int(valid[0]))
            total += reward
            if terminated:
                break
        assert env.episode_reward == pytest.approx(total, abs=0.01)

    def test_episode_history_grows_per_step(self, env):
        _, info = env.reset()
        for i in range(3):
            valid = np.where(info["action_mask"])[0]
            if not len(valid):
                break
            _, _, terminated, _, info = env.step(int(valid[0]))
            assert len(env.episode_history) == i + 1
            if terminated:
                break

    def test_terminated_when_all_problems_attempted(self):
        # Con solo 2 problemas fáciles y tiempo amplio → termina tras intentar todos
        p = make_problems(2)
        e = TrainingEnv(p, session_budget_min=300, random_seed=0)
        _, info = e.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 10:
            valid = np.where(info["action_mask"])[0]
            if not len(valid):
                break
            _, _, terminated, _, info = e.step(int(valid[0]))
            steps += 1
        assert steps <= 2


class TestTrainingEnvRender:
    def test_render_does_not_raise(self, env, capsys):
        env.reset()
        env.render()
        out = capsys.readouterr().out
        assert "Rating" in out
        assert "Fatiga" in out