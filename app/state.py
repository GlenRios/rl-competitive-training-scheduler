"""
state.py -- Gestion centralizada de st.session_state
"""

import sys
from pathlib import Path

# Anadir la raiz del proyecto a sys.path (necesario con Streamlit)
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dataclasses import dataclass
import streamlit as st

PROBLEMS_CSV  = "data/processed/problems_500.csv"
PROFILES_JSON = "data/processed/student_profiles.json"
AGENT_PT      = "experiments/results/best_agent.pt"
HISTORY_CSV   = "experiments/results/training_history.csv"


@dataclass
class SessionConfig:
    algorithm          : str   = "greedy"
    session_budget_min : float = 120.0
    profile_index      : int   = 0
    use_llm_reward     : bool  = False
    manual_mode        : bool  = False


def init_state() -> None:
    defaults = {
        "problems"           : None,
        "profiles"           : None,
        "agent"              : None,
        "obs_builder"        : None,
        "data_loaded"        : False,
        "config"             : SessionConfig(),
        "sim_student"        : None,
        "sim_history"        : [],
        "sim_attempted"      : set(),
        "sim_finished"       : False,
        "sim_event_log"      : [],
        "sim_initial_rating" : 1400.0,
        "train_history"      : [],
        "train_running"      : False,
        "train_stop_flag"    : False,
        "comparison_results" : None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_data() -> bool:
    if st.session_state.get("data_loaded"):
        return True

    root = Path(_project_root)
    for rel_path in [PROBLEMS_CSV, PROFILES_JSON]:
        if not (root / rel_path).exists():
            return False

    try:
        import pandas as pd
        from src.environment.problem import Problem
        from src.environment.student_generator import StudentProfileGenerator
        from src.environment.observation_builder import ObservationBuilder, FULL_OBS_DIM

        df       = pd.read_csv(root / PROBLEMS_CSV)
        problems = Problem.from_dataframe(df)
        profiles = StudentProfileGenerator.load_profiles(str(root / PROFILES_JSON))

        st.session_state["problems"]    = problems
        st.session_state["profiles"]    = profiles
        st.session_state["obs_builder"] = ObservationBuilder(
            problems=problems, session_budget_min=120.0
        )

        agent_path = root / AGENT_PT
        if agent_path.exists():
            from src.agent.dqn import DQNAgent
            agent = DQNAgent(
                obs_dim=FULL_OBS_DIM, n_actions=len(problems), hidden_sizes=[128, 64]
            )
            agent.load(str(agent_path))
            agent.online_net.eval()
            st.session_state["agent"] = agent

        st.session_state["data_loaded"] = True
        return True

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return False


def reset_simulation() -> None:
    """Reinicia la simulacion usando el budget del CONFIG, no del perfil LLM."""
    config   = st.session_state.get("config", SessionConfig())
    profiles = st.session_state.get("profiles", [])
    if not profiles:
        return

    profile = profiles[config.profile_index % len(profiles)]

    # IMPORTANTE: usar config.session_budget_min (slider del usuario)
    # NO profile.session_budget_min (que el LLM puede fijar en otro valor)
    from src.environment.student_model import StudentModel
    student = StudentModel(
        topic_ratings      = profile.topic_ratings,
        global_rating      = profile.global_rating,
        session_budget_min = config.session_budget_min,
        random_seed        = 42,
    )

    st.session_state["sim_student"]        = student
    st.session_state["sim_history"]        = []
    st.session_state["sim_attempted"]      = set()
    st.session_state["sim_finished"]       = False
    st.session_state["sim_event_log"]      = []
    st.session_state["sim_initial_rating"] = profile.global_rating