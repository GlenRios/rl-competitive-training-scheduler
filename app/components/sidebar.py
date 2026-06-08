"""
sidebar.py -- Configuracion en la barra lateral

Renderiza todos los controles de configuracion y devuelve
un SessionConfig con los valores seleccionados.
"""

from __future__ import annotations

import streamlit as st
from app.state import SessionConfig
from src.environment.problem import CANONICAL_TOPICS


ALGORITHM_OPTIONS = {
    "greedy"  : "Greedy Heuristico",
    "knapsack": "Mochila 0/1",
    "rollout" : "Rollout Horizonte 2",
    "dqn"     : "Agente DQN",
}

PROFILE_PRESETS = {
    "Principiante"  : {t: 900  for t in CANONICAL_TOPICS},
    "Intermedio"    : {t: 1400 for t in CANONICAL_TOPICS},
    "Avanzado"      : {t: 1900 for t in CANONICAL_TOPICS},
    "Personalizado" : None,
}


def render_sidebar() -> SessionConfig:
    """Renderiza la sidebar y devuelve la configuracion activa."""
    with st.sidebar:
        st.markdown("## RL TRAINING SCHEDULER")
        st.markdown("---")

        # ── Algoritmo ─────────────────────────────────────────────────
        st.markdown("### ALGORITMO")
        algorithm = st.selectbox(
            label     = "Selector",
            options   = list(ALGORITHM_OPTIONS.keys()),
            format_func = lambda k: ALGORITHM_OPTIONS[k],
            key       = "sb_algorithm",
        )

        if algorithm == "dqn" and not st.session_state.get("agent"):
            st.warning("DQN no cargado. Entrena primero.")
            uploaded = st.file_uploader("Cargar pesos .pt", type=["pt"])
            if uploaded:
                _load_agent_from_upload(uploaded)

        st.markdown("---")

        # ── Perfil de estudiante ───────────────────────────────────────
        st.markdown("### PERFIL DE ESTUDIANTE")
        profile_type = st.selectbox(
            "Tipo",
            options = list(PROFILE_PRESETS.keys()),
            key     = "sb_profile_type",
        )

        profile_index = 0
        if profile_type != "Personalizado":
            profiles = st.session_state.get("profiles", [])
            if profiles:
                # Buscar el perfil del dataset que mas se acerca al preset
                target = sum(PROFILE_PRESETS[profile_type].values()) / len(CANONICAL_TOPICS)
                profile_index = min(
                    range(len(profiles)),
                    key=lambda i: abs(profiles[i].global_rating - target)
                )
                p = profiles[profile_index]
                st.caption(f"**{p.archetype[:40]}**")
                st.caption(f"Rating global: **{p.global_rating:.0f}**")
        else:
            profile_index = st.slider(
                "Indice de perfil", 0,
                max(0, len(st.session_state.get("profiles", [1])) - 1),
                0, key="sb_profile_idx"
            )
            profiles = st.session_state.get("profiles", [])
            if profiles:
                p = profiles[profile_index]
                st.caption(f"{p.archetype[:40]}")

        st.markdown("---")

        # ── Sesion ────────────────────────────────────────────────────
        st.markdown("### SESION")
        budget = st.slider(
            "Tiempo maximo (min)",
            min_value = 30,
            max_value = 240,
            value     = 120,
            step      = 15,
            key       = "sb_budget",
        )

        st.markdown("---")

        # ── Opciones avanzadas ─────────────────────────────────────────
        st.markdown("### OPCIONES")
        use_llm = st.checkbox(
            "Recompensa pedagogica LLM",
            value = False,
            key   = "sb_use_llm",
            help  = "Requiere Ollama corriendo localmente",
        )
        manual_mode = st.checkbox(
            "Modo manual (humano vs IA)",
            value = False,
            key   = "sb_manual",
        )

        st.markdown("---")
        st.caption("v1.0 -- RL Competitive Training")

    config = SessionConfig(
        algorithm          = algorithm,
        session_budget_min = float(budget),
        profile_index      = profile_index,
        use_llm_reward     = use_llm,
        manual_mode        = manual_mode,
    )
    st.session_state["config"] = config
    return config


def _load_agent_from_upload(uploaded_file) -> None:
    """Carga el agente DQN desde un archivo subido."""
    import tempfile, torch
    from pathlib import Path
    from src.agent.dqn import DQNAgent
    from src.environment.observation_builder import FULL_OBS_DIM

    problems = st.session_state.get("problems", [])
    if not problems:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    try:
        agent = DQNAgent(
            obs_dim=FULL_OBS_DIM, n_actions=len(problems), hidden_sizes=[128, 64]
        )
        agent.load(tmp_path)
        agent.online_net.eval()
        st.session_state["agent"] = agent
        st.success("Agente DQN cargado correctamente.")
    except Exception as e:
        st.error(f"Error cargando agente: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)