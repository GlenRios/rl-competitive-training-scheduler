"""
3_Training.py -- Entrenamiento DQN interactivo

Permite entrenar el agente DQN con parametros configurables,
visualizando la curva de aprendizaje en tiempo real.
"""

import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import pandas as pd

from state import init_state, load_data
from styles import inject_styles
from components.sidebar import render_sidebar
from components.charts import training_curve

st.set_page_config(page_title="Entrenamiento DQN", layout="wide")
inject_styles()
init_state()

if not load_data():
    st.error("Datos no cargados.")
    st.stop()

config   = render_sidebar()
problems = st.session_state["problems"]
profiles = st.session_state["profiles"]

st.title("ENTRENAMIENTO DQN INTERACTIVO")
st.markdown("---")

# ── Parametros de entrenamiento ────────────────────────────────────────────
st.markdown("### Configuracion")
c1, c2, c3, c4 = st.columns(4)
with c1:
    n_episodes     = st.number_input("Episodios", 50, 2000, 300, 50, key="tr_episodes")
with c2:
    epsilon_end    = st.slider("Epsilon final", 0.01, 0.3, 0.05, key="tr_eps_end")
with c3:
    lr             = st.select_slider(
        "Learning rate",
        options = [1e-4, 5e-4, 1e-3, 5e-3],
        value   = 1e-3,
        key     = "tr_lr",
    )
with c4:
    hidden_str     = st.selectbox(
        "Arquitectura red",
        ["[128, 64]", "[256, 128]", "[64, 32]"],
        key = "tr_hidden",
    )
    hidden_sizes = list(map(int, hidden_str.strip("[]").split(",")))

col_run, col_stop, col_dl = st.columns([1, 1, 2])

with col_run:
    run_btn  = st.button("Entrenar DQN", type="primary", use_container_width=True)
with col_stop:
    stop_btn = st.button("Detener",      use_container_width=True)

# Detener entrenamiento
if stop_btn:
    st.session_state["train_stop_flag"] = True

# ── Entrenamiento ──────────────────────────────────────────────────────────

if run_btn:
    from src.agent.dqn import DQNAgent
    from src.agent.replay_buffer import ReplayBuffer
    from src.agent.trainer import DQNTrainer
    from src.environment.env import TrainingEnv
    from src.environment.observation_builder import ObservationBuilder, FULL_OBS_DIM
    from src.environment.student_generator import StudentProfileGenerator

    st.session_state["train_stop_flag"] = False
    st.session_state["train_history"]   = []

    generator = StudentProfileGenerator(use_llm=False, random_seed=42)
    env = TrainingEnv(
        problems           = problems,
        session_budget_min = config.session_budget_min,
        random_seed        = 42,
        profiles           = profiles,
    )
    obs_builder = ObservationBuilder(
        problems=problems, session_budget_min=config.session_budget_min
    )
    agent = DQNAgent(
        obs_dim      = FULL_OBS_DIM,
        n_actions    = len(problems),
        hidden_sizes = hidden_sizes,
        lr           = lr,
    )
    buffer = ReplayBuffer(capacity=10_000)

    st.markdown("---")
    st.markdown("### Progreso del entrenamiento")
    progress_bar  = st.progress(0.0)
    status_text   = st.empty()
    chart_placeholder = st.empty()

    history = []

    # Loop de entrenamiento con actualizacion de UI cada 10 episodios
    from src.agent.replay_buffer import Transition
    import numpy as np
    import time

    epsilon_start = 1.0
    epsilon_decay = int(n_episodes * 0.6)
    batch_size    = 64
    log_every     = 10

    for ep in range(1, n_episodes + 1):
        if st.session_state.get("train_stop_flag"):
            st.warning(f"Entrenamiento detenido en episodio {ep}.")
            break

        # Calcular epsilon
        ratio   = min(1.0, (ep - 1) / max(1, epsilon_decay - 1))
        epsilon = epsilon_start + ratio * (epsilon_end - epsilon_start)

        # Episodio
        obs, info = env.reset()
        ep_reward = 0.0
        ep_losses = []
        terminated = False

        while not terminated:
            mask  = info["action_mask"]
            p_mat = info["problem_matrix"]

            action      = agent.select_action(obs, p_mat, mask, epsilon)
            state_input = obs_builder.single_problem_input(env.student, action)

            next_obs, reward, terminated, _, next_info = env.step(action)
            ep_reward += reward

            buffer.push(Transition(
                state_input      = state_input,
                action           = action,
                reward           = float(reward),
                next_student_obs = next_obs,
                next_mask        = next_info["action_mask"],
                terminated       = terminated,
            ))

            if buffer.is_ready(batch_size):
                from src.agent.trainer import DQNTrainer
                trainer_tmp = DQNTrainer(
                    env=env, agent=agent, buffer=buffer,
                    obs_builder=obs_builder, batch_size=batch_size,
                )
                loss = trainer_tmp._update(buffer.sample(batch_size))
                ep_losses.append(loss)

            obs  = next_obs
            info = next_info

        if ep % 50 == 0:
            agent.sync_target()

        avg_loss = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
        history.append({
            "episode"     : ep,
            "reward"      : ep_reward,
            "epsilon"     : round(epsilon, 3),
            "avg_loss"    : round(avg_loss, 5),
            "n_solved"    : env.student.n_solved,
            "n_attempted" : env.student.n_attempted,
        })
        st.session_state["train_history"] = history

        if ep % log_every == 0 or ep == n_episodes:
            progress_bar.progress(ep / n_episodes)
            status_text.text(
                f"Ep {ep}/{n_episodes} | "
                f"eps={epsilon:.3f} | "
                f"reward={ep_reward:.1f} | "
                f"loss={avg_loss:.4f} | "
                f"solved={env.student.n_solved}/{env.student.n_attempted}"
            )
            chart_placeholder.plotly_chart(
                training_curve(history),
                use_container_width=True,
            )

    # Guardar agente entrenado
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    agent.save("experiments/results/best_agent.pt")
    st.session_state["agent"] = agent
    st.success("Entrenamiento completado. Agente guardado en experiments/results/best_agent.pt")

# ── Historial existente ────────────────────────────────────────────────────
history = st.session_state.get("train_history", [])
if history:
    st.markdown("---")
    st.markdown("### Curva de aprendizaje")
    st.plotly_chart(training_curve(history), use_container_width=True)

    # Descarga de pesos
    agent_path = Path("experiments/results/best_agent.pt")
    if agent_path.exists():
        with open(agent_path, "rb") as f:
            st.download_button(
                label     = "Descargar pesos del agente (.pt)",
                data      = f,
                file_name = "best_agent.pt",
                mime      = "application/octet-stream",
            )

    # Descarga del historial
    hist_df = pd.DataFrame(history)
    st.download_button(
        label     = "Descargar historial CSV",
        data      = hist_df.to_csv(index=False).encode("utf-8"),
        file_name = "training_history.csv",
        mime      = "text/csv",
    )

    # Metricas finales
    st.markdown("### Metricas finales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mejor reward",    f"{max(h['reward'] for h in history):.1f}")
    col2.metric("Reward final (media 20)",
                f"{sum(h['reward'] for h in history[-20:]) / 20:.1f}")
    col3.metric("Episodios totales", len(history))
    col4.metric("Epsilon final",   f"{history[-1]['epsilon']:.3f}")
else:
    st.info("Configura los parametros y pulsa 'Entrenar DQN' para comenzar.")