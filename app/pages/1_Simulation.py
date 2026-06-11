"""
1_Simulation.py -- Simulacion paso a paso

Permite ejecutar una sesion completa problema a problema,
observando como el algoritmo selecciona y el estudiante evoluciona.
"""

import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import numpy as np

from state import init_state, load_data, reset_simulation
from styles import inject_styles
from components.sidebar import render_sidebar
from components.student_panel import render_student_panel
from components.problems_table import render_problems_table
from components.results_panel import render_results

st.set_page_config(page_title="Simulacion", layout="wide")
inject_styles()
init_state()

if not load_data():
    st.error("Datos no cargados. Ejecuta primero python -m main.")
    st.stop()

config   = render_sidebar()
problems = st.session_state["problems"]
profiles = st.session_state["profiles"]

st.title("SIMULACION PASO A PASO")
st.markdown("---")

# ── Controles principales ──────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3, _ = st.columns([1, 1, 1, 3])

with col_btn1:
    if st.button("Iniciar / Reiniciar", use_container_width=True):
        reset_simulation()
        st.session_state["sim_initial_rating"] = (
            profiles[config.profile_index % len(profiles)].global_rating
        )
        st.rerun()

student = st.session_state.get("sim_student")

with col_btn2:
    next_disabled = (
        student is None
        or st.session_state.get("sim_finished", False)
    )
    run_next = st.button(
        "Siguiente paso",
        disabled         = next_disabled,
        use_container_width = True,
    )

with col_btn3:
    auto_run = st.button(
        "Correr completo",
        disabled         = next_disabled,
        use_container_width = True,
    )

# ── Ejecutar paso ──────────────────────────────────────────────────────────

def get_selector(config):
    """Devuelve el selector activo segun la configuracion."""
    algorithm = config.algorithm
    if algorithm == "greedy":
        from src.baselines.greedy import GreedySelector
        return GreedySelector()
    elif algorithm == "knapsack":
        from src.baselines.knapsack import KnapsackSelector
        return KnapsackSelector()
    elif algorithm == "rollout":
        from src.baselines.rollout import RolloutSelector
        return RolloutSelector()
    elif algorithm == "dqn":
        agent       = st.session_state.get("agent")
        obs_builder = st.session_state.get("obs_builder")
        if agent is None:
            st.error("DQN no cargado.")
            return None
        from src.baselines.dqn_selector import DQNSelectorAdapter
        return DQNSelectorAdapter(agent, obs_builder)
    return None


def build_mask(student, attempted: set) -> list[bool]:
    mask = []
    for i, p in enumerate(problems):
        if i in attempted:
            mask.append(False)
        else:
            t = student.estimate_solve_time(p.rating, p.tags_list)
            mask.append(t <= student.time_remaining_min)
    return mask


def execute_step(manual_action: int | None = None) -> None:
    """Ejecuta un paso de la simulacion."""
    student   = st.session_state["sim_student"]
    attempted = st.session_state["sim_attempted"]
    history   = st.session_state["sim_history"]
    log       = st.session_state["sim_event_log"]

    mask = build_mask(student, attempted)
    if not any(mask):
        st.session_state["sim_finished"] = True
        return

    # Seleccionar accion
    if manual_action is not None:
        action = manual_action
    else:
        selector = get_selector(config)
        if selector is None:
            return
        from src.baselines.selector import SessionState
        state = SessionState(
            student            = student,
            problems           = problems,
            available_mask     = np.array(mask, dtype=bool),
            session_budget_min = config.session_budget_min,
        )
        action = selector.select_action(state, mask)

    if action >= len(problems):
        st.session_state["sim_finished"] = True
        return

    problem = problems[action]
    outcome = student.attempt(
        problem_rating = problem.rating,
        problem_tags   = problem.tags_list,
        problem_id     = problem.problem_id,
    )
    attempted.add(action)

    step = {
        "step"             : len(history) + 1,
        "problem_id"       : problem.problem_id,
        "problem_rating"   : problem.rating,
        "problem_tags"     : problem.tags_list,
        "solved"           : outcome.solved,
        "p_solve"          : outcome.p_solve,
        "time_min"         : outcome.time_min,
        "reward"           : outcome.reward,
        "topic_deltas"     : outcome.topic_deltas,
        "new_global_rating": outcome.new_global_rating,
        "fatigue"          : outcome.fatigue,
        "time_remaining"   : outcome.time_remaining_min,
    }
    history.append(step)

    status = "RESUELTO" if outcome.solved else "FALLIDO"
    deltas = ", ".join(f"{t}+{v:.1f}" for t, v in outcome.topic_deltas.items())
    log_entry = (
        f"Paso {step['step']}: [{problem.problem_id}] "
        f"rating={problem.rating} | {status} | "
        f"reward={outcome.reward:+.1f} | {deltas} | "
        f"fatiga={outcome.fatigue:.0%} | "
        f"{outcome.time_remaining_min:.0f}min restantes"
    )
    log.append((outcome.solved, log_entry))

    if outcome.session_over or not any(build_mask(student, attempted)):
        st.session_state["sim_finished"] = True


# Botones de accion
manual_action = None
if student and config.manual_mode:
    mask = build_mask(student, st.session_state["sim_attempted"])
    manual_action = render_problems_table(problems, mask, student, manual_mode=True)

if run_next and student:
    execute_step(manual_action)
    st.rerun()

if auto_run and student:
    progress_bar = st.progress(0.0)
    max_steps    = len(problems)
    for i in range(max_steps):
        execute_step()
        progress_bar.progress((i + 1) / max_steps)
        if st.session_state.get("sim_finished"):
            break
    st.rerun()

# ── Panel principal ────────────────────────────────────────────────────────
st.markdown("---")
left, right = st.columns([1, 1])

with left:
    st.markdown("### Estado del Estudiante")
    render_student_panel(
        st.session_state.get("sim_student"),
        config.session_budget_min,
    )

with right:
    st.markdown("### Log de Eventos")
    log = st.session_state.get("sim_event_log", [])
    if log:
        lines = []
        for solved, entry in reversed(log[-20:]):
            cls = "event-solved" if solved else "event-failed"
            lines.append(f'<div class="{cls}">{entry}</div>')
        st.markdown(
            f'<div class="event-log">{"".join(lines)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Pulsa 'Iniciar' y luego 'Siguiente paso' para comenzar.")

    # Tabla de problemas (modo automatico)
    if student and not config.manual_mode:
        st.markdown("### Problemas Disponibles")
        mask = build_mask(student, st.session_state["sim_attempted"])
        render_problems_table(problems, mask, student, manual_mode=False)

# ── Resultados finales ─────────────────────────────────────────────────────
if st.session_state.get("sim_finished") and st.session_state.get("sim_history"):
    st.markdown("---")
    initial_rating = st.session_state.get("sim_initial_rating", 1400.0)
    render_results(
        history        = st.session_state["sim_history"],
        student        = st.session_state["sim_student"],
        initial_rating = initial_rating,
    )