"""
2_Comparison.py -- Comparativa de los 4 algoritmos

Ejecuta N episodios por selector y muestra estadisticas comparativas.
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
from components.charts import comparison_bar, comparison_boxplot

st.set_page_config(page_title="Comparativa", layout="wide")
inject_styles()
init_state()

if not load_data():
    st.error("Datos no cargados.")
    st.stop()

config   = render_sidebar()
problems = st.session_state["problems"]
profiles = st.session_state["profiles"]

st.title("COMPARATIVA DE ALGORITMOS")
st.markdown("---")

# ── Configuracion de la comparativa ───────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    n_episodes = st.slider("Episodios por algoritmo", 5, 50, 10, key="cmp_episodes")
with col2:
    algorithms = st.multiselect(
        "Algoritmos a comparar",
        options  = ["greedy", "knapsack", "rollout", "dqn"],
        default  = ["greedy", "knapsack", "rollout"],
        key      = "cmp_algorithms",
    )
with col3:
    run_comparison = st.button("Comparar", use_container_width=True, type="primary")

# ── Ejecucion ──────────────────────────────────────────────────────────────

def build_selectors(algorithms: list[str]) -> dict:
    selectors = {}
    for alg in algorithms:
        if alg == "greedy":
            from src.baselines.greedy import GreedySelector
            selectors["greedy"] = GreedySelector()
        elif alg == "knapsack":
            from src.baselines.knapsack import KnapsackSelector
            selectors["knapsack"] = KnapsackSelector()
        elif alg == "rollout":
            from src.baselines.rollout import RolloutSelector
            selectors["rollout"] = RolloutSelector()
        elif alg == "dqn":
            agent       = st.session_state.get("agent")
            obs_builder = st.session_state.get("obs_builder")
            if agent:
                from evaluate import DQNSelectorAdapter
                selectors["dqn"] = DQNSelectorAdapter(agent, obs_builder)
            else:
                st.warning("DQN no disponible -- omitido.")
    return selectors


if run_comparison and algorithms:
    selectors = build_selectors(algorithms)
    if not selectors:
        st.error("No hay algoritmos disponibles.")
        st.stop()

    from src.eval.evaluator import Evaluator
    evaluator = Evaluator(problems=problems, n_episodes=n_episodes, verbose=False)

    progress = st.progress(0.0)
    status   = st.empty()
    all_rows = []

    for i, (name, selector) in enumerate(selectors.items()):
        status.text(f"Evaluando [{name}]... ({i+1}/{len(selectors)})")
        rows = evaluator.run_single_selector(name, selector, profiles[:n_episodes])
        all_rows.append(rows)
        progress.progress((i + 1) / len(selectors))

    results = pd.concat(all_rows, ignore_index=True)
    st.session_state["comparison_results"] = results
    status.empty()
    progress.empty()
    st.success(f"Comparativa completada: {len(results)} episodios totales.")

# ── Resultados ─────────────────────────────────────────────────────────────
results = st.session_state.get("comparison_results")

if results is not None and not results.empty:
    st.markdown("---")

    # Tabla de resumen
    st.markdown("### Resumen estadistico")
    summary = (
        results
        .groupby("selector")[
            ["total_reward", "success_rate", "topic_coverage",
             "spearman_progression", "rating_improvement"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    st.dataframe(summary, use_container_width=True)

    # Boton de descarga
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar resultados CSV",
        data      = csv,
        file_name = "comparison_results.csv",
        mime      = "text/csv",
    )

    st.markdown("---")

    # Graficas comparativas
    tab1, tab2, tab3 = st.tabs(["Recompensa", "Tasa de exito", "Temas cubiertos"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                comparison_bar(results, "total_reward", "RECOMPENSA TOTAL"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                comparison_boxplot(results, "total_reward", "DISTRIBUCION RECOMPENSA"),
                use_container_width=True,
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                comparison_bar(results, "success_rate", "TASA DE EXITO"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                comparison_boxplot(results, "success_rate", "DISTRIBUCION TASA EXITO"),
                use_container_width=True,
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                comparison_bar(results, "topic_coverage", "COBERTURA DE TEMAS"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                comparison_bar(results, "rating_improvement", "MEJORA DE RATING"),
                use_container_width=True,
            )

elif not run_comparison:
    st.info("Selecciona los algoritmos y pulsa 'Comparar' para iniciar la evaluacion.")