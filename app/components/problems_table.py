"""
problems_table.py -- Tabla de problemas disponibles

Muestra los problemas con columnas configuradas y
botones de seleccion manual cuando el modo lo requiere.
"""

import pandas as pd
import streamlit as st
from src.environment.problem import Problem
from src.environment.student_model import StudentModel


def render_problems_table(
    problems       : list[Problem],
    available_mask : list[bool],
    student        : StudentModel,
    manual_mode    : bool = False,
) -> int | None:
    """Renderiza la tabla de problemas disponibles.

    Returns
    -------
    int | None -- indice del problema seleccionado manualmente, o None.
    """
    rows = []
    for i, (problem, available) in enumerate(zip(problems, available_mask)):
        if not available:
            continue
        t_est   = student.estimate_solve_time(problem.rating, problem.tags_list)
        p_solve = student.probability_of_solving(problem.rating, problem.tags_list)
        rows.append({
            "idx"        : i,
            "ID"         : problem.problem_id,
            "Rating"     : problem.rating,
            "Tags"       : ", ".join(problem.tags_list[:3]),
            "T. est (min)": round(t_est, 1),
            "P(resolver)": round(p_solve, 3),
            "Banda"      : problem.difficulty_band,
        })

    if not rows:
        st.warning("No hay problemas disponibles (sin tiempo o todos intentados).")
        return None

    df = pd.DataFrame(rows)

    # Tabla con columnas configuradas
    st.dataframe(
        df.drop(columns=["idx"]),
        use_container_width = True,
        column_config = {
            "Rating"      : st.column_config.NumberColumn("Rating", format="%d"),
            "T. est (min)": st.column_config.NumberColumn("Tiempo est.", format="%.1f"),
            "P(resolver)" : st.column_config.ProgressColumn(
                "P(resolver)", min_value=0, max_value=1, format="%.3f"
            ),
        },
        hide_index = True,
    )

    # Modo manual: selector de problema
    if manual_mode:
        st.markdown("**Elige el siguiente problema:**")
        options = {f"{r['ID']} (rating {r['Rating']})": r["idx"] for r in rows}
        chosen_label = st.selectbox("Problema", options=list(options.keys()), key="manual_sel")
        if st.button("Confirmar seleccion", key="manual_confirm"):
            return options[chosen_label]

    return None