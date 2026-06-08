"""
student_panel.py -- Panel de estado del estudiante en tiempo real

Muestra metricas grandes, barras ELO por tema,
barra de tiempo y medidor de fatiga.
"""

import streamlit as st
import plotly.graph_objects as go
from .charts import topic_elo_bars


def render_student_panel(student, session_budget_min: float) -> None:
    """Renderiza el panel completo del estado del estudiante."""
    if student is None:
        st.info("Configura y arranca una simulacion para ver el estado del estudiante.")
        return

    # ── Metricas grandes ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rating Global",    f"{student.global_rating:.0f}")
    c2.metric("Tiempo Restante",  f"{student.time_remaining_min:.0f} min")
    c3.metric("Fatiga",           f"{student.fatigue:.0%}")
    c4.metric("Resueltos",        f"{student.n_solved} / {student.n_attempted}")

    # ── Progreso de tiempo ─────────────────────────────────────────────
    time_pct = min(1.0, student.time_spent_min / session_budget_min)
    st.caption(f"Tiempo consumido: {student.time_spent_min:.1f} / {session_budget_min:.0f} min")
    st.progress(time_pct)

    # ── Fatiga (barra con color) ───────────────────────────────────────
    fatigue_color = (
        "🟢" if student.fatigue < 0.3 else
        "🟡" if student.fatigue < 0.6 else
        "🔴"
    )
    st.caption(f"Fatiga {fatigue_color}: {student.fatigue:.0%}")
    st.progress(student.fatigue)

    # ── ELO por tema ───────────────────────────────────────────────────
    st.plotly_chart(
        topic_elo_bars(student.topic_ratings),
        use_container_width = True,
    )