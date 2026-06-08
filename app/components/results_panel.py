"""
results_panel.py -- Resumen final de una sesion completada

Muestra metricas finales, grafica de secuencia y nube de tags.
"""

import streamlit as st
from .charts import rating_evolution, difficulty_sequence
from src.eval.metrics import compute_episode_metrics


def render_results(history: list[dict], student, initial_rating: float) -> None:
    """Renderiza el panel de resultados al finalizar una sesion."""
    if not history:
        return

    metrics = compute_episode_metrics(
        history        = history,
        initial_rating = initial_rating,
        final_rating   = student.global_rating,
    )

    st.markdown("## Resumen de la sesion")

    # ── Metricas clave ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recompensa total",  f"{metrics['total_reward']:.1f}")
    c2.metric("Tasa de exito",     f"{metrics['success_rate']:.0%}")
    c3.metric("Temas cubiertos",   metrics['topic_coverage'])
    c4.metric("Progresion",        f"{metrics['spearman_progression']:.2f}")
    c5.metric("Mejora rating",     f"+{metrics['rating_improvement']:.1f}")

    st.markdown("---")

    # ── Graficas ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(rating_evolution(history), use_container_width=True)
    with col2:
        st.plotly_chart(difficulty_sequence(history), use_container_width=True)

    # ── Word cloud de tags ─────────────────────────────────────────────
    _render_tag_cloud(history)

    # ── Tabla detallada ────────────────────────────────────────────────
    with st.expander("Ver tabla detallada de pasos"):
        import pandas as pd
        rows = [{
            "Paso"    : h["step"],
            "Problema": h["problem_id"],
            "Rating"  : h["problem_rating"],
            "Resuelto": "Si" if h["solved"] else "No",
            "Reward"  : round(h["reward"], 2),
            "Tiempo"  : round(h["time_min"], 1),
            "Rating g.": round(h["new_global_rating"], 1),
        } for h in history]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_tag_cloud(history: list[dict]) -> None:
    """Genera y muestra una nube de tags de los temas cubiertos."""
    from collections import Counter
    import matplotlib.pyplot as plt
    import matplotlib

    try:
        from wordcloud import WordCloud
    except ImportError:
        st.caption("(wordcloud no instalado -- pip install wordcloud)")
        return

    tag_counts: Counter = Counter()
    for step in history:
        for tag in step.get("problem_tags", []):
            tag_counts[tag.replace(" ", "_")] += 1

    if not tag_counts:
        return

    matplotlib.use("Agg")
    wc = WordCloud(
        width            = 600,
        height           = 220,
        background_color = "#0d1729",
        color_func       = lambda *a, **kw: "#00e5ff",
        prefer_horizontal = 0.9,
        max_font_size    = 60,
        min_font_size    = 12,
    ).generate_from_frequencies(tag_counts)

    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor="#0d1729")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.markdown("**Temas cubiertos en la sesion:**")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)