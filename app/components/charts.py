"""
charts.py -- Graficas Plotly reutilizables
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

from src.environment.problem import CANONICAL_TOPICS

CYAN  = "#00e5ff"
GREEN = "#00e676"
RED   = "#ff5252"
AMBER = "#ffab40"
NAVY  = "#0d1729"
GRID  = "#1a2e50"
TEXT  = "#7ecfff"


def _base_layout(title: str = "", height: int = 320) -> dict:
    """Layout base SIN yaxis -- cada grafica lo configura por separado."""
    return dict(
        title         = dict(text=title, font=dict(color=CYAN, size=13, family="JetBrains Mono")),
        paper_bgcolor = NAVY,
        plot_bgcolor  = NAVY,
        font          = dict(color=TEXT, family="Sora"),
        height        = height,
        margin        = dict(l=40, r=20, t=40, b=40),
        xaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis         = dict(gridcolor=GRID, zerolinecolor=GRID),
    )


def topic_elo_bars(topic_ratings: dict[str, float]) -> go.Figure:
    """Barras horizontales de ELO por tema."""
    sorted_items = sorted(topic_ratings.items(), key=lambda x: x[1], reverse=True)
    topics  = [t for t, _ in sorted_items]
    ratings = [r for _, r in sorted_items]

    colors = [
        GREEN if r >= 1800 else
        CYAN  if r >= 1400 else
        AMBER if r >= 1100 else
        RED
        for r in ratings
    ]

    fig = go.Figure(go.Bar(
        x            = ratings,
        y            = topics,
        orientation  = "h",
        marker_color = colors,
        marker_line_width = 0,
        text         = [f"{r:.0f}" for r in ratings],
        textposition = "outside",
        textfont     = dict(size=10, color=TEXT, family="JetBrains Mono"),
    ))

    layout = _base_layout("ELO POR TEMA", height=420)
    # Sobreescribir xaxis e yaxis individualmente para evitar duplicados
    layout["xaxis"] = dict(gridcolor=GRID, zerolinecolor=GRID, range=[700, 2200])
    layout["yaxis"] = dict(gridcolor=GRID, tickfont=dict(size=10))
    fig.update_layout(**layout)
    return fig


def rating_evolution(history: list[dict]) -> go.Figure:
    """Linea de evolucion del rating global problema a problema."""
    if not history:
        return go.Figure()

    steps   = list(range(1, len(history) + 1))
    ratings = [h["new_global_rating"] for h in history]
    solved  = [h["solved"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x      = steps,
        y      = ratings,
        mode   = "lines+markers",
        line   = dict(color=CYAN, width=2),
        marker = dict(
            color  = [GREEN if s else RED for s in solved],
            size   = 8,
            symbol = ["circle" if s else "x" for s in solved],
        ),
        name = "Rating global",
    ))
    fig.update_layout(**_base_layout("EVOLUCION DEL RATING GLOBAL", height=280))
    return fig


def difficulty_sequence(history: list[dict]) -> go.Figure:
    """Scatter de rating del problema por orden de intento."""
    if not history:
        return go.Figure()

    steps   = list(range(1, len(history) + 1))
    ratings = [h["problem_rating"] for h in history]
    solved  = [h["solved"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x      = steps,
        y      = ratings,
        mode   = "lines+markers",
        line   = dict(color=AMBER, width=1, dash="dot"),
        marker = dict(color=[GREEN if s else RED for s in solved], size=10),
        name   = "Rating del problema",
    ))
    fig.update_layout(**_base_layout("SECUENCIA DE DIFICULTAD", height=260))
    return fig


def comparison_bar(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Barras agrupadas para comparar selectores."""
    agg = df.groupby("selector")[metric].agg(["mean", "std"]).reset_index()

    fig = go.Figure(go.Bar(
        x            = agg["selector"],
        y            = agg["mean"],
        error_y      = dict(type="data", array=agg["std"], visible=True),
        marker_color = CYAN,
        marker_line_width = 0,
        text         = [f"{v:.2f}" for v in agg["mean"]],
        textposition = "outside",
    ))
    fig.update_layout(**_base_layout(title, height=300))
    return fig


def comparison_boxplot(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    """Boxplot de distribucion por selector."""
    selectors = df["selector"].unique()
    colors    = [CYAN, GREEN, AMBER, RED]

    fig = go.Figure()
    for i, sel in enumerate(selectors):
        values = df[df["selector"] == sel][metric]
        fig.add_trace(go.Box(
            y         = values,
            name      = sel,
            marker_color = colors[i % len(colors)],
            line_color   = colors[i % len(colors)],
            fillcolor    = f"{colors[i % len(colors)]}20",
        ))
    fig.update_layout(**_base_layout(title, height=340))
    return fig


def training_curve(history: list[dict]) -> go.Figure:
    """Curva de recompensa durante el entrenamiento del DQN."""
    if not history:
        return go.Figure()

    episodes = [h["episode"] for h in history]
    rewards  = [h["reward"]  for h in history]

    window   = 20
    smoothed = []
    for i in range(len(rewards)):
        w = rewards[max(0, i - window + 1): i + 1]
        smoothed.append(sum(w) / len(w))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x    = episodes, y = rewards,
        mode = "lines",
        line = dict(color=f"{CYAN}40", width=1),
        name = "Recompensa por episodio",
    ))
    fig.add_trace(go.Scatter(
        x    = episodes, y = smoothed,
        mode = "lines",
        line = dict(color=CYAN, width=2),
        name = f"Media movil ({window} ep.)",
    ))
    fig.update_layout(**_base_layout("CURVA DE ENTRENAMIENTO DQN", height=300))
    return fig