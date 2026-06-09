"""
charts.py -- Graficas Plotly reutilizables
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

CYAN  = "#00e5ff"
GREEN = "#00e676"
RED   = "#ff5252"
AMBER = "#ffab40"
NAVY  = "#0d1729"
GRID  = "#1a2e50"
TEXT  = "#7ecfff"


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convierte hex a rgba para compatibilidad con Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout(title: str = "", height: int = 320) -> dict:
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
    sorted_items = sorted(topic_ratings.items(), key=lambda x: x[1], reverse=True)
    topics  = [t for t, _ in sorted_items]
    ratings = [r for _, r in sorted_items]
    colors  = [
        GREEN if r >= 1800 else CYAN if r >= 1400 else AMBER if r >= 1100 else RED
        for r in ratings
    ]
    fig = go.Figure(go.Bar(
        x             = ratings,
        y             = topics,
        orientation   = "h",
        marker_color  = colors,
        marker_line_width = 0,
        text          = [f"{r:.0f}" for r in ratings],
        textposition  = "outside",
        textfont      = dict(size=10, color=TEXT, family="JetBrains Mono"),
    ))
    layout = _base_layout("ELO POR TEMA", height=420)
    layout["xaxis"] = dict(gridcolor=GRID, zerolinecolor=GRID, range=[700, 2200])
    layout["yaxis"] = dict(gridcolor=GRID, tickfont=dict(size=10))
    fig.update_layout(**layout)
    return fig


def rating_evolution(history: list[dict]) -> go.Figure:
    if not history:
        return go.Figure()
    steps   = list(range(1, len(history) + 1))
    ratings = [h["new_global_rating"] for h in history]
    solved  = [h["solved"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x      = steps, y = ratings, mode = "lines+markers",
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
    if not history:
        return go.Figure()
    steps   = list(range(1, len(history) + 1))
    ratings = [h["problem_rating"] for h in history]
    solved  = [h["solved"] for h in history]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x      = steps, y = ratings, mode = "lines+markers",
        line   = dict(color=AMBER, width=1, dash="dot"),
        marker = dict(color=[GREEN if s else RED for s in solved], size=10),
        name   = "Rating del problema",
    ))
    fig.update_layout(**_base_layout("SECUENCIA DE DIFICULTAD", height=260))
    return fig


def comparison_bar(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    agg = df.groupby("selector")[metric].agg(["mean", "std"]).reset_index()
    fig = go.Figure(go.Bar(
        x                = agg["selector"],
        y                = agg["mean"],
        error_y          = dict(type="data", array=agg["std"], visible=True),
        marker_color     = CYAN,
        marker_line_width = 0,
        text             = [f"{v:.2f}" for v in agg["mean"]],
        textposition     = "outside",
    ))
    fig.update_layout(**_base_layout(title, height=300))
    return fig


def comparison_boxplot(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    selectors = df["selector"].unique()
    colors    = [CYAN, GREEN, AMBER, RED]
    fig       = go.Figure()
    for i, sel in enumerate(selectors):
        values = df[df["selector"] == sel][metric]
        color  = colors[i % len(colors)]
        fig.add_trace(go.Box(
            y            = values,
            name         = sel,
            marker_color = color,
            line_color   = color,
            fillcolor    = _hex_to_rgba(color, alpha=0.12),  # rgba en lugar de hex+alpha
        ))
    fig.update_layout(**_base_layout(title, height=340))
    return fig


def training_curve(history: list[dict]) -> go.Figure:
    if not history:
        return go.Figure()
    episodes = [h["episode"] for h in history]
    rewards  = [h["reward"]  for h in history]
    window   = 20
    smoothed = [
        sum(rewards[max(0, i-window+1):i+1]) / len(rewards[max(0, i-window+1):i+1])
        for i in range(len(rewards))
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards, mode="lines",
        line=dict(color=_hex_to_rgba(CYAN, 0.3).replace("rgba", "rgba"), width=1),
        name="Recompensa por episodio",
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=smoothed, mode="lines",
        line=dict(color=CYAN, width=2),
        name=f"Media movil ({window} ep.)",
    ))
    fig.update_layout(**_base_layout("CURVA DE ENTRENAMIENTO DQN", height=300))
    return fig