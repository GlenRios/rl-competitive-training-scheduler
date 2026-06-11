"""
styles.py -- Tema visual de la aplicacion

Tema oscuro tecnico inspirado en IDEs de programacion competitiva.
Paleta: navy profundo, cyan electrico, verde neon, rojo para fallos.
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #080d1a;
    color: #c9d1e0;
}

/* Encabezados */
h1, h2, h3 { font-family: 'JetBrains Mono', monospace; }
h1 { color: #00e5ff; letter-spacing: -1px; }
h2 { color: #7ecfff; }
h3 { color: #a8d8ff; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1022;
    border-right: 1px solid #1a2540;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #00e5ff;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Cards */
.card {
    background: #0d1729;
    border: 1px solid #1a2e50;
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid #00e5ff;
}

/* Metricas */
[data-testid="stMetric"] {
    background: #0d1729;
    border: 1px solid #1a2e50;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] {
    color: #5a7a9a !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    color: #00e5ff !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem !important;
    font-weight: 700;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* Botones */
.stButton > button {
    background: transparent;
    border: 1px solid #00e5ff;
    color: #00e5ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    border-radius: 4px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00e5ff15;
    box-shadow: 0 0 12px #00e5ff40;
}
.stButton > button[kind="primary"] {
    background: #00e5ff20;
    border-color: #00e5ff;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #00e5ff, #0077ff);
    border-radius: 2px;
}

/* Dataframe */
.stDataFrame { border: 1px solid #1a2e50 !important; }

/* Tags de exito/fallo */
.tag-solved {
    background: #00e67620;
    color: #00e676;
    border: 1px solid #00e67640;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}
.tag-failed {
    background: #ff525220;
    color: #ff5252;
    border: 1px solid #ff525240;
    border-radius: 12px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Log de eventos */
.event-log {
    background: #060b16;
    border: 1px solid #1a2540;
    border-radius: 6px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    max-height: 280px;
    overflow-y: auto;
    color: #7ecfff;
}
.event-solved { color: #00e676; }
.event-failed { color: #ff5252; }

/* Separador */
hr { border-color: #1a2540; }

/* Selectbox, slider */
.stSelectbox > div > div,
.stRadio > div {
    background: #0d1729;
    border-color: #1a2e50;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1022;
    border-bottom: 1px solid #1a2540;
}
.stTabs [data-baseweb="tab"] {
    color: #5a7a9a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
}
</style>
"""


def inject_styles() -> None:
    """Inyecta el CSS en la pagina actual."""
    st.markdown(CSS, unsafe_allow_html=True)


def card(content_fn, accent: bool = False) -> None:
    """Renderiza contenido dentro de un card con estilo."""
    cls = "card card-accent" if accent else "card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)