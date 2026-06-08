"""
main.py -- Pagina de inicio

Ejecutar con:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

# Anadir la raiz del proyecto a sys.path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from app.state import init_state, load_data
from app.styles import inject_styles

st.set_page_config(
    page_title = "RL Training Scheduler",
    page_icon  = "brain",
    layout     = "wide",
)

inject_styles()
init_state()

st.title("RL COMPETITIVE TRAINING SCHEDULER")
st.markdown(
    "Sistema de seleccion optima de problemas para entrenamiento competitivo "
    "usando Deep Q-Network con modelo de estudiante basado en ELO por tema."
)
st.markdown("---")

ok = load_data()

c1, c2, c3 = st.columns(3)
problems = st.session_state.get("problems", [])
profiles = st.session_state.get("profiles", [])
agent    = st.session_state.get("agent")

c1.metric("Problemas cargados",     len(problems) if problems else "---")
c2.metric("Perfiles de estudiante", len(profiles) if profiles else "---")
c3.metric("Agente DQN",             "Listo" if agent else "No entrenado")

if not ok:
    st.warning(
        "Datos no encontrados. Ejecuta primero `python -m main` para "
        "generar el dataset y entrenar el agente."
    )
else:
    st.success("Sistema listo. Navega a las paginas del menu lateral.")

st.markdown("---")
st.markdown("### GUIA RAPIDA")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**1 - Simulation**
Ejecuta una sesion paso a paso con el algoritmo elegido.
Observa como evoluciona el ELO del estudiante problema a problema.

**2 - Comparison**
Compara los 4 algoritmos (Greedy, Mochila, Rollout, DQN)
con estadisticas sobre 30 episodios por algoritmo.
""")
with col2:
    st.markdown("""
**3 - Training**
Entrena el agente DQN interactivamente con parametros configurables.
Observa la curva de aprendizaje en tiempo real.

**4 - LLM Evaluator**
Evalua pedagogicamente una sesion usando el LLM local (Ollama).
Obtiene una puntuacion y explicacion de la secuencia generada.
""")