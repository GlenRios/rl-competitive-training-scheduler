"""
4_LLM_Evaluator.py -- Evaluador pedagogico con LLM

Envia la secuencia de problemas al LLM (Ollama) y muestra
una puntuacion pedagogica y una explicacion en lenguaje natural.
"""

import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import streamlit as st
import requests

from state import init_state, load_data
from styles import inject_styles
from components.sidebar import render_sidebar

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2:3b"

st.set_page_config(page_title="Evaluador LLM", layout="wide")
inject_styles()
init_state()

if not load_data():
    st.error("Datos no cargados.")
    st.stop()

render_sidebar()

st.title("EVALUADOR PEDAGOGICO LLM")
st.markdown("Usa el LLM local (Ollama) para evaluar la calidad pedagogica de una sesion de entrenamiento.")
st.markdown("---")

# ── Verificar Ollama ───────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except:
        return False

ollama_ok = check_ollama()
if ollama_ok:
    st.success("Ollama conectado correctamente.")
else:
    st.error("Ollama no disponible. Asegurate de que este corriendo con: ollama serve")

# ── Sesion a evaluar ───────────────────────────────────────────────────────
st.markdown("### Sesion a evaluar")

history = st.session_state.get("sim_history", [])
if not history:
    st.info("No hay sesion completada aun. Ve a la pagina Simulation, completa una sesion y vuelve aqui.")
    st.stop()

# Resumen de la sesion
st.caption(f"Sesion actual: {len(history)} pasos | "
           f"{sum(1 for h in history if h['solved'])} resueltos")

# Mostrar secuencia
with st.expander("Ver secuencia de problemas"):
    for h in history:
        status = "RESUELTO" if h["solved"] else "FALLIDO"
        st.markdown(
            f"**Paso {h['step']}** | {h['problem_id']} | "
            f"rating {h['problem_rating']} | "
            f"tags: {', '.join(h['problem_tags'][:3])} | "
            f"**{status}** | reward: {h['reward']:+.1f}"
        )

st.markdown("---")

# ── Evaluacion pedagogica ──────────────────────────────────────────────────

def build_eval_prompt(history: list[dict], student) -> str:
    lines = []
    for h in history:
        status = "solved" if h["solved"] else "failed"
        lines.append(
            f"Step {h['step']}: problem {h['problem_id']} "
            f"(rating={h['problem_rating']}, tags={h['problem_tags'][:3]}) "
            f"-> {status}, reward={h['reward']:+.1f}"
        )

    sequence = "\n".join(lines)

    return f"""You are an expert competitive programming coach evaluating a training session.

A student completed the following problem sequence:
{sequence}

Student final state:
- Global rating: {student.global_rating:.0f}
- Problems solved: {student.n_solved} / {student.n_attempted}
- Topics covered: {list(student.topics_seen)[:8]}

Evaluate this training session on a scale from 1 to 10. Consider:
1. Difficulty progression (are problems ordered from easier to harder?)
2. Topic diversity (are different topics covered?)
3. Appropriate challenge (are problems at the right difficulty level?)
4. Learning efficiency (is time used well?)

Respond in this exact JSON format:
{{
  "score": <number 1-10>,
  "difficulty_progression": "<good/average/poor>",
  "topic_diversity": "<good/average/poor>",
  "challenge_level": "<too easy/appropriate/too hard>",
  "explanation": "<2-3 sentence explanation in Spanish>",
  "recommendation": "<1 sentence improvement suggestion in Spanish>"
}}"""


col1, col2 = st.columns(2)

with col1:
    st.markdown("### Puntuacion Pedagogica")
    if st.button("Evaluar sesion", disabled=not ollama_ok, use_container_width=True, type="primary"):
        student = st.session_state.get("sim_student")
        if not student:
            st.error("Estudiante no disponible.")
        else:
            prompt = build_eval_prompt(history, student)
            with st.spinner("Consultando al LLM..."):
                try:
                    resp = requests.post(
                        OLLAMA_URL,
                        json={"model": MODEL, "prompt": prompt, "stream": False,
                              "options": {"temperature": 0.3}},
                        timeout=60,
                    )
                    raw = resp.json().get("response", "")
                    match = __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL)
                    if match:
                        data = json.loads(match.group())
                        score = data.get("score", 0)

                        # Mostrar score grande
                        color = "#00e676" if score >= 7 else "#ffab40" if score >= 5 else "#ff5252"
                        st.markdown(
                            f'<div style="text-align:center; padding:20px;">'
                            f'<span style="font-size:72px; font-weight:700; '
                            f'color:{color}; font-family: JetBrains Mono">{score}</span>'
                            f'<span style="color:#7ecfff; font-size:24px">/10</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Progresion",   data.get("difficulty_progression", "-"))
                        c2.metric("Diversidad",   data.get("topic_diversity", "-"))
                        c3.metric("Desafio",      data.get("challenge_level", "-"))

                        st.markdown("**Evaluacion:**")
                        st.info(data.get("explanation", "-"))
                        st.markdown("**Recomendacion:**")
                        st.warning(data.get("recommendation", "-"))
                    else:
                        st.text_area("Respuesta raw del LLM:", raw, height=200)

                except Exception as e:
                    st.error(f"Error consultando Ollama: {e}")


# ── Exportar sesion ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Exportar sesion")

col1, col2 = st.columns(2)
with col1:
    if history:
        import pandas as pd
        df = pd.DataFrame(history)
        st.download_button(
            "Descargar sesion CSV",
            data      = df.to_csv(index=False).encode("utf-8"),
            file_name = "session.csv",
            mime      = "text/csv",
        )
with col2:
    if history:
        st.download_button(
            "Descargar sesion JSON",
            data      = json.dumps(history, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name = "session.json",
            mime      = "application/json",
        )