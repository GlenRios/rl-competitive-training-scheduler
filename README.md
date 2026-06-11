# rl-competitive-training-scheduler

A Deep Q-Network agent that selects and sequences competitive programming problems to maximize learning within a time-constrained training session. The system uses real Codeforces problem data, a simulated student model with per-topic ELO ratings, and LLM-based pedagogical evaluation via Ollama.

---

## Project Structure

```
├── app/                        Streamlit dashboard (4 pages)
│   ├── pages/
│   │   ├── 1_Simulation.py     Step-by-step session visualization
│   │   ├── 2_Comparison.py     Multi-algorithm comparison
│   │   ├── 3_Training.py       Interactive DQN training
│   │   └── 4_LLM_Evaluator.py  Pedagogical evaluation via Ollama
│   ├── components/             Reusable UI components
│   ├── main.py
│   └── state.py
├── src/
│   ├── agent/                  DQN agent, replay buffer, trainer
│   ├── baselines/              Greedy, Knapsack, Rollout selectors
│   ├── data/                   Dataset loader and preprocessor
│   ├── environment/            Gymnasium env, student model, problem
│   └── eval/                   Evaluator and metrics
├── data/
│   ├── raw/                    Codeforces CSV datasets (Kaggle)
│   └── processed/              problems_500.csv, student_profiles.json
├── experiments/results/        Saved DQN checkpoint (best_agent.pt)
├── tests/                      8 test modules (pytest)
└── requirements.txt
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional — for LLM Evaluator page
ollama pull llama3.2:3b
```

---

## Usage

**Run the dashboard:**
```bash
streamlit run app/main.py
```

**Train the DQN agent:**
```bash
python train.py

# With custom parameters:
python train.py --n_episodes 500 --lr 1e-3 --epsilon_end 0.05
python train.py --n_episodes 300 --checkpoint_dir experiments/results
```

**Run the evaluation:**
```python
from src.eval.evaluator import Evaluator
from src.baselines.greedy import GreedySelector
from src.environment.problem import Problem
import pandas as pd

problems = Problem.from_dataframe(pd.read_csv("data/processed/problems_500.csv"))
evaluator = Evaluator(problems=problems, n_episodes=30)
results = evaluator.run({"greedy": GreedySelector()})
```

**Run tests:**
```bash
pytest
```

---

## LLM Integration (Ollama)

The system uses `llama3.2:3b` locally in two ways:

- **Student profile generation** — produces realistic per-topic ELO distributions correlated by archetype (e.g. "Math specialist weak in graphs").
- **Pedagogical evaluation** — scores a completed session (1–10) across difficulty progression, topic diversity, and challenge level.

Requires Ollama running locally (`ollama serve`). Falls back to predefined archetypes if unavailable.

---

## Dependencies

```
pandas · pytest · requests · gymnasium · torch
streamlit · plotly · wordcloud · matplotlib
```

See `informe.pdf` for full technical documentation including formal MDP model, algorithm design, experimental results, and analysis.