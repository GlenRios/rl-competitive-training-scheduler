"""
evaluator.py -- Evaluador comparativo de selectores

Responsabilidad unica:
    Ejecutar N episodios por cada combinacion (selector, perfil de estudiante),
    calcular metricas por episodio y devolver un DataFrame consolidado.

Flujo
-----
    Para cada selector:
        Para cada perfil de estudiante:
            Para cada episodio (1..N):
                1. Crear StudentModel desde el perfil
                2. Ejecutar episodio con el selector
                3. Calcular metricas
                4. Registrar fila en el DataFrame

    El resultado final es un DataFrame con una fila por episodio,
    listo para analisis estadistico y visualizacion.

Compatibilidad
--------------
    El evaluador es compatible con cualquier objeto que implemente
    la interfaz ProblemSelector (baselines y agente DQN).

Uso
---
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(problems=problems, n_episodes=30)

    results = evaluator.run({
        "greedy"   : GreedySelector(),
        "knapsack" : KnapsackSelector(),
        "rollout"  : RolloutSelector(),
        "dqn"      : DQNSelectorAdapter(agent),
    }, profiles=profiles)

    results.to_csv("experiments/results/evaluation.csv", index=False)
"""

import logging
import time
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd

from src.environment.problem import Problem
from src.environment.student_generator import StudentProfile
from src.environment.student_model import StudentModel
from src.eval.metrics import compute_episode_metrics, aggregate_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Ejecuta episodios de evaluacion para comparar selectores.

    Parameters
    ----------
    problems   : list[Problem]  -- pool de problemas (identico para todos)
    n_episodes : int            -- episodios por combinacion (default 30)
    random_seed: int | None     -- semilla base para reproducibilidad
    verbose    : bool           -- loggear progreso detallado
    """

    def __init__(
        self,
        problems    : list[Problem],
        n_episodes  : int  = 30,
        random_seed : Optional[int] = None,
        verbose     : bool = True,
    ) -> None:
        self.problems    = problems
        self.n_episodes  = n_episodes
        self.random_seed = random_seed
        self.verbose     = verbose

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def run(
        self,
        selectors : dict,
        profiles  : list[StudentProfile],
    ) -> pd.DataFrame:
        """Ejecuta la evaluacion completa.

        Parameters
        ----------
        selectors : dict[str, ProblemSelector]
            Mapa de nombre -> selector. Puede incluir baselines y el agente DQN.
        profiles  : list[StudentProfile]
            Perfiles de estudiante a evaluar. Se usan los primeros n_episodes
            (o se ciclan si hay menos que n_episodes).

        Returns
        -------
        pd.DataFrame con columnas:
            selector, archetype, global_rating, episode,
            total_reward, spearman_progression, topic_coverage,
            success_rate, time_used_min, rating_improvement,
            n_attempted, n_solved, compute_time_s
        """
        rows       = []
        total_runs = len(selectors) * self.n_episodes * len(profiles[:self.n_episodes])

        logger.info("=" * 60)
        logger.info("  EVALUACION COMPARATIVA")
        logger.info(f"  Selectores  : {list(selectors.keys())}")
        logger.info(f"  Perfiles    : {len(profiles)}")
        logger.info(f"  Episodios   : {self.n_episodes} por combinacion")
        logger.info(f"  Total runs  : {len(selectors) * self.n_episodes}")
        logger.info("=" * 60)

        for sel_name, selector in selectors.items():
            logger.info(f"\n  Evaluando selector: [{sel_name}]")
            selector.reset()

            sel_rows = self._evaluate_selector(sel_name, selector, profiles)
            rows.extend(sel_rows)

            # Resumen por selector
            sel_df  = pd.DataFrame(sel_rows)
            summary = aggregate_metrics(sel_rows)
            logger.info(
                f"  [{sel_name}] reward={summary.get('total_reward_mean', 0):.2f}"
                f" (+/-{summary.get('total_reward_std', 0):.2f})"
                f" | solved_rate={summary.get('success_rate_mean', 0):.3f}"
                f" | topics={summary.get('topic_coverage_mean', 0):.1f}"
                f" | rating_gain={summary.get('rating_improvement_mean', 0):.1f}"
            )

        df = pd.DataFrame(rows)
        logger.info("\n" + "=" * 60)
        logger.info("  EVALUACION COMPLETA")
        logger.info(f"  Total filas : {len(df)}")
        logger.info("=" * 60)
        return df

    def run_single_selector(
        self,
        selector_name : str,
        selector,
        profiles      : list[StudentProfile],
    ) -> pd.DataFrame:
        """Evalua un unico selector. Util para evaluar solo el DQN."""
        rows = self._evaluate_selector(selector_name, selector, profiles)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Ejecucion por selector
    # ------------------------------------------------------------------

    def _evaluate_selector(
        self,
        selector_name : str,
        selector,
        profiles      : list[StudentProfile],
    ) -> list[dict]:
        """Ejecuta n_episodes episodios para un selector."""
        rows = []
        for ep_idx in range(self.n_episodes):
            # Ciclar perfiles si hay menos que episodios
            profile = profiles[ep_idx % len(profiles)]

            seed = (
                (self.random_seed or 0) + ep_idx * 31
                if self.random_seed is not None else None
            )

            t_start = time.time()
            history, student = self._run_episode(selector, profile, seed)
            elapsed = time.time() - t_start

            metrics = compute_episode_metrics(
                history        = history,
                initial_rating = profile.global_rating,
                final_rating   = student.global_rating,
                compute_time_s = elapsed,
            )

            row = {
                "selector"     : selector_name,
                "archetype"    : profile.archetype,
                "global_rating": profile.global_rating,
                "episode"      : ep_idx + 1,
                **metrics,
            }
            rows.append(row)

            if self.verbose and (ep_idx + 1) % 10 == 0:
                logger.info(
                    f"    [{selector_name}] ep {ep_idx+1}/{self.n_episodes}"
                    f" | reward={metrics['total_reward']:.1f}"
                    f" | solved={metrics['n_solved']}/{metrics['n_attempted']}"
                    f" | topics={metrics['topic_coverage']}"
                    f" | gain={metrics['rating_improvement']:+.1f}"
                )

        return rows

    # ------------------------------------------------------------------
    # Ejecucion de un episodio
    # ------------------------------------------------------------------

    def _run_episode(
        self,
        selector,
        profile  : StudentProfile,
        seed     : Optional[int],
    ) -> tuple[list[dict], StudentModel]:
        """Ejecuta un episodio completo con un selector dado.

        Returns
        -------
        (history, student_final_state)
        """
        from src.baselines.selector import SessionState

        student  = profile.to_student_model(random_seed=seed)
        history  = []
        attempted: set[int] = set()

        while not student.session_over:
            # Construir mascara de acciones validas
            available_mask = self._build_mask(student, attempted)
            if not any(available_mask):
                break

            # Construir estado de sesion para el selector
            state = self._build_state(student, available_mask)

            # Seleccionar accion
            try:
                action = selector.select_action(
                    state          = state,
                    available_mask = available_mask,
                )
            except Exception as e:
                logger.warning(f"Selector error: {e} -- terminando episodio")
                break

            # Accion de parar (N = len(problems))
            if action >= len(self.problems):
                break

            # Validar accion
            if not available_mask[action]:
                logger.warning(f"Selector eligio accion invalida {action} -- saltando")
                break

            # Ejecutar intento
            problem = self.problems[action]
            outcome = student.attempt(
                problem_rating = problem.rating,
                problem_tags   = problem.tags_list,
                problem_id     = problem.problem_id,
            )
            attempted.add(action)

            step = {
                "step"             : len(history) + 1,
                "problem_id"       : problem.problem_id,
                "problem_rating"   : problem.rating,
                "problem_tags"     : problem.tags_list,
                "solved"           : outcome.solved,
                "p_solve"          : outcome.p_solve,
                "time_min"         : outcome.time_min,
                "reward"           : outcome.reward,
                "topic_deltas"     : outcome.topic_deltas,
                "new_global_rating": outcome.new_global_rating,
                "fatigue"          : outcome.fatigue,
                "time_remaining"   : outcome.time_remaining_min,
            }
            history.append(step)

        return history, student

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_mask(
        self, student: StudentModel, attempted: set[int]
    ) -> list[bool]:
        """Construye la mascara de acciones validas."""
        mask = []
        for i, problem in enumerate(self.problems):
            if i in attempted:
                mask.append(False)
            else:
                t_est = student.estimate_solve_time(problem.rating, problem.tags_list)
                mask.append(t_est <= student.time_remaining_min)
        return mask

    def _build_state(self, student: StudentModel, available_mask: list[bool]):
        """Construye el SessionState para el selector."""
        from src.baselines.selector import SessionState

        return SessionState(
            student            = student,
            problems           = self.problems,
            available_mask     = np.array(available_mask, dtype=bool),
            session_budget_min = student.session_budget_min,
        )