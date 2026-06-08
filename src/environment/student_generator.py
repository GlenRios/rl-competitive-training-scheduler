"""
student_generator.py -- Generador de perfiles de estudiante via LLM

Responsabilidad unica:
    Usar Ollama (llama3.2:3b) para generar perfiles de estudiante
    realistas y coherentes al inicio de cada episodio de entrenamiento.

Por que usar un LLM para esto
------------------------------
    Un perfil aleatorio uniforme (todos los temas en rating random)
    no es realista: los estudiantes reales tienen fortalezas y debilidades
    correlacionadas (quien es bueno en dp suele ser razonable en math,
    quien es fuerte en graphs suele conocer trees y dfs).

    El LLM captura estas correlaciones porque ha procesado miles de
    descripciones de estudiantes de programacion competitiva.

Perfil generado
---------------
    {
      "archetype":    "string -- descripcion breve del tipo de estudiante",
      "global_rating": int,
      "session_budget_min": int,
      "topic_ratings": {
        "math":               int,
        "dp":                 int,
        "graphs":             int,
        ... (20 temas canonicos)
      }
    }

Fallback
--------
    Si Ollama no esta disponible o la respuesta no es parseable,
    el generador usa perfiles predefinidos (arquetipos hardcoded)
    para no bloquear el entrenamiento.

Uso
---
    from src.environment.student_generator import StudentProfileGenerator

    gen     = StudentProfileGenerator(model="llama3.2:3b")
    profile = gen.generate()
    student = profile.to_student_model()
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

from src.environment.problem import CANONICAL_TOPICS
from src.environment.student_model import StudentModel

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL  = "http://localhost:11434"
REQUEST_TIMEOUT  = 30

# ---------------------------------------------------------------------------
# Arquetipos predefinidos (fallback sin LLM)
# ---------------------------------------------------------------------------

_ARCHETYPES: list[dict] = [
    {
        "archetype": "Math specialist - strong in math and number theory, weak in graphs",
        "global_rating": 1400,
        "session_budget_min": 120,
        "topic_ratings": {
            "math": 1800, "number theory": 1700, "combinatorics": 1600,
            "implementation": 1500, "greedy": 1400, "binary search": 1300,
            "dp": 1200, "strings": 1200, "sortings": 1300, "brute force": 1100,
            "graphs": 900, "trees": 950, "dfs and similar": 900,
            "shortest paths": 850, "data structures": 1000,
            "constructive algorithms": 1200, "two pointers": 1100,
            "bitmasks": 1000, "geometry": 900, "hashing": 1100,
        }
    },
    {
        "archetype": "Graph expert - strong in graphs and trees, average in math",
        "global_rating": 1600,
        "session_budget_min": 120,
        "topic_ratings": {
            "graphs": 1900, "trees": 1850, "dfs and similar": 1800,
            "shortest paths": 1750, "data structures": 1700,
            "dp": 1500, "greedy": 1400, "implementation": 1500,
            "binary search": 1400, "sortings": 1400,
            "math": 1200, "number theory": 1100, "combinatorics": 1100,
            "constructive algorithms": 1300, "two pointers": 1300,
            "bitmasks": 1200, "brute force": 1000, "strings": 1100,
            "geometry": 950, "hashing": 1200,
        }
    },
    {
        "archetype": "DP master - excellent at dynamic programming, struggles with geometry",
        "global_rating": 1700,
        "session_budget_min": 120,
        "topic_ratings": {
            "dp": 2000, "bitmasks": 1700, "data structures": 1700,
            "math": 1500, "greedy": 1600, "binary search": 1600,
            "graphs": 1400, "trees": 1400, "sortings": 1500,
            "implementation": 1600, "constructive algorithms": 1500,
            "number theory": 1300, "combinatorics": 1400,
            "dfs and similar": 1300, "shortest paths": 1300,
            "two pointers": 1500, "strings": 1200, "hashing": 1300,
            "brute force": 1000, "geometry": 900,
        }
    },
    {
        "archetype": "Beginner - all topics around 900-1100, just starting competitive programming",
        "global_rating": 1000,
        "session_budget_min": 90,
        "topic_ratings": {
            "implementation": 1100, "brute force": 1000, "math": 1000,
            "greedy": 950, "sortings": 1000, "binary search": 900,
            "strings": 900, "dp": 850, "data structures": 900,
            "graphs": 850, "trees": 850, "dfs and similar": 800,
            "shortest paths": 800, "constructive algorithms": 900,
            "number theory": 850, "combinatorics": 850, "two pointers": 900,
            "bitmasks": 800, "geometry": 800, "hashing": 850,
        }
    },
    {
        "archetype": "Balanced intermediate - no strong specialization, solid fundamentals",
        "global_rating": 1350,
        "session_budget_min": 120,
        "topic_ratings": {
            t: random.randint(1200, 1500) for t in CANONICAL_TOPICS
        }
    },
]

# ---------------------------------------------------------------------------
# StudentProfile
# ---------------------------------------------------------------------------

@dataclass
class StudentProfile:
    """Perfil generado para un estudiante simulado.

    Attributes
    ----------
    archetype           : str   -- descripcion del tipo de estudiante
    global_rating       : float -- rating global aproximado
    session_budget_min  : float -- duracion de la sesion en minutos
    topic_ratings       : dict  -- {tema: rating} para los 20 temas canonicos
    source              : str   -- "llm" o "fallback"
    """
    archetype          : str
    global_rating      : float
    session_budget_min : float
    topic_ratings      : dict[str, float]
    source             : str = "llm"

    def to_student_model(self, random_seed: Optional[int] = None) -> StudentModel:
        """Construye un StudentModel con este perfil."""
        return StudentModel(
            topic_ratings      = self.topic_ratings,
            global_rating      = self.global_rating,
            session_budget_min = self.session_budget_min,
            random_seed        = random_seed,
        )

    def __repr__(self) -> str:
        top3 = sorted(
            self.topic_ratings.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top3_str = ", ".join(f"{t}={int(r)}" for t, r in top3)
        return (
            f"StudentProfile({self.archetype!r}, "
            f"global={self.global_rating:.0f}, "
            f"top3=[{top3_str}], source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# StudentProfileGenerator
# ---------------------------------------------------------------------------

class StudentProfileGenerator:
    """Genera perfiles de estudiante usando Ollama como LLM local.

    Parameters
    ----------
    model       : str   -- modelo Ollama a usar (default: llama3.2:3b)
    base_url    : str   -- URL de Ollama
    use_llm     : bool  -- si False, usa solo arquetipos hardcoded
    random_seed : int | None
    """

    _PROMPT_TEMPLATE = """You are generating a realistic competitive programming student profile for a training simulation.

Generate a JSON profile for a student with the following constraints:
- The profile must be internally consistent (e.g., strong dp students tend to be decent at math)
- topic_ratings must be integers between 800 and 3000
- global_rating should roughly match the average of topic_ratings
- session_budget_min should be between 60 and 180

Respond with ONLY valid JSON, no markdown, no explanation.

Use exactly these topic keys: {topics}

Example format:
{{
  "archetype": "brief description of student type",
  "global_rating": 1400,
  "session_budget_min": 120,
  "topic_ratings": {{
    "math": 1600,
    "dp": 1300,
    ...all 20 topics...
  }}
}}

Generate a profile for: {student_type}"""

    _STUDENT_TYPES = [
        "a math specialist who struggles with graph problems",
        "a graph theory expert with average dynamic programming skills",
        "a dynamic programming master who avoids geometry",
        "a well-rounded intermediate student with no clear specialty",
        "a beginner just starting competitive programming",
        "a string algorithms specialist",
        "an advanced student strong in data structures and algorithms",
        "a student who excels at greedy and constructive problems",
        "a competitive programmer with strong number theory background",
        "a student who is good at implementation but weak at proofs",
    ]

    def __init__(
        self,
        model       : str  = "llama3.2:3b",
        base_url    : str  = OLLAMA_BASE_URL,
        use_llm     : bool = True,
        random_seed : Optional[int] = None,
    ) -> None:
        self.model    = model
        self.base_url = base_url
        self.use_llm  = use_llm
        self._rng     = random.Random(random_seed)

    def generate(self) -> StudentProfile:
        """Genera un perfil de estudiante.

        Intenta usar el LLM primero. Si falla, usa un arquetipo predefinido.

        Returns
        -------
        StudentProfile listo para convertir a StudentModel.
        """
        if self.use_llm:
            profile = self._generate_from_llm()
            if profile is not None:
                return profile

        return self._generate_from_archetypes()

    def generate_batch(self, n: int) -> list[StudentProfile]:
        """Genera n perfiles distintos."""
        return [self.generate() for _ in range(n)]

    def generate_and_save(self, n: int, path: str) -> list[StudentProfile]:
        """Genera n perfiles, los guarda en JSON y los devuelve.

        Parameters
        ----------
        n    : int  -- numero de perfiles a generar
        path : str  -- ruta del archivo JSON de salida

        Returns
        -------
        list[StudentProfile]
        """
        import json as _json
        from pathlib import Path as _Path

        logger.info(f"Generando {n} perfiles de estudiante...")
        profiles = []
        for i in range(n):
            profile = self.generate()
            profiles.append(profile)
            if (i + 1) % 10 == 0 or (i + 1) == n:
                logger.info(f"  {i+1}/{n} perfiles generados ({profile.source})")

        _Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "archetype"          : p.archetype,
                "global_rating"      : p.global_rating,
                "session_budget_min" : p.session_budget_min,
                "topic_ratings"      : p.topic_ratings,
                "source"             : p.source,
            }
            for p in profiles
        ]
        _Path(path).write_text(
            _json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"  {n} perfiles guardados en {path}")
        return profiles

    @staticmethod
    def load_profiles(path: str) -> list[StudentProfile]:
        """Carga perfiles desde un archivo JSON generado previamente.

        Parameters
        ----------
        path : str -- ruta al archivo JSON

        Returns
        -------
        list[StudentProfile]

        Raises
        ------
        FileNotFoundError si el archivo no existe.
        """
        import json as _json
        from pathlib import Path as _Path

        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo de perfiles: {path}\n"
                "Ejecuta primero: generator.generate_and_save(n, path)"
            )

        data     = _json.loads(p.read_text(encoding="utf-8"))
        profiles = []
        for d in data:
            profiles.append(StudentProfile(
                archetype          = d["archetype"],
                global_rating      = float(d["global_rating"]),
                session_budget_min = float(d["session_budget_min"]),
                topic_ratings      = {k: float(v) for k, v in d["topic_ratings"].items()},
                source             = d.get("source", "loaded"),
            ))
        logger.info(f"  {len(profiles)} perfiles cargados desde {path}")
        return profiles

    # ------------------------------------------------------------------
    # Generacion via LLM
    # ------------------------------------------------------------------

    def _generate_from_llm(self) -> Optional[StudentProfile]:
        """Llama a Ollama y parsea la respuesta JSON."""
        student_type = self._rng.choice(self._STUDENT_TYPES)
        topics_str   = ", ".join(f'"{t}"' for t in CANONICAL_TOPICS)

        prompt = self._PROMPT_TEMPLATE.format(
            topics       = topics_str,
            student_type = student_type,
        )

        payload = {
            "model" : self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json    = payload,
                timeout = REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
            return self._parse_llm_response(raw_text)

        except requests.RequestException as e:
            logger.warning(f"Ollama no disponible, usando arquetipo predefinido: {e}")
            return None

    def _parse_llm_response(self, raw_text: str) -> Optional[StudentProfile]:
        """Parsea y valida el JSON devuelto por el LLM."""
        # Extraer bloque JSON de la respuesta
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            logger.warning("LLM no devolvio JSON valido")
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"JSON malformado en respuesta LLM: {e}")
            return None

        return self._validate_and_build(data, source="llm")

    def _validate_and_build(
        self, data: dict, source: str = "llm"
    ) -> Optional[StudentProfile]:
        """Valida el dict y construye un StudentProfile."""
        try:
            topic_ratings_raw = data.get("topic_ratings", {})

            # Completar temas faltantes con valor por defecto
            topic_ratings: dict[str, float] = {}
            for topic in CANONICAL_TOPICS:
                val = topic_ratings_raw.get(topic, 1200)
                # Clampar a rango realista (max 1900 para evitar perfiles irreales)
                topic_ratings[topic] = max(800, min(1900, float(val)))

            global_rating = float(
                data.get("global_rating",
                         sum(topic_ratings.values()) / len(topic_ratings))
            )
            global_rating = max(800, min(1900, global_rating))

            budget = float(data.get("session_budget_min", 120))
            budget = max(60, min(180, budget))

            return StudentProfile(
                archetype          = str(data.get("archetype", "Unknown archetype")),
                global_rating      = global_rating,
                session_budget_min = budget,
                topic_ratings      = topic_ratings,
                source             = source,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error construyendo StudentProfile: {e}")
            return None

    # ------------------------------------------------------------------
    # Fallback: arquetipos predefinidos
    # ------------------------------------------------------------------

    def _generate_from_archetypes(self) -> StudentProfile:
        """Selecciona aleatoriamente uno de los arquetipos predefinidos."""
        archetype_data = self._rng.choice(_ARCHETYPES).copy()

        # Anadir variacion aleatoria pequeña (+/- 100 puntos por tema)
        topic_ratings = {}
        for topic in CANONICAL_TOPICS:
            base  = archetype_data["topic_ratings"].get(topic, 1200)
            noise = self._rng.randint(-100, 100)
            topic_ratings[topic] = max(800, min(3000, base + noise))

        profile = self._validate_and_build(
            {
                "archetype"          : archetype_data["archetype"],
                "global_rating"      : archetype_data["global_rating"],
                "session_budget_min" : archetype_data["session_budget_min"],
                "topic_ratings"      : topic_ratings,
            },
            source="fallback",
        )
        return profile  # siempre valido con datos predefinidos