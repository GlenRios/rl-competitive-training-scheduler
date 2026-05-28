"""
preprocessor.py — Limpieza, extracción y muestreo estratificado

Responsabilidad:
    Toma el DataFrame crudo del fetcher y produce un dataset limpio de N
    problemas (default 500) listo para ser procesado por el embedder.

Pipeline interno
----------------
    1. build_problem_id   — contest (int) + problem_name (letra) → "325A"
    2. extract_rating     — parsea el token "*XXXX" dentro de problem_tags
    3. clean_tags         — elimina "*XXXX", normaliza a lista Python de strings
    4. clean_statement    — elimina whitespace excesivo y artefactos de scraping
    5. filter_valid       — descarta filas sin rating o sin statement
    6. stratified_sample  — muestrea N problemas balanceados por dificultad y tema

Formatos de tags soportados
----------------------------
    immortal3 : "implementation,greedy,*1500"
    erchhh    : "['dp', 'graphs', '*3000']"
    Ambos se normalizan a: ["implementation", "greedy"]  (rating extraído aparte)

Uso
---
    from src.data.fetcher import DatasetLoader
    from src.data.preprocessor import Preprocessor

    raw_df = DatasetLoader().load("data/raw/codeforces_dataset.csv", source="immortal3")
    pre    = Preprocessor(n_problems=500, random_seed=42)
    df     = pre.run(raw_df)
    df.to_csv("data/processed/problems_500.csv", index=False)
"""

import ast
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración de rangos de dificultad
#
# Define cuántos problemas se toman de cada banda de rating en el muestreo.
# La distribución está sesgada hacia ratings accesibles (aprendizaje gradual).
# Los porcentajes suman 1.0 y se convierten a conteos sobre n_problems.
# ---------------------------------------------------------------------------

DIFFICULTY_BINS = [
    (800,  1199, 0.25),   # Newbie / muy fácil
    (1200, 1599, 0.30),   # Pupil / fácil-medio
    (1600, 1999, 0.25),   # Specialist / medio
    (2000, 2399, 0.12),   # Expert / medio-difícil
    (2400, 2799, 0.05),   # Candidate Master / difícil
    (2800, 9999, 0.03),   # Master+ / muy difícil
]

# Patrones de limpieza del statement
_RE_RATING_TOKEN  = re.compile(r"\*\d{3,4}")   # "*1500", "*800"
_RE_WHITESPACE    = re.compile(r"[ \t]+")       # espacios y tabs múltiples
_RE_BLANK_LINES   = re.compile(r"\n{3,}")       # más de 2 saltos de línea seguidos
_RE_LATEX_INLINE  = re.compile(r"\$([^$]+)\$")  # $formula$ → formula
_RE_META_HEADER   = re.compile(                 # cabecera embebida en erchhh:
    r"^.*?time limit per test.*?standard output",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """Limpia y muestrea el dataset crudo del fetcher.

    Parameters
    ----------
    n_problems : int
        Número de problemas a incluir en el dataset final (default 500).
    random_seed : int
        Semilla para reproducibilidad del muestreo (default 42).
    min_statement_len : int
        Longitud mínima del statement en caracteres tras limpieza.
        Problemas con enunciados muy cortos suelen ser artifacts del scraping.
    """

    def __init__(
        self,
        n_problems: int = 500,
        random_seed: int = 42,
        min_statement_len: int = 80,
    ) -> None:
        self.n_problems       = n_problems
        self.random_seed      = random_seed
        self.min_statement_len = min_statement_len

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ejecuta el pipeline completo sobre el DataFrame del fetcher.

        Parameters
        ----------
        df : pd.DataFrame
            Salida directa de DatasetLoader.load() o load_and_merge().

        Returns
        -------
        pd.DataFrame
            Dataset limpio con n_problems filas y columnas:
            [problem_id, name, rating, tags, statement, time_limit_ms,
             solved_count, tags_list, difficulty_band]
        """
        logger.info(f"Preprocessor iniciado — {len(df):,} problemas de entrada")

        df = df.copy()
        df = self._build_problem_id(df)
        df = self._extract_rating(df)
        df = self._clean_tags(df)
        df = self._clean_statement(df)
        df = self._filter_valid(df)
        df = self._add_difficulty_band(df)
        df = self._stratified_sample(df)

        logger.info(f"Preprocessor completo — {len(df):,} problemas en el dataset final")
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Paso 1 — Construir problem_id
    # ------------------------------------------------------------------

    def _build_problem_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construye problem_id si aún no existe o está vacío.

        En immortal3 el fetcher lo construye como contest+problem_name ("325A").
        En erchhh generó un ID sintético. Aquí solo validamos.
        """
        if "problem_id" not in df.columns or df["problem_id"].isna().all():
            logger.warning("problem_id ausente — generando desde índice")
            df["problem_id"] = [f"prob_{i}" for i in range(len(df))]
        return df

    # ------------------------------------------------------------------
    # Paso 2 — Extraer rating desde tags
    # ------------------------------------------------------------------

    def _extract_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae el rating numérico del campo tags.

        Formatos soportados:
            "implementation,*1500"              → 1500
            "['dp', 'graphs', '*3000']"         → 3000
            "greedy,math"  (sin rating)         → pd.NA
        """
        def _parse_rating(raw: str) -> Optional[int]:
            if pd.isna(raw):
                return None
            match = _RE_RATING_TOKEN.search(str(raw))
            return int(match.group()[1:]) if match else None

        # Solo rellenar si rating no existe o tiene todos nulos
        if "rating" not in df.columns or df["rating"].isna().all():
            df["rating"] = df["tags"].apply(_parse_rating).astype("Int64")
            n_found = df["rating"].notna().sum()
            logger.info(f"  Rating extraído de tags: {n_found:,} / {len(df):,} problemas")
        else:
            # Completar nulos con lo extraído de tags
            extracted = df["tags"].apply(_parse_rating).astype("Int64")
            df["rating"] = df["rating"].combine_first(extracted)

        return df

    # ------------------------------------------------------------------
    # Paso 3 — Limpiar tags
    # ------------------------------------------------------------------

    def _clean_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza tags a lista Python de strings, sin el token de rating.

        Añade la columna `tags_list` (list[str]) y sobreescribe `tags`
        con la versión limpia en formato string separado por comas.

        Ejemplos:
            "implementation,greedy,*1500" → ["implementation", "greedy"]
            "['dp', 'graphs', '*3000']"   → ["dp", "graphs"]
        """
        def _parse_tags(raw: str) -> list[str]:
            if pd.isna(raw):
                return []
            raw = str(raw).strip()

            # Intentar parsear como lista Python ("['dp', 'graphs', '*3000']")
            if raw.startswith("["):
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, list):
                        return [
                            t.strip().lower()
                            for t in parsed
                            if isinstance(t, str) and not t.startswith("*")
                        ]
                except (ValueError, SyntaxError):
                    pass

            # Formato comma-separated: "implementation,greedy,*1500"
            return [
                t.strip().lower()
                for t in raw.split(",")
                if t.strip() and not t.strip().startswith("*")
            ]

        df["tags_list"] = df["tags"].apply(_parse_tags)
        df["tags"]      = df["tags_list"].apply(lambda lst: ",".join(lst))

        n_empty = (df["tags"] == "").sum()
        if n_empty > 0:
            logger.warning(f"  {n_empty} problemas con tags vacíos tras limpieza")

        return df

    # ------------------------------------------------------------------
    # Paso 4 — Limpiar statement
    # ------------------------------------------------------------------

    def _clean_statement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia el enunciado textual.

        Operaciones:
        - Elimina cabecera de metadata embebida (formato erchhh)
        - Elimina fórmulas LaTeX inline ($...$) → texto plano
        - Normaliza whitespace
        - Trunca a 2000 caracteres (suficiente para embeddings)
        """
        def _clean(text: str) -> str:
            if pd.isna(text):
                return ""
            text = str(text)

            # Eliminar cabecera de metadata de erchhh
            text = _RE_META_HEADER.sub("", text)

            # LaTeX inline: $n \leq 100$ → n \leq 100
            text = _RE_LATEX_INLINE.sub(r"\1", text)

            # Normalizar whitespace
            text = _RE_WHITESPACE.sub(" ", text)
            text = _RE_BLANK_LINES.sub("\n\n", text)
            text = text.strip()

            # Truncar (los embeddings no necesitan el problema completo)
            return text[:2000]

        df["statement"] = df["statement"].apply(_clean)
        return df

    # ------------------------------------------------------------------
    # Paso 5 — Filtrar registros inválidos
    # ------------------------------------------------------------------

    def _filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descarta filas que no son útiles para el pipeline.

        Criterios de descarte:
        - Sin rating (no pueden entrar al entorno de simulación)
        - Rating fuera del rango válido de Codeforces (800–3500)
        - Statement demasiado corto (artifact de scraping)
        """
        before = len(df)

        # Sin rating
        df = df[df["rating"].notna()]

        # Rating fuera de rango
        df = df[(df["rating"] >= 800) & (df["rating"] <= 3500)]

        # Statement demasiado corto
        df = df[df["statement"].str.len() >= self.min_statement_len]

        removed = before - len(df)
        logger.info(f"  Filtrado: {removed:,} problemas descartados → {len(df):,} válidos")
        return df

    # ------------------------------------------------------------------
    # Paso 6 — Añadir banda de dificultad
    # ------------------------------------------------------------------

    def _add_difficulty_band(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade columna `difficulty_band` con etiqueta legible del rango.

        Ejemplos: "800-1199", "1200-1599", "1600-1999", etc.
        """
        def _band(rating: int) -> str:
            for lo, hi, _ in DIFFICULTY_BINS:
                if lo <= rating <= hi:
                    return f"{lo}-{hi}"
            return "unknown"

        df["difficulty_band"] = df["rating"].apply(lambda r: _band(int(r)))
        return df

    # ------------------------------------------------------------------
    # Paso 7 — Muestreo estratificado
    # ------------------------------------------------------------------

    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Muestrea n_problems problemas balanceados por dificultad y tema.

        Estrategia
        ----------
        1. Para cada banda de dificultad, calcular el cupo (n_problems × porcentaje).
        2. Dentro de cada banda, ordenar por diversidad de tags:
           primero los problemas que introducen tags no vistos aún en esa banda.
        3. Si una banda no tiene suficientes problemas, el déficit se redistribuye
           a la banda inmediatamente anterior (más accesible).

        Returns
        -------
        pd.DataFrame con exactamente n_problems filas (o menos si el dataset
        total es menor que n_problems).
        """
        if len(df) <= self.n_problems:
            logger.warning(
                f"Dataset tiene solo {len(df):,} problemas — "
                f"se usan todos (pedidos: {self.n_problems})"
            )
            return df

        selected_frames = []
        deficit = 0

        for i, (lo, hi, pct) in enumerate(DIFFICULTY_BINS):
            # Cupo de esta banda + déficit acumulado de bandas previas
            quota = round(self.n_problems * pct) + deficit
            band_df = df[df["difficulty_band"] == f"{lo}-{hi}"].copy()

            if len(band_df) == 0:
                deficit += quota
                logger.warning(f"  Banda {lo}-{hi}: sin problemas disponibles")
                continue

            if len(band_df) <= quota:
                # Tomamos todos los disponibles y acumulamos déficit
                deficit = quota - len(band_df)
                selected_frames.append(band_df)
                logger.info(
                    f"  Banda {lo}-{hi}: {len(band_df):,} disponibles "
                    f"(cupo {quota}, déficit={deficit})"
                )
            else:
                # Muestreo con diversidad de tags dentro de la banda
                sampled = self._sample_diverse(band_df, quota)
                deficit = 0
                selected_frames.append(sampled)
                logger.info(f"  Banda {lo}-{hi}: {quota} seleccionados de {len(band_df):,}")

        result = pd.concat(selected_frames, ignore_index=True)

        # Si hay déficit final (bandas altas vacías), completar desde el pool general
        if len(result) < self.n_problems:
            remaining = df[~df["problem_id"].isin(result["problem_id"])]
            extra_needed = self.n_problems - len(result)
            extra = remaining.sample(
                min(extra_needed, len(remaining)),
                random_state=self.random_seed,
            )
            result = pd.concat([result, extra], ignore_index=True)
            logger.info(f"  Completado con {len(extra)} problemas extra del pool general")

        return result.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

    def _sample_diverse(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Selecciona n problemas maximizando diversidad de tags.

        Algoritmo greedy: en cada iteración elige el problema que añade
        el mayor número de tags nuevos al conjunto ya seleccionado.
        Para grupos grandes usa un pre-filtro aleatorio para eficiencia.
        """
        rng = pd.Series(range(len(df))).sample(
            min(len(df), n * 5),
            random_state=self.random_seed,
        ).tolist()
        pool = df.iloc[rng].copy()

        selected_indices = []
        seen_tags: set[str] = set()

        # Greedy hasta llenar el cupo
        remaining = pool.copy()
        while len(selected_indices) < n and len(remaining) > 0:
            # Score = número de tags nuevos que aporta cada problema
            scores = remaining["tags_list"].apply(
                lambda tlist: len(set(tlist) - seen_tags)
            )
            best_idx = scores.idxmax()
            selected_indices.append(best_idx)
            seen_tags.update(remaining.loc[best_idx, "tags_list"])
            remaining = remaining.drop(index=best_idx)

        result = pool.loc[selected_indices]

        # Si greedy no llenó el cupo (todos los tags ya vistos), completar aleatoriamente
        if len(result) < n:
            extra = df[~df.index.isin(selected_indices)].sample(
                min(n - len(result), len(df) - len(result)),
                random_state=self.random_seed,
            )
            result = pd.concat([result, extra])

        return result.head(n)