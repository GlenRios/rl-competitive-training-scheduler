"""
fetcher.py — Módulo 1: Ingesta de datos

Responsabilidad: cargar uno o dos datasets de Kaggle con problemas de Codeforces
desde archivos locales (CSV/JSON), aplicar el mapeo de columnas al esquema canónico
del proyecto, validar integridad básica y opcionalmente combinar ambas fuentes.

Esquema canónico de salida
--------------------------
    problem_id   : str   — identificador único (ej. "325A")
    name         : str   — título del problema; pd.NA si no disponible
    rating       : Int64 — dificultad numérica (800–3500); pd.NA si desconocida
                           NOTA: se extrae de tags en el preprocessor, aquí es pd.NA
    tags         : str   — tags crudos tal como vienen del dataset
    statement    : str   — enunciado textual completo
    time_limit_ms: Int64 — límite de tiempo en milisegundos; pd.NA si no disponible
    solved_count : Int64 — número de usuarios que resolvieron; pd.NA si no disponible

Columnas reales observadas en cada dataset
------------------------------------------
immortal3 (codeforces_dataset.csv):
    contest           int64  — ID del contest (ej. 325)
    problem_name      str    — letra del problema (ej. "A")
    problem_statement str    — enunciado completo
    problem_tags      str    — "implementation,*1500" (rating embebido como *XXXX)

erchhh (codeforces_problemset.csv):
    problem_statement str    — enunciado (con metadata embebida al inicio)
    input             str    — ejemplo de entrada
    output            str    — ejemplo de salida
    time_limit        str    — "4 seconds"
    memory_limit      str    — "256 megabytes"
    tags              str    — "['dp', 'graphs', '*3000']"

NOTA: el rating no es columna separada en ninguno de los dos datasets.
Está embebido en tags como "*XXXX". Lo extrae el preprocessor.

Uso básico
----------
    from src.data.fetcher import DatasetLoader

    loader = DatasetLoader()

    # Fuente única
    df = loader.load("data/raw/codeforces_dataset.csv", source="immortal3")

    # Dos fuentes combinadas (recomendado)
    df = loader.load_and_merge(
        primary_path="data/raw/codeforces_dataset.csv",
        secondary_path="data/raw/codeforces_problemset.csv",
        primary_source="immortal3",
        secondary_source="erchhh",
    )
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Esquema canónico
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: list[str] = [
    "problem_id",
    "name",
    "rating",
    "tags",
    "statement",
    "time_limit_ms",
    "solved_count",
]

# Columnas mínimas para que un registro sea útil en el pipeline.
# rating se excluye aquí porque se extrae de tags en el preprocessor.
REQUIRED_COLUMNS: list[str] = ["problem_id", "tags", "statement"]

# ---------------------------------------------------------------------------
# Mapeos de columnas por fuente conocida
# ---------------------------------------------------------------------------

COLUMN_MAPS: dict[str, dict[str, str]] = {
    # kaggle.com/datasets/immortal3/codeforces-dataset
    # Columnas reales: contest, problem_name, problem_statement, problem_tags
    "immortal3": {
        "contest":           "_contest_id",   # auxiliar para construir problem_id
        "problem_name":      "_index",         # letra del problema (ej. "A")
        "problem_statement": "statement",
        "problem_tags":      "tags",           # "implementation,*1500"
        # name, rating, time_limit_ms, solved_count → pd.NA
    },
    # kaggle.com/datasets/erchhh/codeforces-problemset
    # Columnas reales: problem_statement, input, output, time_limit, memory_limit, tags
    "erchhh": {
        "problem_statement": "statement",      # tiene metadata embebida al inicio
        "time_limit":        "time_limit_ms",  # "4 seconds" → preprocessor convierte
        "tags":              "tags",            # "['dp', 'graphs', '*3000']"
        # input, output, memory_limit → se descartan
    },
    # Fuente genérica: mapeo automático por similitud de nombre
    "generic": {},
}


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


class DatasetLoader:
    """Carga y estandariza datasets de Codeforces desde archivos locales.

    Parameters
    ----------
    extra_column_maps : dict, optional
        Mapeos adicionales para fuentes no contempladas.
    drop_missing_required : bool
        Si True (default), elimina filas con columnas requeridas nulas.
    """

    def __init__(
        self,
        extra_column_maps: Optional[dict[str, dict[str, str]]] = None,
        drop_missing_required: bool = True,
    ) -> None:
        self.column_maps: dict[str, dict[str, str]] = {**COLUMN_MAPS}
        if extra_column_maps:
            self.column_maps.update(extra_column_maps)
        self.drop_missing_required = drop_missing_required

    # ------------------------------------------------------------------
    # Método principal: carga única
    # ------------------------------------------------------------------

    def load(self, path: str | Path, source: str = "generic") -> pd.DataFrame:
        """Carga un archivo CSV o JSON y lo convierte al esquema canónico.

        Parameters
        ----------
        path : str | Path
            Ruta al archivo descargado de Kaggle.
        source : str
            Identificador de la fuente: "immortal3", "erchhh" o "generic".

        Returns
        -------
        pd.DataFrame
            DataFrame con exactamente las columnas de CANONICAL_COLUMNS.
        """
        path = Path(path)
        logger.info(f"Cargando dataset '{source}' desde: {path}")

        raw_df = self._read_file(path)
        logger.info(f"  → {len(raw_df):,} filas cargadas | columnas: {list(raw_df.columns)}")

        df = self._apply_column_map(raw_df, source)
        df = self._build_problem_id(df, source)
        df = self._fill_missing_canonical_columns(df)
        df = self._cast_dtypes(df)
        df = self._deduplicate(df, source)

        if self.drop_missing_required:
            df = self._drop_incomplete_rows(df)

        logger.info(f"  → {len(df):,} problemas válidos tras limpieza básica")
        return df[CANONICAL_COLUMNS]

    # ------------------------------------------------------------------
    # Método principal: carga y merge de dos fuentes
    # ------------------------------------------------------------------

    def load_and_merge(
        self,
        primary_path: str | Path,
        secondary_path: str | Path,
        primary_source: str = "immortal3",
        secondary_source: str = "erchhh",
    ) -> pd.DataFrame:
        """Combina dos datasets usando la fuente primaria como base.

        Estrategia de merge
        -------------------
        1. Se carga cada dataset al esquema canónico por separado.
        2. Se hace outer join por problem_id.
        3. En conflicto, ganan los valores de la fuente primaria.
        4. Campos nulos de la primaria se rellenan desde la secundaria
           (especialmente time_limit_ms de erchhh).

        Returns
        -------
        pd.DataFrame con el esquema canónico.
        """
        logger.info("Iniciando carga y merge de dos fuentes...")
        df_primary = self.load(primary_path, source=primary_source)
        df_secondary = self.load(secondary_path, source=secondary_source)

        merged = self._merge_datasets(df_primary, df_secondary)
        logger.info(f"  → Merge completo: {len(merged):,} problemas únicos")
        return merged

    # ------------------------------------------------------------------
    # Inspección rápida (útil en notebooks)
    # ------------------------------------------------------------------

    def inspect(self, path: str | Path) -> None:
        """Imprime un resumen de columnas y tipos sin aplicar mapeos."""
        path = Path(path)
        raw_df = self._read_file(path)
        print(f"\n{'─'*55}")
        print(f"  Archivo : {path.name}")
        print(f"  Filas   : {len(raw_df):,}")
        print(f"  Columnas: {len(raw_df.columns)}")
        print(f"{'─'*55}")
        info = pd.DataFrame({
            "dtype": raw_df.dtypes,
            "non_null": raw_df.notna().sum(),
            "null_%": (raw_df.isna().mean() * 100).round(1),
            "sample": [
                str(raw_df[c].dropna().iloc[0])[:120] if raw_df[c].notna().any() else "—"
                for c in raw_df.columns
            ],
        })
        print(info.to_string())
        print(f"{'─'*55}\n")

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> pd.DataFrame:
        """Lee CSV o JSON según la extensión."""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix in (".json", ".jsonl"):
            try:
                return pd.read_json(path)
            except ValueError:
                return pd.read_json(path, lines=True)
        raise ValueError(
            f"Formato no soportado: '{suffix}'. "
            "El fetcher acepta archivos .csv, .json o .jsonl."
        )

    def _apply_column_map(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Renombra columnas según el mapa de la fuente."""
        if source not in self.column_maps or source == "generic":
            logger.warning(f"Fuente '{source}' no registrada. Aplicando mapeo automático.")
            return self._auto_map_columns(df)

        col_map = self.column_maps[source]
        rename_map = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        return df

    def _auto_map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mapeo automático por similitud de nombre para fuentes genéricas."""
        alias_map: dict[str, str] = {
            "id": "problem_id", "problemid": "problem_id", "problem_id": "problem_id",
            "name": "name", "title": "name", "problemname": "name",
            "rating": "rating", "difficulty": "rating",
            "tags": "tags", "topics": "tags", "categories": "tags",
            "statement": "statement", "description": "statement",
            "problemstatement": "statement", "body": "statement",
            "timelimit": "time_limit_ms", "time_limit": "time_limit_ms",
            "solvedcount": "solved_count", "solved_count": "solved_count",
        }
        normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
        rename_map = {}
        for norm_name, original_col in normalized.items():
            if norm_name in alias_map:
                rename_map[original_col] = alias_map[norm_name]

        logger.info(f"  Mapeo automático aplicado: {rename_map}")
        return df.rename(columns=rename_map)

    def _build_problem_id(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Construye problem_id desde columnas auxiliares si no existe.

        immortal3: contest (int) + _index (letra) → "325A"
        erchhh:    no tiene identificador → usa índice numérico como fallback.
        """
        if "problem_id" not in df.columns:
            if "_contest_id" in df.columns and "_index" in df.columns:
                df["problem_id"] = (
                    df["_contest_id"].astype(str) + df["_index"].astype(str)
                )
                logger.info("  problem_id construido desde contest + problem_name")
            else:
                # erchhh no tiene identificador único de problema
                df["problem_id"] = [f"{source}_{i}" for i in range(len(df))]
                logger.warning(
                    f"  [{source}] No se encontró problem_id. "
                    "Se generó un identificador sintético basado en índice."
                )

        df = df.drop(columns=["_contest_id", "_index"], errors="ignore")
        return df

    def _fill_missing_canonical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade columnas canónicas ausentes rellenas con pd.NA."""
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def _cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica tipos de datos consistentes al esquema canónico."""
        for col in ("problem_id", "name", "tags", "statement"):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", pd.NA)

        for col in ("rating", "solved_count"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # time_limit_ms: en immortal3 no existe; en erchhh viene como "4 seconds"
        # El preprocessor se encarga de parsear el string a número.
        # Aquí solo intentamos conversión numérica si ya viene como número.
        if "time_limit_ms" in df.columns:
            numeric = pd.to_numeric(df["time_limit_ms"], errors="coerce")
            # Si la mayoría son nulos tras la conversión, probablemente es string → dejar crudo
            if numeric.notna().sum() > 0:
                df["time_limit_ms"] = numeric.astype("Int64")
            # Si son todos nulos (string como "4 seconds"), dejamos el valor crudo
            # para que preprocessor lo parsee correctamente
        return df

    def _deduplicate(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Elimina duplicados por problem_id."""
        before = len(df)
        df = df.drop_duplicates(subset=["problem_id"], keep="first")
        removed = before - len(df)
        if removed > 0:
            logger.info(f"  [{source}] {removed} duplicados eliminados por problem_id")
        return df

    def _drop_incomplete_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descarta filas con valores nulos en columnas requeridas."""
        before = len(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)
        removed = before - len(df)
        if removed > 0:
            logger.info(
                f"  {removed} filas descartadas por columnas requeridas nulas "
                f"({REQUIRED_COLUMNS})"
            )
        return df

    def _merge_datasets(
        self, df_primary: pd.DataFrame, df_secondary: pd.DataFrame
    ) -> pd.DataFrame:
        """Combina dos DataFrames canónicos con prioridad a la fuente primaria."""
        df_sec_new = df_secondary[
            ~df_secondary["problem_id"].isin(df_primary["problem_id"])
        ]

        df_shared_sec = df_secondary[
            df_secondary["problem_id"].isin(df_primary["problem_id"])
        ].set_index("problem_id")

        df_primary = df_primary.set_index("problem_id")

        for col in CANONICAL_COLUMNS:
            if col == "problem_id":
                continue
            if col in df_shared_sec.columns:
                df_primary[col] = df_primary[col].combine_first(df_shared_sec[col])

        df_primary = df_primary.reset_index()
        merged = pd.concat([df_primary, df_sec_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["problem_id"], keep="first")
        return merged[CANONICAL_COLUMNS]