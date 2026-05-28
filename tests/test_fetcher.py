"""
tests/test_fetcher.py

Tests unitarios para src/data/fetcher.py.
Los fixtures imitan exactamente el schema real de cada dataset de Kaggle.

Schema real confirmado:
    immortal3: contest (int), problem_name (str), problem_statement (str), problem_tags (str)
    erchhh:    problem_statement (str), time_limit (str), tags (str), input, output, memory_limit

Ejecutar con:
    pytest tests/test_fetcher.py -v
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.fetcher import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    DatasetLoader,
)

# ---------------------------------------------------------------------------
# Fixtures con schema real de cada dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def immortal3_csv(tmp_path: Path) -> Path:
    """Imita codeforces_dataset.csv (immortal3) con sus columnas reales.

    contest + problem_name → problem_id (ej. 1+"A" → "1A")
    problem_tags incluye el rating embebido como *XXXX.
    Una fila tiene problem_statement nulo → debe descartarse.
    """
    data = pd.DataFrame({
        "contest":           [1,    2,    3,    4],
        "problem_name":      ["A",  "A",  "A",  "A"],
        "problem_statement": [
            "Find the minimum number of tiles to cover a n×m square.",
            "Find the winner of a game given n scores and rules.",
            "Solve a simple equation with two unknowns.",
            None,   # ← nulo → debe descartarse
        ],
        "problem_tags": [
            "math,*1000",
            "implementation,*1200",
            "greedy,math,*800",
            "greedy,*900",
        ],
    })
    path = tmp_path / "immortal3.csv"
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def erchhh_csv(tmp_path: Path) -> Path:
    """Imita codeforces_problemset.csv (erchhh) con sus columnas reales.

    No tiene contest/index → genera IDs sintéticos ("erchhh_N").
    time_limit viene como string "X seconds".
    tags incluye el rating embebido como *XXXX.
    """
    data = pd.DataFrame({
        "problem_statement": [
            "Find the minimum number of tiles for a rectangle floor.",
            "Determine the winner using dynamic programming strategy.",
            "Design a chat server handling concurrent connections.",
            "Calculate the area of intersection of two convex polygons.",
        ],
        "time_limit":   ["1 second", "2 seconds", "1 second", "4 seconds"],
        "memory_limit": ["256 megabytes"] * 4,
        "tags": [
            "['math', '*1000']",
            "['dp', '*1200']",
            "['data structures', '*1500']",
            "['constructive algorithms', 'geometry', '*1600']",
        ],
        "input":  ["1\n2", "2\n1 2", "3\n1 2 3", "4\n1 2 3 4"],
        "output": ["2",    "1",      "3",         "6"],
    })
    path = tmp_path / "erchhh.csv"
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def loader() -> DatasetLoader:
    return DatasetLoader(drop_missing_required=True)


# ---------------------------------------------------------------------------
# Tests de carga — immortal3
# ---------------------------------------------------------------------------

class TestLoadImmortal3:
    def test_returns_canonical_columns(self, loader, immortal3_csv):
        df = loader.load(immortal3_csv, source="immortal3")
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_builds_problem_id_from_contest_and_problem_name(self, loader, immortal3_csv):
        df = loader.load(immortal3_csv, source="immortal3")
        # contest=1, problem_name="A" → "1A"
        assert "1A" in df["problem_id"].values
        assert "2A" in df["problem_id"].values

    def test_drops_row_with_null_statement(self, loader, immortal3_csv):
        df = loader.load(immortal3_csv, source="immortal3")
        assert df["statement"].notna().all()

    def test_correct_row_count_after_drop(self, loader, immortal3_csv):
        df = loader.load(immortal3_csv, source="immortal3")
        # 4 filas - 1 con statement nulo = 3
        assert len(df) == 3

    def test_problem_id_is_string(self, loader, immortal3_csv):
        df = loader.load(immortal3_csv, source="immortal3")
        assert pd.api.types.is_string_dtype(df["problem_id"])

    def test_no_duplicate_problem_ids(self, loader, immortal3_csv):
        df = loader.load(immortal3_csv, source="immortal3")
        assert df["problem_id"].is_unique

    def test_tags_preserved_raw(self, loader, immortal3_csv):
        # El fetcher NO limpia los tags — eso es trabajo del preprocessor
        df = loader.load(immortal3_csv, source="immortal3")
        assert df["tags"].notna().all()

    def test_optional_columns_are_na(self, loader, immortal3_csv):
        # immortal3 no tiene time_limit_ms ni solved_count
        df = loader.load(immortal3_csv, source="immortal3")
        assert df["time_limit_ms"].isna().all()
        assert df["solved_count"].isna().all()

    def test_rating_column_is_na(self, loader, immortal3_csv):
        # immortal3 no tiene rating como columna separada
        # el rating está en tags y lo extrae el preprocessor
        df = loader.load(immortal3_csv, source="immortal3")
        assert df["rating"].isna().all()

    def test_keep_all_rows_when_flag_false(self, immortal3_csv):
        loader = DatasetLoader(drop_missing_required=False)
        df = loader.load(immortal3_csv, source="immortal3")
        assert len(df) == 4


# ---------------------------------------------------------------------------
# Tests de carga — erchhh
# ---------------------------------------------------------------------------

class TestLoadErchhh:
    def test_returns_canonical_columns(self, loader, erchhh_csv):
        df = loader.load(erchhh_csv, source="erchhh")
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_generates_synthetic_problem_id(self, loader, erchhh_csv):
        # erchhh no tiene contest/index → IDs sintéticos tipo "erchhh_N"
        df = loader.load(erchhh_csv, source="erchhh")
        assert all(pid.startswith("erchhh_") for pid in df["problem_id"])

    def test_no_duplicate_problem_ids(self, loader, erchhh_csv):
        df = loader.load(erchhh_csv, source="erchhh")
        assert df["problem_id"].is_unique

    def test_time_limit_ms_is_populated(self, loader, erchhh_csv):
        # time_limit ("1 second") se mapea a time_limit_ms como string crudo
        # el preprocessor lo convertirá a minutos más tarde
        df = loader.load(erchhh_csv, source="erchhh")
        assert df["time_limit_ms"].notna().any()

    def test_all_rows_loaded(self, loader, erchhh_csv):
        df = loader.load(erchhh_csv, source="erchhh")
        assert len(df) == 4

    def test_tags_preserved(self, loader, erchhh_csv):
        df = loader.load(erchhh_csv, source="erchhh")
        assert df["tags"].notna().all()


# ---------------------------------------------------------------------------
# Tests de merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_returns_canonical_columns(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        assert list(df.columns) == CANONICAL_COLUMNS

    def test_merge_no_duplicate_problem_ids(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        assert df["problem_id"].is_unique

    def test_merge_contains_immortal3_problems(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        assert "1A" in df["problem_id"].values

    def test_merge_contains_erchhh_problems(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        # Los de erchhh tienen IDs sintéticos
        assert any(pid.startswith("erchhh_") for pid in df["problem_id"])

    def test_merge_total_count(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        # 3 de immortal3 (1 descartado por null) + 4 de erchhh = 7
        assert len(df) == 7

    def test_erchhh_rows_have_time_limit(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        erchhh_rows = df[df["problem_id"].str.startswith("erchhh_")]
        assert erchhh_rows["time_limit_ms"].notna().all()

    def test_immortal3_rows_have_no_time_limit(self, loader, immortal3_csv, erchhh_csv):
        df = loader.load_and_merge(immortal3_csv, erchhh_csv)
        immortal3_rows = df[~df["problem_id"].str.startswith("erchhh_")]
        assert immortal3_rows["time_limit_ms"].isna().all()


# ---------------------------------------------------------------------------
# Tests de fuente genérica
# ---------------------------------------------------------------------------

class TestGenericSource:
    def test_auto_map_common_aliases(self, tmp_path):
        data = pd.DataFrame({
            "problem_id":  ["10A"],
            "title":       ["A generic problem"],
            "description": ["Solve this problem with n constraints given."],
            "difficulty":  [1400],
            "topics":      ["dp,graphs"],
        })
        path = tmp_path / "generic.csv"
        data.to_csv(path, index=False)

        loader = DatasetLoader(drop_missing_required=True)
        df = loader.load(path, source="generic")

        assert "name" in df.columns
        assert "statement" in df.columns
        assert "rating" in df.columns
        assert "tags" in df.columns


# ---------------------------------------------------------------------------
# Tests de robustez
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_json_file_loads_correctly(self, tmp_path):
        # JSON con schema real de immortal3
        data = [
            {
                "contest": 100,
                "problem_name": "A",
                "problem_statement": "Given n integers find the maximum subarray sum.",
                "problem_tags": "dp,*1800",
            }
        ]
        path = tmp_path / "problems.json"
        path.write_text(json.dumps(data))

        loader = DatasetLoader()
        df = loader.load(path, source="immortal3")
        assert len(df) == 1
        assert df["problem_id"].iloc[0] == "100A"

    def test_unsupported_format_raises_value_error(self, tmp_path):
        path = tmp_path / "problems.xlsx"
        path.write_text("dummy")
        loader = DatasetLoader()
        with pytest.raises(ValueError, match="Formato no soportado"):
            loader.load(path, source="immortal3")

    def test_required_columns_are_correct(self):
        assert "problem_id" in REQUIRED_COLUMNS
        assert "tags"       in REQUIRED_COLUMNS
        assert "statement"  in REQUIRED_COLUMNS
        # rating NO es requerido en el fetcher — lo extrae el preprocessor
        assert "rating" not in REQUIRED_COLUMNS
        assert "name"   not in REQUIRED_COLUMNS