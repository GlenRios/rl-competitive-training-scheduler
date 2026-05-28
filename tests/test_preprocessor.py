"""
tests/test_preprocessor.py

Tests unitarios para src/data/preprocessor.py.
Usan datasets sintéticos en memoria para no depender de Kaggle.

Ejecutar con:
    pytest tests/test_preprocessor.py -v
"""

import pandas as pd
import pytest

from src.data.preprocessor import Preprocessor, DIFFICULTY_BINS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_raw_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Genera un DataFrame sintético que imita la salida del fetcher."""
    import random
    random.seed(seed)

    ratings     = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
                   1600, 1700, 1800, 1900, 2000, 2200, 2400, 2800]
    topics      = ["dp", "graphs", "greedy", "math", "implementation",
                   "trees", "binary search", "strings", "geometry", "number theory"]
    tag_formats = ["immortal3", "erchhh"]

    rows = []
    for i in range(n):
        rating  = random.choice(ratings)
        topic1  = random.choice(topics)
        topic2  = random.choice(topics)
        fmt     = random.choice(tag_formats)

        if fmt == "immortal3":
            tags = f"{topic1},{topic2},*{rating}"
        else:
            tags = f"['{topic1}', '{topic2}', '*{rating}']"

        rows.append({
            "problem_id":    f"{300 + i}A",
            "name":          pd.NA,
            "rating":        pd.NA,        # sin rating explícito — está en tags
            "tags":          tags,
            "statement":     f"This is a problem about {topic1} and {topic2}. " * 8,
            "time_limit_ms": pd.NA,
            "solved_count":  pd.NA,
        })

    return pd.DataFrame(rows)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw_df(n=300, seed=42)


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor(n_problems=100, random_seed=42)


# ---------------------------------------------------------------------------
# Tests de extracción de rating
# ---------------------------------------------------------------------------

class TestExtractRating:
    def test_extracts_rating_from_immortal3_format(self):
        df = pd.DataFrame({
            "problem_id": ["1A"], "name": [pd.NA], "rating": [pd.NA],
            "tags": ["implementation,greedy,*1500"],
            "statement": ["A" * 100], "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._extract_rating(df)
        assert result["rating"].iloc[0] == 1500

    def test_extracts_rating_from_erchhh_format(self):
        df = pd.DataFrame({
            "problem_id": ["2A"], "name": [pd.NA], "rating": [pd.NA],
            "tags": ["['dp', 'graphs', '*3000']"],
            "statement": ["B" * 100], "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._extract_rating(df)
        assert result["rating"].iloc[0] == 3000

    def test_rating_is_na_when_no_token(self):
        df = pd.DataFrame({
            "problem_id": ["3A"], "name": [pd.NA], "rating": [pd.NA],
            "tags": ["implementation,greedy"],
            "statement": ["C" * 100], "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._extract_rating(df)
        assert pd.isna(result["rating"].iloc[0])

    def test_all_problems_get_rating_extracted(self, raw_df):
        pre = Preprocessor(n_problems=50)
        result = pre._extract_rating(raw_df.copy())
        # Todos los sintéticos tienen *XXXX en los tags → todos deben tener rating
        assert result["rating"].notna().all()


# ---------------------------------------------------------------------------
# Tests de limpieza de tags
# ---------------------------------------------------------------------------

class TestCleanTags:
    def test_removes_rating_token_from_tags(self):
        df = pd.DataFrame({
            "problem_id": ["1A"], "name": [pd.NA], "rating": [1500],
            "tags": ["implementation,greedy,*1500"],
            "statement": ["A" * 100], "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._clean_tags(df)
        assert "*1500" not in result["tags"].iloc[0]
        assert "*1500" not in result["tags_list"].iloc[0]

    def test_tags_list_is_python_list(self, raw_df):
        pre = Preprocessor(n_problems=50)
        result = pre._clean_tags(raw_df.copy())
        assert all(isinstance(t, list) for t in result["tags_list"])

    def test_handles_both_tag_formats(self):
        df = pd.DataFrame({
            "problem_id": ["1A", "2A"],
            "name": [pd.NA, pd.NA],
            "rating": [1500, 2000],
            "tags": ["implementation,*1500", "['dp', 'graphs', '*2000']"],
            "statement": ["A" * 100, "B" * 100],
            "time_limit_ms": [pd.NA, pd.NA],
            "solved_count": [pd.NA, pd.NA],
        })
        pre = Preprocessor(n_problems=2)
        result = pre._clean_tags(df)
        assert result["tags_list"].iloc[0] == ["implementation"]
        assert result["tags_list"].iloc[1] == ["dp", "graphs"]

    def test_tags_are_lowercase(self, raw_df):
        pre = Preprocessor(n_problems=50)
        result = pre._clean_tags(raw_df.copy())
        for tag_list in result["tags_list"]:
            for tag in tag_list:
                assert tag == tag.lower()


# ---------------------------------------------------------------------------
# Tests de limpieza de statement
# ---------------------------------------------------------------------------

class TestCleanStatement:
    def test_truncates_to_2000_chars(self):
        long_text = "word " * 1000   # 5000 chars
        df = pd.DataFrame({
            "problem_id": ["1A"], "name": [pd.NA], "rating": [1200],
            "tags": ["math"], "statement": [long_text],
            "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._clean_statement(df)
        assert len(result["statement"].iloc[0]) <= 2000

    def test_removes_excess_whitespace(self):
        df = pd.DataFrame({
            "problem_id": ["1A"], "name": [pd.NA], "rating": [1200],
            "tags": ["math"],
            "statement": ["Hello    world   here  is   a problem " * 5],
            "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._clean_statement(df)
        assert "  " not in result["statement"].iloc[0]

    def test_removes_latex_inline(self):
        df = pd.DataFrame({
            "problem_id": ["1A"], "name": [pd.NA], "rating": [1200],
            "tags": ["math"],
            "statement": ["You are given $n \\leq 100$ integers and $m$ queries. " * 4],
            "time_limit_ms": [pd.NA], "solved_count": [pd.NA],
        })
        pre = Preprocessor(n_problems=1)
        result = pre._clean_statement(df)
        assert "$" not in result["statement"].iloc[0]


# ---------------------------------------------------------------------------
# Tests de filtrado
# ---------------------------------------------------------------------------

class TestFilterValid:
    def test_removes_rows_without_rating(self, raw_df):
        raw_df = raw_df.copy()
        raw_df.loc[:5, "rating"] = pd.NA   # forzar nulos
        pre = Preprocessor(n_problems=50)
        df = pre._extract_rating(raw_df)
        # Ahora esos 6 tienen rating extraído de tags, no deben filtrarse
        result = pre._filter_valid(df)
        assert result["rating"].notna().all()

    def test_removes_rows_with_short_statement(self):
        df = pd.DataFrame({
            "problem_id": ["1A", "2A"],
            "name": [pd.NA, pd.NA],
            "rating": [1200, 1400],
            "tags": ["math", "dp"],
            "statement": ["short", "A proper problem statement " * 5],
            "time_limit_ms": [pd.NA, pd.NA],
            "solved_count": [pd.NA, pd.NA],
            "tags_list": [["math"], ["dp"]],
            "difficulty_band": ["1200-1599", "1200-1599"],
        })
        pre = Preprocessor(n_problems=2, min_statement_len=80)
        result = pre._filter_valid(df)
        assert len(result) == 1
        assert result["problem_id"].iloc[0] == "2A"

    def test_removes_out_of_range_ratings(self):
        df = pd.DataFrame({
            "problem_id": ["1A", "2A", "3A"],
            "name": [pd.NA, pd.NA, pd.NA],
            "rating": [500, 1200, 4000],   # 500 y 4000 fuera de rango
            "tags": ["math", "dp", "graphs"],
            "statement": ["Valid statement here. " * 5] * 3,
            "time_limit_ms": [pd.NA, pd.NA, pd.NA],
            "solved_count": [pd.NA, pd.NA, pd.NA],
            "tags_list": [["math"], ["dp"], ["graphs"]],
            "difficulty_band": ["unknown", "1200-1599", "unknown"],
        })
        pre = Preprocessor(n_problems=3)
        result = pre._filter_valid(df)
        assert len(result) == 1
        assert result["rating"].iloc[0] == 1200


# ---------------------------------------------------------------------------
# Tests del muestreo estratificado
# ---------------------------------------------------------------------------

class TestStratifiedSample:
    def test_returns_correct_number_of_problems(self, raw_df):
        pre = Preprocessor(n_problems=100, random_seed=42)
        result = pre.run(raw_df)
        assert len(result) == 100

    def test_no_duplicate_problem_ids(self, raw_df):
        pre = Preprocessor(n_problems=100, random_seed=42)
        result = pre.run(raw_df)
        assert result["problem_id"].is_unique

    def test_all_ratings_in_valid_range(self, raw_df):
        pre = Preprocessor(n_problems=100, random_seed=42)
        result = pre.run(raw_df)
        assert (result["rating"] >= 800).all()
        assert (result["rating"] <= 3500).all()

    def test_multiple_difficulty_bands_represented(self, raw_df):
        pre = Preprocessor(n_problems=100, random_seed=42)
        result = pre.run(raw_df)
        bands_present = result["difficulty_band"].nunique()
        assert bands_present >= 3  # al menos 3 bandas distintas

    def test_uses_all_problems_when_dataset_smaller_than_n(self):
        small_df = make_raw_df(n=30, seed=1)
        pre = Preprocessor(n_problems=100, random_seed=42)
        result = pre.run(small_df)
        # Con menos problemas que el cupo, devuelve todos los válidos
        assert len(result) <= 30

    def test_tags_list_column_exists(self, raw_df):
        pre = Preprocessor(n_problems=100)
        result = pre.run(raw_df)
        assert "tags_list" in result.columns

    def test_difficulty_band_column_exists(self, raw_df):
        pre = Preprocessor(n_problems=100)
        result = pre.run(raw_df)
        assert "difficulty_band" in result.columns

    def test_reproducible_with_same_seed(self, raw_df):
        pre1 = Preprocessor(n_problems=100, random_seed=7)
        pre2 = Preprocessor(n_problems=100, random_seed=7)
        r1 = pre1.run(raw_df.copy())
        r2 = pre2.run(raw_df.copy())
        assert list(r1["problem_id"]) == list(r2["problem_id"])

    def test_different_seeds_give_different_results(self, raw_df):
        pre1 = Preprocessor(n_problems=100, random_seed=1)
        pre2 = Preprocessor(n_problems=100, random_seed=99)
        r1 = pre1.run(raw_df.copy())
        r2 = pre2.run(raw_df.copy())
        # Con semillas distintas el orden (al menos) debe diferir
        assert list(r1["problem_id"]) != list(r2["problem_id"])