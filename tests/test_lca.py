from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from scipy import sparse as sps

from pathways.lca import (
    _build_column_aggregation_matrix,
    _create_multilca,
    _effective_distribution_count,
    _get_multilca_class,
    load_matrix_and_index,
    read_indices_csv,
)
from pathways.jacobi_gmres_multi_lca import JacobiGMRESMultiLCA
from pathways.utils import create_lca_results_array


def test_read_indices_csv_success():
    mock_csv_data = "activity;product;location;unit;1\nanother_activity;another_product;another_location;another_unit;2"
    expected_dict = {
        ("activity", "product", "location", "unit"): 1,
        ("another_activity", "another_product", "another_location", "another_unit"): 2,
    }
    with patch("builtins.open", mock_open(read_data=mock_csv_data)):
        result = read_indices_csv(Path("dummy_path.csv"))
        assert result == expected_dict


def test_read_indices_csv_with_header():
    mock_csv_data = (
        "activity;product;location;unit;index\n"
        "activity;product;location;unit;1\n"
        "another_activity;another_product;another_location;another_unit;2"
    )
    expected_dict = {
        ("activity", "product", "location", "unit"): 1,
        ("another_activity", "another_product", "another_location", "another_unit"): 2,
    }
    with patch("builtins.open", mock_open(read_data=mock_csv_data)):
        result = read_indices_csv(Path("dummy_path.csv"))
        assert result == expected_dict


def test_load_matrix_and_index(tmp_path):
    mock_csv_data = (
        "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip"
        "\n1;0;3.5;3;4;5;6;7;8;0;0"
        "\n1;1;0.5;3;4;5;6;7;8;0;1"
    )
    expected_output = (
        np.array([3.5, 0.5]),
        np.array([(0, 1), (1, 1)], dtype=[("row", "i4"), ("col", "i4")]),
        np.array([False, True]),
        np.array(
            [(3, 4.0, 5.0, 6.0, 7.0, 8.0, False), (3, 4.0, 5.0, 6.0, 7.0, 8.0, False)],
            dtype=[
                ("uncertainty_type", "i4"),
                ("loc", "f4"),
                ("scale", "f4"),
                ("shape", "f4"),
                ("minimum", "f4"),
                ("maximum", "f4"),
                ("negative", "?"),
            ],
        ),
    )

    # Write mock CSV data to a temporary file
    temp_file = tmp_path / "temp.csv"
    temp_file.write_text(mock_csv_data)

    # Call the function with the path to the temporary file
    data_array, indices_array, flip_array, distributions_array = load_matrix_and_index(
        temp_file
    )

    # Check that the output matches the expected output
    # but they have different dtypes

    assert np.allclose(data_array, expected_output[0])
    assert np.array_equal(indices_array, expected_output[1])
    assert np.array_equal(flip_array, expected_output[2])
    assert np.array_equal(distributions_array, expected_output[3])


def test_load_matrix_and_index_single_row(tmp_path):
    csv_data = (
        "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip\n"
        "1;0;3.5;3;4;5;6;7;8;0;1"
    )
    temp_file = tmp_path / "single.csv"
    temp_file.write_text(csv_data)

    data_array, indices_array, flip_array, distributions_array = load_matrix_and_index(
        temp_file
    )

    assert data_array.shape == (1,)
    assert tuple(indices_array[0]) == (0, 1)
    assert bool(flip_array[0]) is True
    assert distributions_array.shape == (1,)


def test_load_matrix_and_index_reuses_persistent_cache(tmp_path, monkeypatch):
    csv_data = (
        "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip\n"
        "1;0;3.5;3;4;5;6;7;8;0;1\n"
        "1;1;0.5;3;4;5;6;7;8;0;0"
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    temp_file = tmp_path / "cached.csv"
    temp_file.write_text(csv_data)

    monkeypatch.setattr("pathways.lca.DIR_CACHED_DB", cache_dir)

    first = load_matrix_and_index(temp_file)
    cache_files = list((cache_dir / "matrix_arrays").glob("*.npz"))
    assert len(cache_files) == 1

    def fail_genfromtxt(*args, **kwargs):
        raise AssertionError("genfromtxt should not be called on a warm cache load")

    monkeypatch.setattr("pathways.lca.np.genfromtxt", fail_genfromtxt)
    second = load_matrix_and_index(temp_file)

    for left, right in zip(first, second):
        assert np.array_equal(left, right)


def test_load_matrix_and_index_cache_survives_datapackage_reextract(
    tmp_path, monkeypatch
):
    csv_data = (
        "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip\n"
        "1;0;3.5;3;4;5;6;7;8;0;1\n"
        "1;1;0.5;3;4;5;6;7;8;0;0"
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    first_file = (
        tmp_path
        / "extract_a"
        / "inventories"
        / "model"
        / "scenario"
        / "2025"
        / "A_matrix.csv"
    )
    first_file.parent.mkdir(parents=True)
    first_file.write_text(csv_data)
    second_file = (
        tmp_path
        / "extract_b"
        / "inventories"
        / "model"
        / "scenario"
        / "2025"
        / "A_matrix.csv"
    )
    second_file.parent.mkdir(parents=True)
    second_file.write_text(csv_data)

    monkeypatch.setattr("pathways.lca.DIR_CACHED_DB", cache_dir)

    load_matrix_and_index(first_file)

    def fail_genfromtxt(*args, **kwargs):
        raise AssertionError(
            "genfromtxt should not be called after datapackage re-extraction"
        )

    monkeypatch.setattr("pathways.lca.np.genfromtxt", fail_genfromtxt)
    second = load_matrix_and_index(second_file)

    assert second[0].tolist() == [3.5, 0.5]


def test_load_matrix_and_index_bad_columns(tmp_path):
    csv_data = "row;col;value\n1;0;3.5"
    temp_file = tmp_path / "bad_cols.csv"
    temp_file.write_text(csv_data)

    with pytest.raises(ValueError, match="at least 11"):
        load_matrix_and_index(temp_file)


def test_effective_distribution_count_only_collapses_without_uncertainty():
    varying_shares = {
        "PV": {
            "EUR": {
                2050: {
                    "c-Si": np.array([0.6, 0.5, 0.4]),
                    "CdTe": np.array([0.4, 0.5, 0.6]),
                }
            }
        }
    }
    constant_shares = {
        "PV": {
            "EUR": {
                2020: {
                    "c-Si": np.array([0.9, 0.9, 0.9]),
                    "CdTe": np.array([0.1, 0.1, 0.1]),
                }
            }
        }
    }

    assert _effective_distribution_count(100, [], constant_shares, 2005) == 1
    assert _effective_distribution_count(100, [], constant_shares, 2020) == 1
    assert _effective_distribution_count(100, [], varying_shares, 2050) == 100
    assert _effective_distribution_count(100, [(1, 2)], constant_shares, 2020) == 100
    assert _effective_distribution_count(0, [], varying_shares, 2050) == 0


def test_build_column_aggregation_matrix():
    acts_category_idx_dict = {
        "cat_a": [0, 2],
        "cat_b": [1, 3],
    }
    acts_location_idx_dict = {
        "loc_x": [0, 1],
        "loc_y": [2, 3],
    }

    matrix = _build_column_aggregation_matrix(
        acts_category_idx_dict=acts_category_idx_dict,
        acts_location_idx_dict=acts_location_idx_dict,
        n_cols=4,
    )

    dense = matrix.toarray()
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # cat_a, loc_x
            [0.0, 0.0, 1.0, 0.0],  # cat_b, loc_x
            [0.0, 1.0, 0.0, 0.0],  # cat_a, loc_y
            [0.0, 0.0, 0.0, 1.0],  # cat_b, loc_y
        ]
    )

    assert matrix.shape == (4, 4)
    assert np.array_equal(dense, expected)


def test_create_lca_results_array_collapses_requested_dimensions():
    array = create_lca_results_array(
        methods=["method_a"],
        years=[2020],
        regions=["EUR"],
        locations=["CH", "DE"],
        models=["remind"],
        scenarios=["scenario_a"],
        classifications={
            ("activity a", "product a"): "cat_a",
            ("activity b", "product b"): "cat_b",
        },
        mapping={"var_a": {}},
        aggregate_by=["act_category", "location"],
        use_distributions=True,
    )

    assert array.sizes["act_category"] == 1
    assert array.sizes["location"] == 1
    assert array.coords["act_category"].values.tolist() == ["aggregated"]
    assert array.coords["location"].values.tolist() == ["aggregated"]


def test_get_multilca_class():
    cls = _get_multilca_class("jacobi-gmres")
    assert cls is JacobiGMRESMultiLCA


def test_create_multilca_forwards_iterative_kwargs(monkeypatch):
    captured = {}

    class DummyIterative:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("pathways.lca.JacobiGMRESMultiLCA", DummyIterative)

    _create_multilca(
        demands={"foo": {1: 1.0}},
        data_objs=["dummy"],
        use_distributions=True,
        seed=42,
        solver="jacobi-gmres",
        iterative_rtol=1e-4,
        iterative_atol=1e-6,
        iterative_restart=25,
        iterative_maxiter=150,
        iterative_use_guess=False,
        use_arrays=True,
    )

    assert captured["demands"] == {"foo": {1: 1.0}}
    assert captured["data_objs"] == ["dummy"]
    assert captured["use_distributions"] is True
    assert captured["seed_override"] == 42
    assert captured["use_arrays"] is True
    assert captured["rtol"] == 1e-4
    assert captured["atol"] == 1e-6
    assert captured["restart"] == 25
    assert captured["maxiter"] == 150
    assert captured["use_guess"] is False


def test_jacobi_gmres_multilca_solves_multiple_demands():
    lca = object.__new__(JacobiGMRESMultiLCA)
    lca.rtol = 1e-10
    lca.atol = 0.0
    lca.restart = 20
    lca.maxiter = 100
    lca.use_guess = True
    lca.direct_fallback = True
    lca._matrix_prepared = False
    lca._cached_preconditioner = None
    lca.guesses = {}
    lca.demands = {"a": {0: 1.0}, "b": {1: 1.0}}
    lca.demand_arrays = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
    }
    lca.dicts = SimpleNamespace(activity={0: 0, 1: 1})
    lca.technosphere_matrix = sps.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    lca.biosphere_matrix = sps.identity(2, format="csc")

    lca.lci_calculation()

    expected_a = np.linalg.solve(lca.technosphere_matrix.toarray(), np.array([1.0, 0.0]))
    expected_b = np.linalg.solve(lca.technosphere_matrix.toarray(), np.array([0.0, 1.0]))

    assert np.allclose(lca.supply_arrays["a"], expected_a)
    assert np.allclose(lca.supply_arrays["b"], expected_b)
    assert set(lca.inventories) == {"a", "b"}


def test_jacobi_gmres_multilca_falls_back_to_direct(monkeypatch):
    lca = object.__new__(JacobiGMRESMultiLCA)
    lca.rtol = 1e-4
    lca.atol = 0.0
    lca.restart = 10
    lca.maxiter = 10
    lca.use_guess = False
    lca.direct_fallback = True
    lca._matrix_prepared = False
    lca._cached_preconditioner = None
    lca.guesses = {}
    lca.technosphere_matrix = sps.identity(2, format="csc")

    monkeypatch.setattr(
        "pathways.jacobi_gmres_multi_lca.gmres",
        lambda *args, **kwargs: (np.zeros(2), 1),
    )
    monkeypatch.setattr(
        "pathways.jacobi_gmres_multi_lca.bc.spsolve",
        lambda matrix, demand: np.array([7.0, 8.0]),
    )

    solution = lca._solve_with_gmres(np.array([1.0, 2.0]), demand_name="a")

    assert np.array_equal(solution, np.array([7.0, 8.0]))
