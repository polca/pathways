from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pathways.utils import (
    apply_filters,
    clean_cache_directory,
    create_lca_results_array,
    export_results_to_parquet,
    harmonize_units,
    load_classifications,
    load_mapping,
    load_units_conversion,
    load_numpy_array_from_disk,
)


def test_load_classifications_success(tmp_path, monkeypatch):
    # Create a temporary CSV file that mimics the new format
    csv_content = (
        "name,reference product,ISIC rev.4 ecoinvent,CPC\n"
        "activity1,product1,1111:Foo,2222:Bar\n"
        "activity2,product2,3333:Baz,4444:Qux\n"
    )
    csv_path = tmp_path / "classifications.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    # Point CLASSIFICATIONS to this temporary CSV
    monkeypatch.setattr("pathways.utils.CLASSIFICATIONS", str(csv_path))

    classifications = load_classifications()

    assert classifications == {
        ("activity1", "product1"): [
            ("ISIC rev.4 ecoinvent", "1111:Foo"),
            ("CPC", "2222:Bar"),
        ],
        ("activity2", "product2"): [
            ("ISIC rev.4 ecoinvent", "3333:Baz"),
            ("CPC", "4444:Qux"),
        ],
    }


def test_load_classifications_file_not_found():
    with patch("pathways.utils.CLASSIFICATIONS", new="non_existent_file.yaml"):
        with pytest.raises(FileNotFoundError):
            load_classifications()


def test_harmonize_units_conversion_required():
    scenario = xr.DataArray(
        np.random.rand(2, 2, 2),
        dims=["variables", "x", "y"],
        coords={"variables": ["var1", "var2"]},
    )
    scenario.attrs["units"] = {"var1": "PJ/yr", "var2": "EJ/yr"}
    variables = ["var1", "var2"]

    harmonized_scenario = harmonize_units(scenario, variables)
    assert all(
        harmonized_scenario.attrs["units"][var] == "EJ/yr" for var in variables
    ), "Units not harmonized to EJ/yr"


def test_harmonize_units_no_conversion_required():
    scenario = xr.DataArray(
        np.random.rand(1, 2, 2),
        dims=["variables", "x", "y"],
        coords={"variables": ["var1"]},
    )
    scenario.attrs["units"] = {"var1": "EJ/yr"}
    variables = ["var1"]

    harmonized_scenario = harmonize_units(scenario, variables)
    assert harmonized_scenario.equals(scenario), "Scenario was modified unnecessarily"


def test_harmonize_units_missing_units_attribute():
    scenario = xr.DataArray(
        np.random.rand(1, 2, 2),
        dims=["variables", "x", "y"],
        coords={"variables": ["var1"]},
    )
    variables = ["var1"]

    with pytest.raises(KeyError):
        harmonize_units(scenario, variables)


def test_harmonize_units_empty_data_array():
    scenario = xr.DataArray(
        [[[1]], [[2]], [[3]]],
        dims=["variables", "x", "y"],
        coords={"variables": ["var1", "var2", "var3"]},
    )
    scenario.attrs["units"] = {}
    variables = []

    # should return ValueError
    with pytest.raises(ValueError):
        harmonize_units(scenario, variables)


def test_create_lca_results_array_structure_and_initialization():
    methods = ["method1", "method2"]
    years = [2020, 2025]
    regions = ["region1", "region2"]
    locations = ["location1", "location2"]
    models = ["model1", "model2"]
    scenarios = ["scenario1", "scenario2"]
    classifications = {"activity1": "category1", "activity2": "category2"}
    mapping = {"variable1": "dataset1", "variable2": "dataset2"}

    result = create_lca_results_array(
        methods, years, regions, locations, models, scenarios, classifications, mapping
    )

    # Check dimensions and coordinates
    assert "act_category" in result.coords
    assert "impact_category" in result.coords
    assert "year" in result.coords
    assert "region" in result.coords
    assert "model" in result.coords
    assert "scenario" in result.coords
    assert set(result.coords["impact_category"].values) == set(methods)
    assert set(result.coords["year"].values) == set(years)
    assert set(result.coords["region"].values) == set(regions)
    assert np.all(result == 0), "DataArray should be initialized with zeros"


def test_create_lca_results_array_with_distributions():
    methods = ["method1"]
    years = [2020]
    regions = ["region1"]
    locations = ["location1"]
    models = ["model1"]
    scenarios = ["scenario1"]
    classifications = {"activity1": "category1"}
    mapping = {"variable1": "dataset1"}

    result = create_lca_results_array(
        methods,
        years,
        regions,
        locations,
        models,
        scenarios,
        classifications,
        mapping,
        use_distributions=True,
    )

    # Check for the 'quantile' dimension
    assert "quantile" in result.dims
    assert result.coords["quantile"].values.tolist() == [0.05, 0.5, 0.95]


def test_create_lca_results_array_empty_inputs():
    with pytest.raises(
        Exception
    ):  # Assuming the function raises an exception for empty inputs
        create_lca_results_array([], [], [], [], [], [], {}, {})


def test_create_lca_results_array_input_validation():
    with pytest.raises(Exception):
        create_lca_results_array(None, None, None, None, None, None, None, None)


def test_clean_cache_directory(tmp_path, monkeypatch):
    # Use a temporary directory to simulate the cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "temp_cache_file.npy").write_text("This is a cache file.")
    non_cache_dir = tmp_path / "non_cache"
    non_cache_dir.mkdir()
    (non_cache_dir / "temp_non_cache_file").write_text("This should remain.")

    # Use monkeypatch to set DIR_CACHED_DB for the duration of the test
    monkeypatch.setattr("pathways.utils.DIR_CACHED_DB", str(cache_dir))
    monkeypatch.setattr("pathways.utils.USER_DATA_BASE_DIR", tmp_path)

    clean_cache_directory()

    assert not (cache_dir / "temp_cache_file.npy").exists(), "Cache file was not deleted"
    assert (
        non_cache_dir / "temp_non_cache_file"
    ).exists(), "Non-cache file was incorrectly deleted"


def test_clean_cache_directory_rejects_unsafe_path(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache-outside"
    cache_dir.mkdir()
    monkeypatch.setattr("pathways.utils.DIR_CACHED_DB", str(cache_dir))
    monkeypatch.setattr("pathways.utils.USER_DATA_BASE_DIR", tmp_path / "base")

    with pytest.raises(ValueError, match="Refusing to clean cache directory"):
        clean_cache_directory()


def test_apply_filters_path_matching_is_not_character_based():
    technosphere = {("alpha plant", "prod", "EU", "kg"): 1}
    filters = {"name_fltr": [], "name_mask": [], "product_fltr": [], "product_mask": []}
    exceptions = {"name_fltr": [], "name_mask": [], "product_fltr": [], "product_mask": []}
    paths = [["ab"]]

    _, _, filtered_names, _ = apply_filters(technosphere, filters, exceptions, paths)
    assert filtered_names[("ab",)] == set()


def test_load_mapping_requires_dict_yaml(tmp_path):
    mapping_file = tmp_path / "mapping.yaml"
    mapping_file.write_text("- a\n- b\n")

    with pytest.raises(ValueError, match="expected a YAML dictionary"):
        load_mapping(str(mapping_file))


def test_load_units_conversion_requires_dict_yaml(tmp_path, monkeypatch):
    bad_units = tmp_path / "units.yaml"
    bad_units.write_text("- unit\n")
    monkeypatch.setattr("pathways.utils.UNITS_CONVERSION", bad_units)

    with pytest.raises(ValueError, match="expected a YAML dictionary"):
        load_units_conversion()


def test_load_numpy_array_from_disk_disallows_pickle(tmp_path):
    arr = np.array([{"x": 1}], dtype=object)
    fp = tmp_path / "obj.npy"
    np.save(fp, arr, allow_pickle=True)

    with pytest.raises(ValueError):
        load_numpy_array_from_disk(fp)


def test_export_results_to_parquet_does_not_write_index_column(tmp_path):
    data = xr.DataArray(
        np.array([[1.0, 0.0], [0.0, 2.0]]),
        dims=["row_dim", "col_dim"],
        coords={"row_dim": ["r1", "r2"], "col_dim": ["c1", "c2"]},
    )
    out_base = tmp_path / "results"
    out_path = export_results_to_parquet(data, str(out_base))

    df = pd.read_parquet(out_path)
    assert "index" not in df.columns
    assert "__index_level_0__" not in df.columns
