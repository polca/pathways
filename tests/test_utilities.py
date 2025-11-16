from unittest.mock import mock_open, patch

import numpy as np
import pytest
import xarray as xr

from pathways.utils import (
    clean_cache_directory,
    create_lca_results_array,
    harmonize_units,
    load_classifications,
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
    (cache_dir / "temp_cache_file").write_text("This is a cache file.")
    non_cache_dir = tmp_path / "non_cache"
    non_cache_dir.mkdir()
    (non_cache_dir / "temp_non_cache_file").write_text("This should remain.")

    # Use monkeypatch to set DIR_CACHED_DB for the duration of the test
    monkeypatch.setattr("pathways.utils.DIR_CACHED_DB", str(cache_dir))

    clean_cache_directory()

    assert not (cache_dir / "temp_cache_file").exists(), "Cache file was not deleted"
    assert (
        non_cache_dir / "temp_non_cache_file"
    ).exists(), "Non-cache file was incorrectly deleted"
