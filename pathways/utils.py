import yaml
import numpy as np
import xarray as xr
from typing import Union, List

from . import DATA_DIR

CLASSIFICATIONS = DATA_DIR / "activities_classifications.yaml"
UNITS_CONVERSION = DATA_DIR / "units_conversion.yaml"

def load_classifications():
    """Load the activities classifications."""

    with open(CLASSIFICATIONS, "r") as f:
        data = yaml.full_load(f)

    # ensure that "NO" is not interpreted as False
    new_keys = []
    old_keys = []
    for key, value in data.items():
        # check if last element of key is not nan
        if not isinstance(key[-1], str):
            new_entry = key[:-1] + ("NO",)
            new_keys.append(new_entry)
            old_keys.append(key)

    for new_key, old_key in zip(new_keys, old_keys):
        data[new_key] = data.pop(old_key)

    return data

def load_units_conversion():
    """Load the units conversion."""

    with open(UNITS_CONVERSION, "r") as f:
        data = yaml.full_load(f)

    return data

def create_lca_results_array(
        methods: List[str],
        years: List[int],
        regions: List[str],
        models: List[str],
        scenarios: List[str],
        classifications: dict,
) -> xr.DataArray:
    """
    Create an xarray DataArray to store Life Cycle Assessment (LCA) results.

    The DataArray has dimensions `act_category`, `impact_category`, `year`, `region`, `model`, and `scenario`.

    :param methods: List of impact assessment methods.
    :type methods: List[str]
    :param years: List of years to consider in the LCA results.
    :type years: List[int]
    :param regions: List of regions to consider in the LCA results.
    :type regions: List[str]
    :param models: List of models to consider in the LCA results.
    :type models: List[str]
    :param scenarios: List of scenarios to consider in the LCA results.
    :type scenarios: List[str]

    :return: An xarray DataArray with the appropriate coordinates and dimensions to store LCA results.
    :rtype: xr.DataArray
    """

    # Define the coordinates for the xarray DataArray
    coords = {
        "act_category": list(set(classifications.values())),
        "impact_category": methods,
        "year": years,
        "region": regions,
        "model": models,
        "scenario": scenarios,
    }

    # Create the xarray DataArray with the defined coordinates and dimensions.
    # The array is initialized with zeros.
    return xr.DataArray(
        np.zeros(
            (
                len(coords["act_category"]),
                len(methods),
                len(years),
                len(regions),
                len(models),
                len(scenarios),
            )
        ),
        coords=coords,
        dims=[
            "act_category",
            "impact_category",
            "year",
            "region",
            "model",
            "scenario",
        ],
    )

def display_results(lca_results: Union[xr.DataArray, None], cutoff: float = 0.01) -> xr.DataArray:
    """
    Aggregate and display Life Cycle Assessment (LCA) results in an xarray format.

    Aggregates 'activity categories' if they represent less than `cutoff` percent
    of the total impact and label these as 'other'. It also interpolates years
    to have a continuous time series.

    :param lca_results: Life Cycle Assessment results as an xarray DataArray.
    :type lca_results: Union[xr.DataArray, None]
    :param cutoff: The percentage below which 'activity categories' will be aggregated. Default is 0.01.
    :type cutoff: float, default is 0.01
    :raises ValueError: If `lca_results` is None.
    :return: Aggregated LCA results as an xarray DataArray.
    :rtype: xr.DataArray
    """

    # Check if lca_results is None
    if lca_results is None:
        raise ValueError("No results to display")

    # Convert lca_results to DataFrame and aggregate 'activity categories'
    df = (
        lca_results.to_dataframe("value")
        .reset_index()
        .groupby(
            [
                "model",
                "scenario",
                "act_category",
                "impact_category",
                "year",
                "region",
            ]
        )
        .sum()
        .reset_index()
    )

    # Get total impact per year
    df = df.merge(
        df.groupby(["model", "scenario", "impact_category", "year", "region"])[
            "value"
        ]
        .sum()
        .reset_index(),
        on=["impact_category", "year", "region"],
        suffixes=["", "_total"],
    )

    # Calculate percentage of total
    df["percentage"] = df["value"] / df["value_total"]

    # Aggregate 'activity categories' representing less than 'cutoff' percent of total
    df.loc[df["percentage"] < cutoff, "act_category"] = "other"

    # Drop columns 'value_total' and 'percentage'
    df = df.drop(columns=["value_total", "percentage"])

    # Aggregate again after labelling less represented 'activity categories' as 'other'
    arr = (
        df.groupby(
            [
                "model",
                "scenario",
                "act_category",
                "impact_category",
                "year",
                "region",
            ]
        )["value"]
        .sum()
        .to_xarray()
    )

    # Interpolate years to have a continuous time series if there's more than 1 year
    if len(arr.year) > 1:
        arr = arr.interp(
            year=np.arange(arr.year.min(), arr.year.max() + 1),
            kwargs={"fill_value": "extrapolate"},
            method="linear",
        )

    # Return aggregated results as an xarray DataArray
    return arr




