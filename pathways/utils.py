import sys
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import xarray as xr
import yaml

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


def _get_activity_indices(
    activities: List[Tuple[str, str, str, str]],
    A_index: Dict[Tuple[str, str, str, str], int],
    geo: Any,
) -> List[int]:
    """
    Fetch the indices of activities in the technosphere matrix.

    This function iterates over the provided list of activities. For each activity, it first tries to find the activity
    index in the technosphere matrix for the specific region. If the activity is not found, it looks for the activity
    in possible locations based on the IAM to Ecoinvent mapping. If still not found, it tries to find the activity index
    for some default locations ("RoW", "GLO", "RER", "CH").

    :param activities: A list of tuples, each representing an activity. Each tuple contains four elements: the name of
                       the activity, the reference product, the unit, and the region.
    :type activities: List[Tuple[str, str, str, str]]
    :param A_index: A dictionary mapping activity tuples to their indices in the technosphere matrix.
    :type A_index: Dict[Tuple[str, str, str, str], int]
    :param geo: An object providing an IAM to Ecoinvent location mapping.
    :type geo: Any
    :param region: The region for which to fetch activity indices.
    :type region: str

    :return: A list of activity indices in the technosphere matrix.
    :rtype: List[int]
    """

    indices = []  # List to hold the found indices

    # Iterate over each activity in the provided list
    for a, activity in enumerate(activities):
        # Try to find the index for the specific region
        idx = A_index.get(activity)
        if idx is not None:
            indices.append(int(idx))  # If found, add to the list
        else:
            # If not found, look for the activity in possible locations
            # find out if the location is an IAM location or just an ecoinvent location

            if (geo.model.upper(), activity[-1]) not in geo.geo.keys():
                print(f"{activity[-1]} is not an IAM location.")
                possible_locations = [activity[-1]]
            else:
                possible_locations = geo.iam_to_ecoinvent_location(activity[-1])

            for loc in possible_locations:
                idx = A_index.get((activity[0], activity[1], activity[2], loc))
                if idx is not None:
                    indices.append(int(idx))  # If found, add to the list
                    activities[a] = (activity[0], activity[1], activity[2], loc)
                    break  # Exit the loop as the index was found
            else:
                # If still not found, try some default locations
                for default_loc in ["RoW", "GLO", "RER", "CH"]:
                    idx = A_index.get(
                        (activity[0], activity[1], activity[2], default_loc)
                    )
                    if idx is not None:
                        indices.append(int(idx))  # If found, add to the list
                        activities[a] = (
                            activity[0],
                            activity[1],
                            activity[2],
                            default_loc,
                        )
                        break  # Exit the loop as the index was found
                else:
                    # If still not found, print a message and add None to the list
                    print(f"Activity {activity} not found in the technosphere matrix.")
                    indices.append(None)
        if idx is None:
            print(f"Activity {activity} not found in the technosphere matrix.")

    return indices  # Return the list of indices


def harmonize_units(scenario: xr.DataArray, variables: list) -> xr.DataArray:
    """
    Harmonize the units of a scenario. Some units ar ein PJ/yr, while others are in EJ/yr
    We want to convert everything to the same unit - preferably the largest one.
    :param scenario:
    :return:
    """

    units = [scenario.attrs["units"][var] for var in variables]

    # if not all units are the same, we need to convert
    if len(set(units)) > 1:
        if all(x in ["PJ/yr", "EJ/yr"] for x in units):
            # convert to EJ/yr
            # create vector of conversion factors
            conversion_factors = np.array([1e-3 if u == "PJ/yr" else 1 for u in units])
            # multiply scenario by conversion factors
            scenario.loc[dict(variables=variables)] *= conversion_factors[
                :, np.newaxis, np.newaxis
            ]
            # update units
            scenario.attrs["units"] = {var: "EJ/yr" for var in variables}

    return scenario


def get_unit_conversion_factors(
    scenario_unit: dict, dataset_unit: list, unit_mapping: dict
) -> np.ndarray:
    """
    Get the unit conversion factors for a given scenario unit and dataset unit.
    :param scenario_unit:
    :param dataset_unit:
    :param unit_mapping:
    :return:
    """

    return np.array(unit_mapping[scenario_unit][dataset_unit])


def load_units_conversion():
    """Load the units conversion."""

    with open(UNITS_CONVERSION, "r") as f:
        data = yaml.full_load(f)

    return data


def create_lca_results_array(
    methods: [List[str], None],
    B_indices: Dict,
    years: List[int],
    regions: List[str],
    locations: List[str],
    models: List[str],
    scenarios: List[str],
    classifications: dict,
    mapping: dict,
    flows: List[str] = None,
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
    :param locations: List of locations to consider in the LCA results.
    :type locations: List[str]
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
        "variable": list(mapping.keys()),
        "year": years,
        "region": regions,
        "location": locations,
        "model": models,
        "scenario": scenarios,
    }

    dims = (
        len(coords["act_category"]),
        len(coords["variable"]),
        len(years),
        len(regions),
        len(locations),
        len(models),
        len(scenarios),
    )

    if methods is not None:
        coords["impact_category"] = methods
        dims += (len(methods),)
    else:
        if flows is not None:
            coords["impact_category"] = [" - ".join(a) for a in flows]
            dims += (len(flows),)
        else:
            coords["impact_category"] = [" - ".join(a) for a in list(B_indices.keys())]
            dims += (len(B_indices),)

    # Create the xarray DataArray with the defined coordinates and dimensions.
    # The array is initialized with zeros.
    return xr.DataArray(np.zeros(dims), coords=coords, dims=list(coords.keys()))


def display_results(
    lca_results: Union[xr.DataArray, None], cutoff: float = 0.001
) -> xr.DataArray:
    if lca_results is None:
        raise ValueError("No results to display")

    df = (
        lca_results.to_dataframe("value")
        .reset_index()
        .groupby(
            [
                "model",
                "scenario",
                "year",
                "region",
                "impact_category",
                "variable",
                "act_category",
            ]
        )
        .sum()
        .reset_index()
    )

    # Aggregation for 'act_category'
    df = df.merge(
        df.groupby(
            [
                "model",
                "scenario",
                "year",
                "region",
                "impact_category",
            ]
        )["value"]
        .sum()
        .reset_index(),
        on=[
            "impact_category",
            "year",
            "region",
        ],
        suffixes=["", "_total"],
    )
    df["percentage"] = df["value"] / df["value_total"]
    df.loc[df["percentage"] < cutoff, "act_category"] = "other"

    df = df.drop(columns=["value_total", "percentage"])

    arr = (
        df.groupby(
            [
                "model",
                "scenario",
                "year",
                "region",
                "impact_category",
                "variable",
                "act_category",
            ]
        )["value"]
        .sum()
        .to_xarray()
    )

    if len(arr.year) > 1:
        arr = arr.interp(
            year=np.arange(arr.year.min(), arr.year.max() + 1),
            kwargs={"fill_value": "extrapolate"},
            method="linear",
        )

    return arr
