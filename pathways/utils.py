import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import logging

import numpy as np
import xarray as xr
import yaml
from premise.geomap import Geomap

from .filesystem_constants import DATA_DIR, DIR_CACHED_DB

CLASSIFICATIONS = DATA_DIR / "activities_classifications.yaml"
UNITS_CONVERSION = DATA_DIR / "units_conversion.yaml"


logging.basicConfig(
    level=logging.DEBUG,
    filename="pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

def harmonize_units(scenario: xr.DataArray, variables: list) -> xr.DataArray:
    """
    Harmonize the units of a scenario. Some units are in PJ/yr, while others are in EJ/yr
    We want to convert everything to the same unit - preferably the largest one.
    :param scenario: xr.DataArray
    :param variables: list of variables
    :return: xr.DataArray
    """

    units = [scenario.attrs["units"][var] for var in variables]

    # if not all units are the same, we need to convert
    if len(set(units)) > 1:
        if all(x in ["PJ/yr", "EJ/yr", "PJ/yr."] for x in units):
            # convert to EJ/yr
            # create vector of conversion factors
            conversion_factors = np.array([1e-3 if u in ("PJ/yr", "PJ/yr.") else 1 for u in units])
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
    years: List[int],
    regions: List[str],
    locations: List[str],
    models: List[str],
    scenarios: List[str],
    classifications: dict,
    mapping: dict,
    use_distributions: bool = False,
) -> xr.DataArray:
    """
    Create an xarray DataArray to store Life Cycle Assessment (LCA) results.

    The DataArray has dimensions `act_category`, `impact_category`, `year`, `region`, `model`, and `scenario`.

    :param methods: A list of impact categories.
    :type methods: List[str]
    :param years: A list of years.
    :type years: List[int]
    :param regions: A list of regions.
    :type regions: List[str]
    :param locations: A list of locations.
    :type locations: List[str]
    :param models: A list of models.
    :type models: List[str]
    :param scenarios: A list of scenarios.
    :type scenarios: List[str]
    :param classifications: A dictionary mapping activities to categories.
    :type classifications: dict
    :param mapping: A dictionary mapping scenario variables to LCA datasets.
    :type mapping: dict
    :param use_distributions: A boolean indicating whether to use distributions.
    :type use_distributions: bool

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
        "impact_category": methods,
    }

    if use_distributions is True:
        coords.update({"quantile": [0.05, 0.5, 0.95]})

    dims = (
        len(coords["act_category"]),
        len(coords["variable"]),
        len(years),
        len(regions),
        len(locations),
        len(models),
        len(scenarios),
        len(methods),
    )

    if use_distributions is True:
        dims += (3,)

    # Create the xarray DataArray with the defined coordinates and dimensions.
    # The array is initialized with zeros.
    return xr.DataArray(np.zeros(dims), coords=coords, dims=list(coords.keys()))


def display_results(
    lca_results: Union[xr.DataArray, None],
    cutoff: float = 0.001,
    interpolate: bool = False,
) -> xr.DataArray:
    if lca_results is None:
        raise ValueError("No results to display")

    if len(lca_results.year) > 1 and interpolate:
        lca_results = lca_results.interp(
            year=np.arange(lca_results.year.min(), lca_results.year.max() + 1),
            kwargs={"fill_value": "extrapolate"},
            method="linear",
        )

    above_cutoff = lca_results.where(lca_results > cutoff)

    # Step 2: Aggregate values below the cutoff across the 'act_category' dimension
    # Summing all values below the cutoff for each combination of other dimensions
    below_cutoff = lca_results.where(lca_results <= cutoff).sum(dim="act_category")

    # Since summing removes the 'act_category', we need to add it back
    # Create a new coordinate for 'act_category' that includes 'other'
    new_act_category = np.append(lca_results.act_category.values, "other")

    # Create a new DataArray for below-cutoff values with 'act_category' as 'other'
    # This involves broadcasting below_cutoff to match the original array's dimensions but with 'act_category' replaced
    other_data = below_cutoff.expand_dims({"act_category": ["other"]}, axis=0)

    # Step 3: Combine the above-cutoff data with the new 'other' data
    # Concatenate along the 'act_category' dimension
    combined = xr.concat([above_cutoff, other_data], dim="act_category")

    # Ensure the 'act_category' coordinate is updated to include 'other'
    combined = combined.assign_coords({"act_category": new_act_category})

    return combined


def load_numpy_array_from_disk(filepath):
    """
    Load a numpy array from disk.
    :param filepath: The path to the file containing the numpy array.
    :return: numpy array
    """

    return np.load(filepath, allow_pickle=True)


def get_visible_files(path):
    return [file for file in Path(path).iterdir() if not file.name.startswith(".")]


def clean_cache_directory():
    # clean up the cache directory
    for file in get_visible_files(DIR_CACHED_DB):
        file.unlink()


def resize_scenario_data(
    scenario_data: xr.DataArray,
    model: List[str],
    scenario: List[str],
    region: List[str],
    year: List[int],
    variables: List[str],
) -> xr.DataArray:
    """
    Resize the scenario data to the given scenario, year, region, and variables.
    :param model: List of models.
    :param scenario_data: xarray DataArray with scenario data.
    :param scenario: List of scenarios.
    :param year: List of years.
    :param region: List of regions.
    :param variables: List of variables.
    :return: Resized scenario data.
    """

    # Get the indices for the given scenario, year, region, and variables
    model_idx = [scenario_data.coords["model"].values.tolist().index(x) for x in model]
    scenario_idx = [
        scenario_data.coords["pathway"].values.tolist().index(x) for x in scenario
    ]
    year_idx = [scenario_data.coords["year"].values.tolist().index(x) for x in year]
    region_idx = [
        scenario_data.coords["region"].values.tolist().index(x) for x in region
    ]
    variable_idx = [
        scenario_data.coords["variables"].values.tolist().index(x) for x in variables
    ]

    # Resize the scenario data
    scenario_data = scenario_data.isel(
        model=model_idx,
        pathway=scenario_idx,
        year=year_idx,
        region=region_idx,
        variables=variable_idx,
    )

    return scenario_data

def _get_activity_indices(
        activities: List[Tuple[str, str, str, str]],
        technosphere_index: Dict[Tuple[str, str, str, str], Any],
        geo: Geomap,
        debug: bool = False
) -> List[int]:
    """
    Fetch the indices of activities in the technosphere matrix, optimized for efficiency.
    """

    # Cache for previously computed IAM to Ecoinvent mappings
    location_cache = {}

    indices = []  # Output list of indices

    for activity in activities:
        possible_locations = [activity[-1]]  # Start with the activity's own region

        # Extend possible locations with IAM mappings, if applicable
        if (geo.model.upper(), activity[-1]) in geo.geo:
            # Use cached result if available
            if activity[-1] in location_cache:
                possible_locations.extend(location_cache[activity[-1]])
            else:
                mappings = geo.iam_to_ecoinvent_location(activity[-1])
                location_cache[activity[-1]] = mappings
                possible_locations.extend(mappings)

        # Add default locations to the end of the search list
        possible_locations.extend(["RoW", "GLO", "RER", "CH"])

        # Attempt to find the index in technosphere_index
        for loc in possible_locations:
            idx = technosphere_index.get((activity[0], activity[1], activity[2], loc))
            if idx is not None:
                indices.append(int(idx))
                break
        else:
            # If the index was not found, append None and optionally log
            indices.append(None)

            if debug:
                logging.warning(f"Activity {activity} not found in the technosphere matrix.")

    return indices


def fetch_indices(mapping: dict, regions: list, variables: list, technosphere_index: dict, geo: Geomap) -> dict:
    """
    Fetch the indices for the given activities in the technosphere matrix.

    :param mapping: Mapping of scenario variables to LCA datasets.
    :type mapping: dict
    :param regions: List of regions.
    :type regions: list
    :param variables: List of variables.
    :type variables: list
    :param technosphere_index: Technosphere index.
    :type technosphere_index: dict
    :param geo: Geomap object.
    :type geo: Geomap
    :return: Dictionary of indices.
    :rtype: dict
    """

    # Pre-process mapping data to minimize repetitive data access
    activities_info = {
        variable: (
            mapping[variable]["dataset"][0]["name"],
            mapping[variable]["dataset"][0]["reference product"],
            mapping[variable]["dataset"][0]["unit"]
        )
        for variable in variables
    }

    # Initialize dictionary to hold indices
    vars_idx = {}

    for region in regions:
        # Construct activities list for the current region
        activities = [(name, ref_product, unit, region) for name, ref_product, unit in activities_info.values()]

        # Use _get_activity_indices to fetch indices
        idxs = _get_activity_indices(activities, technosphere_index, geo)

        # Map variables to their indices and associated dataset information
        vars_idx[region] = {
            variable: {
                "idx": idx,
                "dataset": activities[i],
            }
            for i, (variable, idx) in enumerate(zip(variables, idxs))
        }

        if len(variables) != len(idxs):
            logging.warning(f"Could not find all activities for region {region}.")

    return vars_idx


def fetch_inventories_locations(technosphere_indices: Dict[str, Tuple[str, str, str]]) -> List[str]:
    """
    Fetch the locations of the inventories.
    :param technosphere_indices: Dictionary with the indices of the activities in the technosphere matrix.
    :return: List of locations.
    """

    locations = list(set([act[3] for act in technosphere_indices]))
    logging.info(f"Unique locations in LCA database: {locations}")

    return locations


def csv_to_dict(filename: str) -> dict[int, tuple[str, ...]]:
    """
    Convert a CSV file to a dictionary.
    :param filename: The name of the CSV file.
    :return: A dictionary with the data from the CSV file.
    """
    output_dict = {}

    with open(filename, encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            # Making sure there are at least 5 items in the row
            if len(row) >= 5:
                # The first four items are the key, the fifth item is the value
                key = tuple(row[:4])
                value = row[4]
                output_dict[int(value)] = key
            else:
                logging.warning(f"Row {row} has less than 5 items.")

    return output_dict
