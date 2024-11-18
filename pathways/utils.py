"""
Utilities for the pathways module.

These utilities include functions for loading activities classifications and units conversion, harmonizing units,
creating an LCA results array, displaying results, loading a numpy array from disk, getting visible files, cleaning the
cache directory, resizing scenario data, fetching indices, fetching inventories locations, converting a CSV file to a
dictionary, checking unclassified activities, and getting activity indices.

"""

import csv
import logging
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from datapackage import DataPackage, DataPackageException
from premise.geomap import Geomap

from .filesystem_constants import DATA_DIR, DIR_CACHED_DB, USER_LOGS_DIR

CLASSIFICATIONS = DATA_DIR / "activities_classifications.yaml"
UNITS_CONVERSION = DATA_DIR / "units_conversion.yaml"

logging.basicConfig(
    level=logging.DEBUG,
    filename=USER_LOGS_DIR / "pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_indices_csv(file_path: Path) -> dict[tuple[str, str, str, str], int]:
    """
    Reads a CSV file and returns its contents as a dictionary.

    Each row of the CSV file is expected to contain four string values followed by an index.
    These are stored in the dictionary as a tuple of the four strings mapped to the index.

    :param file_path: The path to the CSV file.
    :type file_path: Path

    :return: A dictionary mapping tuples of four strings to indices.
    For technosphere indices, the four strings are the activity name, product name, location, and unit.
    For biosphere indices, the four strings are the flow name, category, subcategory, and unit.
    :rtype: Dict[Tuple[str, str, str, str], str]
    """
    indices = dict()
    with open(file_path, encoding="utf-8") as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=";")
        for row in csv_reader:
            try:
                indices[(row[0], row[1], row[2], row[3])] = int(row[4])
            except IndexError as err:
                logging.error(
                    f"Error reading row {row} from {file_path}: {err}. "
                    f"Could it be that the file uses commas instead of semicolons?"
                )
    # remove any unicode characters
    indices = {tuple([str(x) for x in k]): v for k, v in indices.items()}
    return indices


def load_mapping(mapping: [dict, str]) -> dict:
    """
    Load the geography mapping.
    :param mapping: dict or yaml file with the geography mapping.
    :return: dict
    """

    if isinstance(mapping, dict):
        return mapping
    elif isinstance(mapping, str):
        with open(mapping, "r") as f:
            data = yaml.full_load(f)
        return data
    else:
        raise ValueError("Invalid geography mapping")


def load_classifications():
    """Load the activities classifications."""

    # check if file exists
    if not Path(CLASSIFICATIONS).exists():
        raise FileNotFoundError(f"File {CLASSIFICATIONS} not found")

    with open(CLASSIFICATIONS, "r") as f:
        data = yaml.full_load(f)

    return data


def harmonize_units(scenario: xr.DataArray, variables: list) -> xr.DataArray:
    """
    Harmonize the units of a scenario. Some units are in PJ/yr, while others are in EJ/yr
    We want to convert everything to the same unit - preferably the largest one.
    :param scenario: xr.DataArray
    :param variables: list of variables
    :return: xr.DataArray
    """

    missing_vars = [var for var in variables if var not in scenario.attrs["units"]]
    if missing_vars:
        raise KeyError(
            f"The following variables are missing in 'scenario.attrs[\"units\"]': {missing_vars}"
        )

    units = [scenario.attrs["units"][var] for var in variables]

    if len(variables) == 0:
        raise ValueError("Empty list of variables")

    # if not all units are the same, we need to convert
    if len(set(units)) > 1:
        if all(x in ["PJ/yr", "EJ/yr", "PJ/yr."] for x in units):
            # convert to EJ/yr
            # create vector of conversion factors
            conversion_factors = np.array(
                [1e-3 if u in ("PJ/yr", "PJ/yr.") else 1 for u in units]
            )
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

    if scenario_unit != dataset_unit:
        return np.array(unit_mapping.get(scenario_unit, {})[dataset_unit])
    return np.array([1])


def load_units_conversion() -> dict:
    """Load the units conversion."""

    with open(UNITS_CONVERSION, "r") as f:
        data = yaml.full_load(f)

    return data


def create_lca_results_array(
    methods: List[str],
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

    # check if any of the list parameters is empty, and if so, throw an error
    if len(methods) == 0:
        raise ValueError("Empty list of methods")
    if len(years) == 0:
        raise ValueError("Empty list of years")
    if len(regions) == 0:
        raise ValueError("Empty list of regions")
    if len(locations) == 0:
        raise ValueError("Empty list of locations")
    if len(models) == 0:
        raise ValueError("Empty list of models")
    if len(scenarios) == 0:
        raise ValueError("Empty list of scenarios")

    # Define the coordinates for the xarray DataArray
    coords = {
        "act_category": list(set(list(classifications.values()))),
        "variable": list(mapping.keys()),
        "year": years,
        "region": regions,
        "location": locations,
        "model": models,
        "scenario": scenarios,
        "impact_category": methods,
    }

    if use_distributions is True:
        # we calculate the 5th, 50th, and 95th percentiles
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


def export_results_to_parquet(lca_results: xr.DataArray, filepath: str) -> str:
    """
    Export the LCA results to a parquet file.
    :param lca_results: Xarray DataArray with LCA results.
    :param filepath: The path to the parquet file.
    :return: None
    """
    if filepath is None:
        filepath = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gzip"
    else:
        filepath = f"{filepath}.gzip"

    flattened_data = lca_results.values.flatten()

    # Step 2: Find the indices of non-zero values
    non_zero_indices = np.nonzero(flattened_data)[0]

    # Step 3: Extract non-zero values
    non_zero_values = flattened_data[non_zero_indices]

    # Step 4: Get the shape of the original DataArray
    original_shape = lca_results.shape

    # Step 5: Find the coordinates corresponding to the non-zero indices
    coords = np.array(np.unravel_index(non_zero_indices, original_shape)).T

    # Step 6: Create a pandas DataFrame with non-zero values and corresponding coordinates
    coord_names = list(lca_results.dims)
    coord_values = {
        dim: lca_results.coords[dim].values[coords[:, i]]
        for i, dim in enumerate(coord_names)
    }

    df = pd.DataFrame(coord_values)
    df["value"] = non_zero_values
    df.to_parquet(path=filepath, compression="gzip")

    print(f"Results exported to {filepath}")

    return filepath


def display_results(
    lca_results: Union[xr.DataArray, None],
    cutoff: float = 0.001,
    interpolate: bool = False,
) -> xr.DataArray:
    """
    Display the LCA results.
    Remove results below a cutoff value and aggregate them into a single category.
    :param lca_results: The LCA results.
    :param cutoff: The cutoff value.
    :param interpolate: A boolean indicating whether to interpolate the results.
    :return: The LCA results.
    :rtype: xr.DataArray
    """
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


def get_visible_files(path: str) -> list[Path]:
    """
    Get visible files in a directory.
    :param path: The path to the directory.
    :return: List of visible files.
    """
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


def get_activity_indices(
    activities: List[Tuple],
    technosphere_index: Dict[Tuple, Any],
    geo: Geomap,
    debug: bool = False,
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
                logging.warning(
                    f"Activity {activity} not found in the technosphere matrix."
                )

    return indices


def fetch_indices(
    mapping: dict, regions: list, variables: list, technosphere_index: dict, geo: Geomap
) -> dict:
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
            mapping[variable]["dataset"][0]["unit"],
        )
        for variable in variables
    }

    # Initialize dictionary to hold indices
    vars_idx = {}

    for region in regions:
        # Construct activities list for the current region
        activities = [
            (name, ref_product, unit, region)
            for name, ref_product, unit in activities_info.values()
        ]

        # Use _get_activity_indices to fetch indices
        idxs = get_activity_indices(activities, technosphere_index, geo)

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


def get_all_indices(vars_info: dict) -> list[int]:
    """
    Extract all 'idx' values from the vars_info dictionary.

    :param vars_info: Dictionary of variables information returned by fetch_indices.
    :type vars_info: dict
    :return: List of all 'idx' values.
    :rtype: list[int]
    """
    idx_list = []
    for region_data in vars_info.values():
        for variable_data in region_data.values():
            idx_list.append(variable_data["idx"])
    return idx_list


def fetch_inventories_locations(technosphere_indices: dict) -> List[str]:
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


def check_unclassified_activities(
    technosphere_indices: dict, classifications: dict
) -> List:
    """
    Check if there are activities in the technosphere matrix that are not in the classifications.
    :param technosphere_indices: List of activities in the technosphere matrix.
    :param classifications: Dictionary of activities classifications.
    :return: List of activities not found in the classifications.
    """
    missing_classifications = []
    for act in technosphere_indices:
        if act[:3] not in classifications:
            missing_classifications.append(list(act[:3]))

    if missing_classifications:
        with open("missing_classifications.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(missing_classifications)

    return missing_classifications


def _group_technosphere_indices(
    technosphere_indices: dict,
    group_by,
    group_values: list,
    mapping: dict = None,
) -> dict:
    """
    Generalized function to group technosphere indices by an arbitrary attribute (category, location, etc.).

    :param technosphere_indices: Mapping of activities to their indices in the technosphere matrix.
    :param group_by: A function that takes an activity and returns its group value (e.g., category or location).
    :param group_values: The set of all possible group values (e.g., all categories or locations).
    :param mapping: A dictionary mapping.
    :return: A tuple containing a list of lists of indices, a dictionary mapping group values to lists of indices,
             and a 2D numpy array of indices, where rows have been padded with -1 to ensure equal lengths.
    """

    # create an ordered dictionary to store the indices
    acts_dict = OrderedDict(
        (
            value,
            [
                int(technosphere_indices[a])
                for a in technosphere_indices
                if group_by(a) == value
            ],
        )
        for value in group_values
    )

    if mapping:
        aggregated = {}
        for k, v in acts_dict.items():
            if mapping.get(k, k) in aggregated:
                aggregated[mapping.get(k, k)].extend(v)
            else:
                aggregated[mapping.get(k, k)] = v

        # reorder the dictionary to match with the
        # order of mapping.values()
        aggregated = {k: aggregated[k] for k in mapping.values()}

        return aggregated

    return acts_dict


def read_categories_from_yaml(file_path: Path) -> Dict:
    """
    Read categories from a YAML file.

    :param file_path: The path to the YAML file.
    :return: The categories.
    """
    with open(file_path, "r") as file:
        filters = yaml.safe_load(file)

    return filters


def gather_filters(current_level: Dict, combined_filters: Dict[str, Set[str]]) -> None:
    """
    Recursively gather filters from the current level and all sub-levels.

    :param current_level: The current level in the filters dictionary.
    :param combined_filters: The combined filter criteria dictionary.
    """
    if "ecoinvent_aliases" in current_level:
        ecoinvent_aliases = current_level["ecoinvent_aliases"]
        for k in combined_filters.keys():
            combined_filters[k].update(ecoinvent_aliases.get(k, []))
    for key, value in current_level.items():
        if isinstance(value, dict):
            gather_filters(value, combined_filters)


def get_combined_filters(
    filters: Dict, paths: List[List[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Traverse the filters dictionary to get combined filter criteria based on multiple paths.

    :param filters: The filters dictionary loaded from YAML.
    :param paths: A list of lists, where each inner list represents a path in the hierarchy.
    :return: A tuple with combined filter criteria dictionary and exceptions dictionary.
    """
    combined_filters = {
        "name_fltr": set(),
        "name_mask": set(),
        "product_fltr": set(),
        "product_mask": set(),
    }

    exceptions_filters = {
        "name_fltr": set(),
        "product_fltr": set(),
    }

    for path in paths:
        current_level = filters
        for key in path:
            current_level = current_level.get(key, {})
            if not isinstance(current_level, dict):
                break
        gather_filters(current_level, combined_filters)

    # Gather exceptions from the "Exceptions" path
    exceptions = filters.get("Exceptions", {}).get("ecoinvent_aliases", {})
    for k in exceptions_filters.keys():
        exceptions_filters[k].update(exceptions.get(k, []))

    for k in combined_filters.keys():
        combined_filters[k] = list(combined_filters[k])
    for k in exceptions_filters.keys():
        exceptions_filters[k] = list(exceptions_filters[k])

    return combined_filters, exceptions_filters


def apply_filters(
    technosphere_inds: Dict[Tuple[str, str, str, str], int],
    filters: Dict[str, List[str]],
    exceptions: Dict[str, List[str]],
    paths: List[List[str]],  # Add paths as an argument
) -> Tuple[List[int], List[int], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Apply the filters to the database and return a list of indices and exceptions,
    along with the names of filtered activities and exceptions categorized by paths.

    :param technosphere_inds: Dictionary where keys are tuples of four strings (activity name, product name, location, unit)
                     and values are integers (indices).
    :param filters: Dictionary containing the filter criteria.
    :param exceptions: Dictionary containing the exceptions criteria.
    :param paths: List of lists, where each inner list represents a path in the hierarchy.
    :return: Tuple containing a list of indices of filtered activities, a list of indices of exceptions,
             and dictionaries of categorized names of filtered activities and exceptions.
    """
    name_fltr = filters.get("name_fltr", [])
    name_mask = filters.get("name_mask", [])
    product_fltr = filters.get("product_fltr", [])
    product_mask = filters.get("product_mask", [])

    exception_name_fltr = exceptions.get("name_fltr", [])
    exception_product_fltr = exceptions.get("product_fltr", [])
    exception_name_mask = exceptions.get("name_mask", [])
    exception_product_mask = exceptions.get("product_mask", [])

    def match_filter(item, filter_values):
        return any(fltr in item for fltr in filter_values)

    def match_mask(item, mask_values):
        return any(msk in item for msk in mask_values)

    filtered_indices = []
    exception_indices = []
    filtered_names = {tuple(path): set() for path in paths}
    exception_names = {tuple(path): set() for path in paths}

    for key, value in technosphere_inds.items():
        name, product, location, unit = key

        if name_fltr and not match_filter(name, name_fltr):
            continue
        if product_fltr and not match_filter(product, product_fltr):
            continue
        if name_mask and match_mask(name, name_mask):
            continue
        if product_mask and match_mask(product, product_mask):
            continue

        filtered_indices.append(value)
        for path in paths:
            path_str = " ".join(path)
            if match_filter(name, path_str):
                filtered_names[tuple(path)].add(name)

    for key, value in technosphere_inds.items():
        name, product, location, unit = key

        if exception_name_fltr and not match_filter(name, exception_name_fltr):
            continue
        if exception_product_fltr and not match_filter(product, exception_product_fltr):
            continue
        if exception_name_mask and match_mask(name, exception_name_mask):
            continue
        if exception_product_mask and match_mask(product, exception_product_mask):
            continue

        exception_indices.append(value)
        for path in paths:
            path_str = " ".join(path)
            if match_filter(name, path_str):
                exception_names[tuple(path)].add(name)

    return filtered_indices, exception_indices, filtered_names, exception_names


# Custom filter function
# Custom context manager for filtering warnings
class CustomFilter:
    def __init__(self, ignore_message):
        self.ignore_message = ignore_message

    def __enter__(self):
        # Capture the original warning show function
        self.original_showwarning = warnings.showwarning
        warnings.showwarning = self.custom_showwarning

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original warning show function
        warnings.showwarning = self.original_showwarning

    def custom_showwarning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        # Check if the warning message should be ignored
        if self.ignore_message not in str(message):
            self.original_showwarning(message, category, filename, lineno, file, line)


def _get_mapping(data) -> dict:
    """
    Read the mapping file which maps scenario variables to LCA datasets.
    It's a YAML file.
    :return: dict

    """
    return yaml.safe_load(data.get_resource("mapping").raw_read())


def _read_scenario_data(data: dict, scenario: str):
    """
    Read the scenario data.
    The scenario data describes scenario variables with production volumes for each time step.
    :param scenario: str. Scenario name.
    :return: pd.DataFrame

    """
    filepath = data["scenarios"][scenario]["path"]
    # if CSV file
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, index_col=0)

    # Excel file
    return pd.read_excel(filepath, index_col=0)


def _read_datapackage(datapackage: str) -> DataPackage:
    """Read the datapackage.json file.

    :return: DataPackage
    """

    return DataPackage(datapackage)
