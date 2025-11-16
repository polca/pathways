"""
Utilities for the pathways module.

These utilities include functions for loading activities classifications and units conversion, harmonizing units,
creating an LCA results array, displaying results, loading a numpy array from disk, getting visible files, cleaning the
cache directory, resizing scenario data, fetching indices, fetching inventories locations, converting a CSV file to a
dictionary, checking unclassified activities, and getting activity indices.

"""

from __future__ import annotations
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

from .filesystem_constants import DATA_DIR, DIR_CACHED_DB

CLASSIFICATIONS = DATA_DIR / "classifications.csv"
UNITS_CONVERSION = DATA_DIR / "units_conversion.yaml"

logger = logging.getLogger(__name__)


def read_indices_csv(file_path: Path) -> dict[tuple[str, str, str, str], int]:
    """Parse a semicolon-separated index CSV into a lookup dictionary.

    :param file_path: Path to the CSV file containing activity metadata.
    :type file_path: pathlib.Path
    :returns: Mapping from ``(name, product, location, unit)`` tuples to indices.
    :rtype: dict[tuple[str, str, str, str], int]
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
    """Load a geography or activity mapping from a dict or YAML file.

    :param mapping: Either a ready mapping dictionary or a YAML path.
    :type mapping: dict | str
    :returns: Mapping dictionary loaded from the provided source.
    :rtype: dict
    :raises ValueError: If ``mapping`` is neither a dict nor a string path.
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
    """Load the bundled activity classification hierarchy from a CSV file.

    The CSV is expected to have at least the columns:
        - 'name'
        - 'reference product'
    and any number of additional columns, each representing a
    classification system (e.g. 'ISIC rev.4 ecoinvent', 'CPC').

    Returns
    -------
    dict
        Mapping keyed by (name, reference_product) tuples, with values
        a list of (system, code) tuples.
    """

    path = Path(CLASSIFICATIONS)
    if not path.exists():
        raise FileNotFoundError(f"File {CLASSIFICATIONS} not found")

    classifications = {}

    # utf-8-sig handles BOM if present
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        raw_fieldnames = reader.fieldnames or []
        # normalize headers: strip whitespace and any BOM remnants
        cleaned_fieldnames = [fn.strip().lstrip("\ufeff") for fn in raw_fieldnames]

        required = {"name", "reference product"}
        missing = required - set(cleaned_fieldnames)
        if missing:
            raise ValueError(
                f"Classification CSV {CLASSIFICATIONS} is missing required "
                f"columns: {', '.join(sorted(missing))}"
            )

        # Build a mapping from original -> cleaned header so we can normalize keys
        header_map = {
            raw: cleaned for raw, cleaned in zip(raw_fieldnames, cleaned_fieldnames)
        }

        for row in reader:
            # normalize row keys using header_map
            row_clean = {header_map.get(k, k).strip(): v for k, v in row.items()}

            name = (row_clean.get("name") or "").strip()
            ref = (row_clean.get("reference product") or "").strip()

            if not name or not ref:
                continue

            key = (name, ref)

            # every other non-empty column is treated as a classification system
            for col, val in row_clean.items():
                if col in {"name", "reference product"}:
                    continue
                if val is None:
                    continue

                code = str(val).strip()
                if not code:
                    continue

                system = col.strip()
                classifications.setdefault(key, []).append((system, code))

    return classifications


def harmonize_units(scenario: xr.DataArray, variables: list) -> xr.DataArray:
    """Convert scenario variables to consistent energy units when necessary.

    :param scenario: Scenario data array with ``attrs["units"]`` metadata.
    :type scenario: xarray.DataArray
    :param variables: Variable names that should be harmonized.
    :type variables: list[str]
    :raises KeyError: When a variable lacks unit metadata.
    :raises ValueError: When no variables are provided.
    :returns: Updated scenario array with consistent units.
    :rtype: xarray.DataArray
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
    """Retrieve conversion factors aligning scenario units with dataset units.

    :param scenario_unit: Unit metadata declared in the scenario data.
    :type scenario_unit: str
    :param dataset_unit: Target unit tuple from the dataset mapping.
    :type dataset_unit: list[str] | str
    :param unit_mapping: Conversion factor dictionary loaded from YAML.
    :type unit_mapping: dict
    :raises KeyError: If no conversion factor is defined for the unit pair.
    :returns: Conversion factors as a NumPy array.
    :rtype: numpy.ndarray
    """

    if scenario_unit != dataset_unit:
        try:
            return np.array(unit_mapping.get(scenario_unit, {})[dataset_unit])
        except KeyError:
            raise KeyError(
                f"Unit conversion factor not found for scenario unit {scenario_unit} and dataset unit {dataset_unit}"
            )
    return np.array([1])


def load_units_conversion() -> dict:
    """Load the units conversion table bundled with the package.

    :returns: Mapping from scenario units to dataset unit conversion factors.
    :rtype: dict
    """

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
    """Create an empty ``xarray.DataArray`` with coordinates for storing LCA results.

    :param methods: LCIA method names.
    :type methods: list[str]
    :param years: Years included in the results tensor.
    :type years: list[int]
    :param regions: IAM regions covered by the results.
    :type regions: list[str]
    :param locations: Technosphere locations associated with activities.
    :type locations: list[str]
    :param models: IAM models included.
    :type models: list[str]
    :param scenarios: Pathways under analysis.
    :type scenarios: list[str]
    :param classifications: Mapping from activities to category labels.
    :type classifications: dict
    :param mapping: Scenario variable mapping used to populate the ``variable`` coordinate.
    :type mapping: dict
    :param use_distributions: Whether to append a ``quantile`` dimension for Monte Carlo statistics.
    :type use_distributions: bool
    :returns: Zero-initialized results array with all coordinates defined.
    :rtype: xarray.DataArray
    :raises ValueError: If any required coordinate list is empty.
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
        "act_category": list(set(list(classifications.values()))) + ["unclassified"],
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
    """Write non-zero LCA results to a gzip-compressed parquet file.

    :param lca_results: Results array to serialize.
    :type lca_results: xarray.DataArray
    :param filepath: Optional base filename without extension.
    :type filepath: str | None
    :returns: Actual ``.gzip`` file path written.
    :rtype: str
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


import numpy as np
import xarray as xr
from typing import Union


def display_results(
    lca_results: Union[xr.DataArray, None],
    cutoff: float = 0.001,
    interpolate: bool = False,
) -> xr.DataArray:
    """Filter and optionally interpolate LCA results for presentation.

    :param lca_results: Computed LCA results.
    :type lca_results: xarray.DataArray
    :param cutoff: Minimum contribution retained per activity category.
    :type cutoff: float
    :param interpolate: Interpolate across years when ``True``.
    :type interpolate: bool
    :raises ValueError: If results are missing or lack an ``act_category`` dimension.
    :returns: Processed DataArray with low contributors grouped as ``other``.
    :rtype: xarray.DataArray
    """
    if lca_results is None:
        raise ValueError("No results to display")
    if "act_category" not in lca_results.dims:
        raise ValueError("'act_category' must be a dimension of lca_results")

    # Optional interpolation
    if (
        interpolate
        and "year" in lca_results.dims
        and lca_results.sizes.get("year", 0) > 1
    ):
        y_min = int(np.asarray(lca_results["year"].min()).item())
        y_max = int(np.asarray(lca_results["year"].max()).item())
        lca_results = lca_results.interp(
            year=np.arange(y_min, y_max + 1),
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

    mask = lca_results > cutoff
    above = lca_results.where(mask)

    # Sum below/equal cutoff across act_category, then add it back as a new "other" category
    other = (
        lca_results.where(~mask)
        .sum(dim="act_category", skipna=True)
        .expand_dims(act_category=["other"])  # <-- create dim and its coord together
    )

    # Optional: drop empty categories to reduce size
    above = above.dropna(dim="act_category", how="all")

    # Concatenate and set final coordinate (existing categories + "other")
    combined = xr.concat([above, other], dim="act_category")
    final_act_cats = list(above.get_index("act_category")) + ["other"]
    combined = combined.assign_coords(act_category=final_act_cats)

    # prune zeros (pick a small tol if you want to ignore tiny numerical noise)
    combined = prune_zero_coords(combined, tol=1e-15)
    return combined


def prune_zero_coords(da: xr.DataArray, tol: float = 0.0) -> xr.DataArray:
    """Drop coordinate labels whose slices are (near) zero along every other dimension.

    :param da: DataArray to prune.
    :type da: xarray.DataArray
    :param tol: Absolute tolerance for considering a value zero.
    :type tol: float
    :returns: Pruned DataArray.
    :rtype: xarray.DataArray
    """
    x = da.fillna(0.0)  # treat NaNs as zeros for pruning
    for dim in list(x.dims):
        other = [d for d in x.dims if d != dim]
        mag = np.abs(x.astype(float))  # works with dask too (np.abs is a ufunc)

        if other:  # multi-dim: keep labels where ANY value across other dims > tol
            keep = (mag > tol).any(dim=other)
        else:  # 1-D case: elementwise keep mask
            keep = mag > tol

        x = x.sel({dim: keep})
    return x


def load_numpy_array_from_disk(filepath):
    """Load a NumPy array saved on disk, allowing pickled objects.

    :param filepath: File path produced by ``numpy.save``.
    :type filepath: str | pathlib.Path
    :returns: Loaded NumPy array.
    :rtype: numpy.ndarray
    """

    return np.load(filepath, allow_pickle=True)


def get_visible_files(path: str) -> list[Path]:
    """List non-hidden entries in a directory.

    :param path: Directory to inspect.
    :type path: str | pathlib.Path
    :returns: Paths of files not starting with ``.``.
    :rtype: list[pathlib.Path]
    """
    return [file for file in Path(path).iterdir() if not file.name.startswith(".")]


def clean_cache_directory():
    """Remove cached arrays stored in :data:`pathways.filesystem_constants.DIR_CACHED_DB`.

    :returns: ``None``
    :rtype: None
    """

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
    """Subset the scenario dataset to the requested models, pathways, regions, years, and variables.

    :param scenario_data: Original scenario data array.
    :type scenario_data: xarray.DataArray
    :param model: Models to retain.
    :type model: list[str]
    :param scenario: Pathways to retain.
    :type scenario: list[str]
    :param region: Regions to retain.
    :type region: list[str]
    :param year: Years to retain.
    :type year: list[int]
    :param variables: Variables to retain.
    :type variables: list[str]
    :returns: Resized scenario data array.
    :rtype: xarray.DataArray
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
    """Resolve technosphere indices for the supplied activity descriptors.

    :param activities: Sequence of ``(name, product, unit, region)`` tuples.
    :type activities: list[tuple[str, str, str, str]]
    :param technosphere_index: Mapping from activity descriptors to indices.
    :type technosphere_index: dict[tuple[str, str, str, str], int]
    :param geo: Geomap helper for geographic fallback logic.
    :type geo: premise.geomap.Geomap
    :param debug: Emit verbose logging when ``True``.
    :type debug: bool
    :returns: List of indices (``None`` entries are omitted).
    :rtype: list[int]
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
        # add IAM equivalents
        possible_locations.append(geo.ecoinvent_to_iam_location(activity[-1]))

        # Attempt to find the index in technosphere_index
        for loc in possible_locations:
            idx = technosphere_index.get((activity[0], activity[1], activity[2], loc))
            if idx is not None:
                indices.append(int(idx))
                logging.info(
                    f"Activity {activity} found at index {idx} using location {loc}."
                )
                break
        else:
            logger.warning(
                f"Activity {activity} not found in technosphere index. Skipping"
            )
            pass
    return indices


def add_lhv(variable, mapping) -> Union[dict, None]:
    """Fetch lower-heating-value metadata for a scenario variable if available.

    :param variable: Scenario variable name.
    :type variable: str
    :param mapping: Scenario-to-dataset mapping dictionary.
    :type mapping: dict
    :returns: LHV metadata dictionary or an empty dict when absent.
    :rtype: dict
    """
    if variable in mapping:
        for ds in mapping[variable]["dataset"]:
            if "lhv" in ds:
                return ds["lhv"]
    return {}


def fetch_indices(
    mapping: dict, regions: list, variables: list, technosphere_index: dict, geo: Geomap
) -> dict:
    """Derive technosphere indices for each variable across all regions.

    :param mapping: Scenario variable mapping with dataset descriptors.
    :type mapping: dict
    :param regions: IAM regions to evaluate.
    :type regions: list[str]
    :param variables: Scenario variables to map.
    :type variables: list[str]
    :param technosphere_index: Mapping from activity descriptors to indices.
    :type technosphere_index: dict
    :param geo: Geomap helper to resolve alternate locations.
    :type geo: premise.geomap.Geomap
    :returns: Nested dictionary of index metadata keyed by region and variable.
    :rtype: dict[str, dict[str, dict[str, Any]]]
    """

    # Pre-process mapping data to minimize repetitive data access
    activities_info = {}
    for variable in variables:
        try:

            activities_info[variable] = (
                mapping[variable]["dataset"][0]["name"],
                mapping[variable]["dataset"][0]["reference product"],
                mapping[variable]["dataset"][0]["unit"],
            )

        except IndexError:
            logging.error(
                f"Variable '{variable}' not found in mapping. Ensure it is correctly defined."
            )
            pass

    # Initialize dictionary to hold indices
    vars_idx = {}

    for region in regions:
        # Construct activities list for the current region
        activities = [
            (name, ref_product, unit, region)
            for name, ref_product, unit in activities_info.values()
        ]
        idxs = None
        # Use _get_activity_indices to fetch indices
        try:
            idxs = get_activity_indices(activities, technosphere_index, geo)
        except ValueError as e:
            logging.error(
                f"Error fetching indices for region {region}: {e}. "
                "Ensure that the activities and regions are correctly defined."
            )
            pass

        for variable in variables:
            if variable not in mapping:
                print(
                    f"Variable '{variable}' not found in mapping. Ensure it is correctly defined."
                )

        if idxs is not None:

            # Map variables to their indices and associated dataset information
            vars_idx[region] = {
                variable: {
                    "idx": idx,
                    "dataset": activities[i],
                    "lhv": add_lhv(variable, mapping) if variable in mapping else {},
                }
                for i, (variable, idx) in enumerate(zip(variables, idxs))
                if idx is not None
            }

        if len(variables) != len(idxs):
            logging.warning(f"Could not find all activities for region {region}.")

    return vars_idx


def get_all_indices(vars_info: dict) -> list[int]:
    """Collect technosphere indices from a variable-index mapping.

    :param vars_info: Mapping returned by :func:`fetch_indices`.
    :type vars_info: dict
    :returns: List of index integers present in the mapping.
    :rtype: list[int]
    """
    idx_list = []
    for region_data in vars_info.values():
        for variable_data in region_data.values():
            idx_list.append(variable_data["idx"])
    return idx_list


def fetch_inventories_locations(technosphere_indices: dict) -> List[str]:
    """Extract the unique locations referenced in technosphere indices.

    :param technosphere_indices: Mapping from activity tuples to indices.
    :type technosphere_indices: dict
    :returns: Sorted list of location strings.
    :rtype: list[str]
    """

    locations = list(set([act[3] for act in technosphere_indices]))
    logging.info(f"Unique locations in LCA database: {locations}")

    return locations


def csv_to_dict(filename: str) -> dict[int, tuple[str, ...]]:
    """Read a five-column CSV file into an index-to-activity mapping.

    :param filename: Path to the CSV file.
    :type filename: str | pathlib.Path
    :returns: Dictionary mapping integer indices to activity tuples.
    :rtype: dict[int, tuple[str, str, str, str]]
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
    technosphere_indices: dict, classifications: dict, reverse_classifications: dict
) -> [list, list, list]:
    """Identify technosphere activities missing from the classification mapping.

    :param technosphere_indices: Activities present in the technosphere matrix.
    :type technosphere_indices: dict
    :param classifications: Known classification mapping.
    :type classifications: dict
    :param reverse_classifications: Known reverse classification mapping.
    :type reverse_classifications: dict
    :returns: List of missing activity descriptors.
    :rtype: list[list[str]]
    """
    missing_classifications = []
    for act in technosphere_indices:
        if act[:2] not in classifications:
            missing_classifications.append(list(act[:2]))

            # we add them to `undefined`
            classifications[act[:2]] = "undefined"
            reverse_classifications["undefined"].append(act[:2])

    if missing_classifications:
        with open("missing_classifications.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(missing_classifications)

    return missing_classifications, classifications, reverse_classifications


def _group_technosphere_indices(
    technosphere_indices: dict,
    group_by,
    group_values: list,
    mapping: dict = None,
) -> dict:
    """Group technosphere indices by an arbitrary classifier function.

    :param technosphere_indices: Mapping from activity descriptors to indices.
    :type technosphere_indices: dict
    :param group_by: Callable returning a group label for each activity tuple.
    :type group_by: collections.abc.Callable
    :param group_values: Ordered list of expected group labels.
    :type group_values: list
    :param mapping: Optional aggregation mapping applied to group labels.
    :type mapping: dict | None
    :returns: Ordered dictionary mapping group labels to index lists.
    :rtype: collections.OrderedDict[str, list[int]]
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
    """Load hierarchical filter definitions from a YAML file.

    :param file_path: Path to the YAML file.
    :type file_path: pathlib.Path
    :returns: Parsed filter dictionary.
    :rtype: dict
    """
    with open(file_path, "r") as file:
        filters = yaml.safe_load(file)

    return filters


def gather_filters(current_level: Dict, combined_filters: Dict[str, Set[str]]) -> None:
    """Recursively merge filter criteria from nested dictionaries.

    :param current_level: Current branch of the filter hierarchy.
    :type current_level: dict
    :param combined_filters: Aggregated include/exclude sets being built.
    :type combined_filters: dict[str, set[str]]
    :returns: ``None`` (updates ``combined_filters`` in place).
    :rtype: None
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
) -> tuple[dict[str, set[Any]], dict[str, set[Any]]]:
    """Collect include/exclude patterns for multiple classification paths.

    :param filters: Parsed filter hierarchy from YAML.
    :type filters: dict
    :param paths: List of hierarchy paths to combine.
    :type paths: list[list[str]]
    :returns: Tuple of (combined filters, exception filters).
    :rtype: tuple[dict[str, list[str]], dict[str, list[str]]]
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
) -> tuple[
    list[Any],
    list[Any],
    dict[tuple[str, ...], set[Any]],
    dict[tuple[str, ...], set[Any]],
]:
    """Apply include/exclude filters to technosphere activities.

    :param technosphere_inds: Mapping from activity descriptors to indices.
    :type technosphere_inds: dict[tuple[str, str, str, str], int]
    :param filters: Combined filter lists produced by :func:`get_combined_filters`.
    :type filters: dict[str, list[str]]
    :param exceptions: Exception filters overriding the main filters.
    :type exceptions: dict[str, list[str]]
    :param paths: Classification paths used to categorize filtered names.
    :type paths: list[list[str]]
    :returns: Tuple containing filtered indices, exception indices, and categorized name sets.
    :rtype: tuple[list[int], list[int], dict[tuple[str, ...], set[str]], dict[tuple[str, ...], set[str]]]
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
    """Temporarily replace :func:`warnings.showwarning` to suppress matching warnings."""

    def __init__(self, ignore_message):
        """Initialize a warning filter that suppresses messages containing ``ignore_message``.

        :param ignore_message: Substring of warnings to silence.
        :type ignore_message: str
        """

        self.ignore_message = ignore_message

    def __enter__(self):
        """Activate the warning filter."""

        self.original_showwarning = warnings.showwarning
        warnings.showwarning = self.custom_showwarning

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the original warning handler."""

        warnings.showwarning = self.original_showwarning

    def custom_showwarning(
        self, message, category, filename, lineno, file=None, line=None
    ):
        """Proxy :func:`warnings.showwarning`, dropping messages that match the filter."""

        if self.ignore_message not in str(message):
            self.original_showwarning(message, category, filename, lineno, file, line)


def _get_mapping(data) -> dict:
    """Read the scenario-to-dataset mapping resource from a datapackage.

    :param data: Loaded datapackage object.
    :type data: datapackage.DataPackage
    :returns: Mapping dictionary parsed from ``mapping`` resource.
    :rtype: dict
    """
    return yaml.safe_load(data.get_resource("mapping").raw_read())


def _read_datapackage(datapackage: str) -> DataPackage:
    """Load a datapackage descriptor from disk.

    :param datapackage: Path to ``datapackage.json`` or zipped archive.
    :type datapackage: str | pathlib.Path
    :returns: DataPackage instance ready for validation.
    :rtype: datapackage.DataPackage
    """

    return DataPackage(datapackage)
