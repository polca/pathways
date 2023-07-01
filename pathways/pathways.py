"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import json
import sys

import pandas as pd
import yaml
from datapackage import DataPackage
from pathlib import Path
from csv import reader
import csv
import numpy as np
from scipy import sparse
from .lcia import fill_characterization_factors_matrix, get_lcia_method_names
from .utils import load_classifications, load_units_conversion, display_results, create_lca_results_array
from .lca import solve_inventory, create_demand_vector, get_lca_matrices
import xarray as xr
from collections import defaultdict
from premise.geomap import Geomap
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple, Union, Optional

# if pypardiso is installed, use it
try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

from .data_validation import validate_datapackage


def csv_to_dict(filename):
    output_dict = {}

    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            # Making sure there are at least 5 items in the row
            if len(row) >= 5:
                # The first four items are the key, the fifth item is the value
                key = tuple(row[:4])
                value = row[4]
                output_dict[int(value)] = key

    return output_dict


def get_visible_files(path):
    return [file for file in Path(path).iterdir() if not file.name.startswith(".")]


def _get_activity_indices(
    activities: List[Tuple[str, str, str, str]],
    A_index: Dict[Tuple[str, str, str, str], int],
    geo: Any,
    region: str,
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
    for activity in activities:
        # Try to find the index for the specific region
        idx = A_index.get((activity[0], activity[1], activity[2], region))
        if idx is not None:
            indices.append(int(idx))  # If found, add to the list
        else:
            # If not found, look for the activity in possible locations
            possible_locations = geo.iam_to_ecoinvent_location(activity[-1])

            for loc in possible_locations:
                idx = A_index.get((activity[0], activity[1], activity[2], loc))
                if idx is not None:
                    indices.append(int(idx))  # If found, add to the list
                    break  # Exit the loop as the index was found
            else:
                # If still not found, try some default locations
                for default_loc in ["RoW", "GLO", "RER", "CH"]:
                    idx = A_index.get(
                        (activity[0], activity[1], activity[2], default_loc)
                    )
                    if idx is not None:
                        indices.append(int(idx))  # If found, add to the list
                        break  # Exit the loop as the index was found

    return indices  # Return the list of indices


def get_unit_conversion_factors(
    scenarios: Any,
    mapping: Dict,
    units_map: Dict[str, Dict[str, float]],
    activities: List[Tuple[Any, Any, Any, Any]],
) -> np.ndarray:
    """
    Generate an array of unit conversion factors based on the scenarios, mapping, unit map and activities provided.

    :param scenarios: Object containing scenario attributes, including a "units" attribute which is a dictionary where keys are scenario variables and values are their units.
    :type scenarios: object

    :param mapping: Dictionary mapping keys to scenario variables. The keys are the indices or identifiers of activities.
    :type mapping: Dict

    :param units_map: Dictionary mapping unit names to their conversion factors. The keys are units and the values are dictionaries mapping activity types to conversion factors.
    :type units_map: Dict[str, Dict[str, float]]

    :param activities: List of tuples, where each tuple represents an activity and the third element of the tuple corresponds to the type of activity.
    :type activities: List[Tuple[Any, Any, str]]

    :return: An array of unit conversion factors.
    :rtype: numpy.ndarray

    This function uses the given mapping to extract the units of each scenario variable, then finds the conversion factor for each unit from the units_map based on the third element of each activity tuple. The conversion factors are returned as a 1D numpy array.
    """

    # Extract units for each activity from the scenarios object using the mapping
    units = [scenarios.attrs["units"][mapping[x]["scenario variable"]] for x in mapping]

    # Initialize a numpy array of ones to hold the conversion factors
    unit_conversion = np.ones(len(units))

    # For each unit, find the conversion factor from the units_map
    # using the third element of the corresponding activity tuple
    for i, unit in enumerate(units):
        unit_conversion[i] = units_map[unit][activities[i][2]]

    return unit_conversion


def process_region(data: Tuple) -> Union[None, Dict[str, Any]]:
    """
    Process the LCA for a specific region based on the provided data.

    This function extracts various data from the input tuple, then computes a demand vector for the region and solves
    the LCA. The results are returned as a dictionary.

    :param data: A tuple containing multiple parameters required for the LCA processing. The parameters include model,
                 scenario, year, region, scenarios object, mapping, units map, technosphere and biosphere matrices (A, B),
                 indices for matrices A and B, LCIA matrix, reverse classifications, LCA results coordinates, and geo.
    :type data: Tuple

    :return: A dictionary containing processed LCA data for the given region, or None if the total demand for the
             region is zero. The dictionary keys are "act_category", "impact_category", "year", "region", "D",
             "scenario", and "model".
    :rtype: Union[None, Dict[str, Any]]

    :raises AssertionError: If the number of activities indices does not match the number of activities.
    """

    # Unpack the data tuple into individual variables
    (
        model,
        scenario,
        year,
        region,
        scenarios,
        mapping,
        units_map,
        A,
        B,
        A_index,
        B_index,
        lcia_matrix,
        reverse_classifications,
        lca_results_coords,
        geo,
    ) = data

    # Fetch the demand for the given region, model, pathway, and year
    demand = scenarios.sel(
        region=region,
        model=model,
        pathway=scenario,
        year=year,
    )

    # If the total demand is zero, return None
    if demand.sum() == 0:
        return None

    # Create a list of activities, with each activity represented by a tuple of four elements
    activities = [
        (
            mapping[x]["dataset"]["name"],
            mapping[x]["dataset"]["reference product"],
            mapping[x]["dataset"]["unit"],
            region,
        )
        for x in demand.coords["variable"].values
    ]

    # Fetch the indices for the given activities in the technosphere matrix
    activities_idx = _get_activity_indices(activities, A_index, geo, region)

    # Ensure that the number of activities indices matches the number of activities
    assert len(activities_idx) == len(activities)

    # Compute the unit conversion vector for the given activities
    unit_vector = get_unit_conversion_factors(scenarios, mapping, units_map, activities)

    # Create the demand vector
    f = create_demand_vector(activities_idx, A, demand, unit_vector)

    print("f", f.shape, f.sum())
    print("A", A.shape)
    print("B", B.shape)
    sys.stdout.flush()

    # Solve the LCA problem to get the LCIA scores
    D = solve_inventory(A, B, f, lcia_matrix, activities_idx)

    # Generate a list of activity indices for each activity category
    acts_idx = []
    for cat in lca_results_coords["act_category"].values:
        acts_idx.append(
            [int(A_index[a]) for a in reverse_classifications[cat] if a in A_index]
        )

    # Pad each list in acts_idx with -1 to make them all the same length
    max_len = max([len(x) for x in acts_idx])
    acts_idx = np.array(
        [np.pad(x, (0, max_len - len(x)), constant_values=-1) for x in acts_idx]
    )

    # Swap the axes of acts_idx to align with the dimensionality of D
    acts_idx = np.swapaxes(acts_idx, 0, 1)

    # Sum over the first axis of D, using acts_idx for advanced indexing
    D = D[acts_idx, ...].sum(axis=0)

    # Return a dictionary containing the processed LCA data for the given region
    return {
        "act_category": lca_results_coords["act_category"].values,
        "impact_category": lca_results_coords["impact_category"].values,
        "year": year,
        "region": region,
        "D": D,
        "scenario": scenario,
        "model": model,
    }


class Pathways:
    """The Pathways class reads in a datapackage that contains scenario data,
    mapping between scenario variables and LCA datasets, and LCA matrices.

    Parameters
    ----------
    datapackage : str
        Path to the datapackage.json file.
    """

    def __init__(self, datapackage):
        self.datapackage = datapackage
        self.data = validate_datapackage(self.read_datapackage())
        self.mapping = self.get_mapping()
        self.scenarios = self.get_scenarios()
        self.classifications = load_classifications()

        # create a reverse mapping
        self.reverse_classifications = defaultdict(list)
        for k, v in self.classifications.items():
            self.reverse_classifications[v].append(k)

        self.lca_results = None
        self.lcia_methods = get_lcia_method_names()
        self.units = load_units_conversion()
        self.lcia_matrix = None

    def read_datapackage(self) -> DataPackage:
        """Read the datapackage.json file.

        Returns
        -------
        dict
            The datapackage as a dictionary.
        """
        return DataPackage(self.datapackage)

    def get_mapping(self) -> dict:
        """
        Read the mapping file.
        It's a YAML file.
        :return: dict

        """
        return yaml.safe_load(self.data.get_resource("mapping").raw_read())

    def read_scenario_data(self, scenario):
        """Read the scenario data.

        Parameters
        ----------
        scenario : str
            Scenario name.

        Returns
        -------
        pd.DataFrame
            The scenario data as a pandas DataFrame.
        """
        filepath = self.data["scenarios"][scenario]["path"]
        # if CSV file
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath, index_col=0)
        else:
            # Excel file
            return pd.read_excel(filepath, index_col=0)

    def get_scenarios(self):
        """
        Load scenarios from filepaths as pandas DataFrame.
        Concatenate them into an xarray DataArray.
        """

        scenario_data = self.data.get_resource("scenarios").read()
        scenario_data = pd.DataFrame(
            scenario_data, columns=self.data.get_resource("scenarios").headers
        )

        # remove rows which do not have a value under the `variable`
        # column that correpsonds to any value in self.mapping for `scenario variable`
        scenario_data = scenario_data[
            scenario_data["variable"].isin(
                [item["scenario variable"] for item in self.mapping.values()]
            )
        ]

        # convert `year` column to integer
        scenario_data["year"] = scenario_data["year"].astype(int)

        # Convert to xarray DataArray
        data = (
            scenario_data.groupby(["model", "pathway", "variable", "region", "year"])[
                "value"
            ]
            .mean()
            .to_xarray()
        )

        # Add units
        units = {}
        for variable in data.coords["variable"].values:
            units[variable] = scenario_data[scenario_data["variable"] == variable].iloc[
                0
            ]["unit"]

        data.attrs["units"] = units

        # Replace variable names with values found in self.mapping
        new_names = []
        for variable in data.coords["variable"].values:
            for p_var, val in self.mapping.items():
                if val["scenario variable"] == variable:
                    new_names.append(p_var)
        data.coords["variable"] = new_names

        return data

    def calculate(
            self,
            methods: Optional[List[str]] = None,
            models: Optional[List[str]] = None,
            scenarios: Optional[List[str]] = None,
            regions: Optional[List[str]] = None,
            years: Optional[List[int]] = None
    ) -> None:
        """
        Calculate Life Cycle Assessment (LCA) results for given methods, models, scenarios, regions, and years.

        If no arguments are provided for methods, models, scenarios, regions, or years,
        the function will default to using all available values from the `scenarios` attribute.

        This function processes each combination of model, scenario, and year in parallel
        and stores the results in the `lca_results` attribute.

        :param methods: List of impact assessment methods. If None, all available methods will be used.
        :type methods: Optional[List[str]], default is None
        :param models: List of models. If None, all available models will be used.
        :type models: Optional[List[str]], default is None
        :param scenarios: List of scenarios. If None, all available scenarios will be used.
        :type scenarios: Optional[List[str]], default is None
        :param regions: List of regions. If None, all available regions will be used.
        :type regions: Optional[List[str]], default is None
        :param years: List of years. If None, all available years will be used.
        :type years: Optional[List[int]], default is None
        """

        # Initialize list to store activities not found in classifications
        missing_class = []

        # Set default values if arguments are not provided
        if methods is None:
            methods = get_lcia_method_names()
        if models is None:
            models = self.scenarios.coords["model"].values
        if scenarios is None:
            scenarios = self.scenarios.coords["pathway"].values
        if regions is None:
            regions = self.scenarios.coords["region"].values
        if years is None:
            years = self.scenarios.coords["year"].values

        # Create xarray for storing LCA results if not already present
        if self.lca_results is None:
            self.lca_results = create_lca_results_array(
                methods, years, regions, models, scenarios, self.classifications
            )

        # Iterate over each combination of model, scenario, and year
        for model in models:
            geo = Geomap(model=model)
            for scenario in scenarios:
                for year in years:

                    # Try to load LCA matrices for the given model, scenario, and year
                    try:
                        A, B, A_index, B_index = get_lca_matrices(
                            self.datapackage, model, scenario, year
                        )
                        B = np.asarray(B.todense())
                    except FileNotFoundError:
                        # If LCA matrices can't be loaded, skip to the next iteration
                        continue

                    # Fill characterization factor matrix for the given methods
                    self.lcia_matrix = fill_characterization_factors_matrix(
                        list(B_index.keys()), methods
                    )

                    # Check for activities not found in classifications
                    for act in A_index:
                        if act not in self.classifications:
                            missing_class.append(list(act))

                    # Prepare data for each region
                    data_for_regions = [
                        (
                            model,
                            scenario,
                            year,
                            region,
                            self.scenarios,
                            self.mapping,
                            self.units,
                            A,
                            B,
                            A_index,
                            B_index,
                            self.lcia_matrix,
                            self.reverse_classifications,
                            self.lca_results.coords,
                            geo,
                        )
                        for region in regions
                    ]

                    # Process each region in parallel
                    with Pool(cpu_count()) as p:
                        results = p.map(process_region, data_for_regions)

                    # Store results in the LCA results xarray
                    for result in results:
                        if result is not None:
                            self.lca_results.loc[
                                {
                                    "act_category": result["act_category"],
                                    "impact_category": result["impact_category"],
                                    "year": result["year"],
                                    "region": result["region"],
                                    "model": result["model"],
                                    "scenario": result["scenario"],
                                }
                            ] = result["D"]

    def display_results(self, cutoff: float = 0.01) -> xr.DataArray:

        return display_results(self.lca_results, cutoff=cutoff)

