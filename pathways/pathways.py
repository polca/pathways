"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import csv
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyprind
import xarray as xr
import yaml
from datapackage import DataPackage
from premise.geomap import Geomap

from . import DATA_DIR
from .lca import (
    characterize_inventory,
    create_demand_vector,
    get_lca_matrices,
    remove_double_counting,
    solve_inventory,
)
from .lcia import fill_characterization_factors_matrix, get_lcia_method_names
from .utils import (
    _get_activity_indices,
    create_lca_results_array,
    display_results,
    get_unit_conversion_factors,
    harmonize_units,
    load_classifications,
    load_units_conversion,
)

# if pypardiso is installed, use it
try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve

from .data_validation import validate_datapackage


def check_unclassified_activities(A_index, classifications):
    """
    Check if there are activities in the technosphere matrix that are not in the classifications.
    :param A:
    :param classifications:
    :return:
    """
    missing_classifications = []
    for act in A_index:
        if act not in classifications:
            missing_classifications.append(list(act))
    if missing_classifications:
        print(
            f"{len(missing_classifications)} activities are not found in the classifications. "
            "There are exported in the file 'missing_classifications.csv'."
        )
        with open("missing_classifications.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(missing_classifications)


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


def resize_scenario_data(
    scenario_data: xr.DataArray,
    model: List[str],
    scenario: List[str],
    region: List[str],
    year: List[int],
    variables: List[str],
):
    """
    Resize the scenario data to the given scenario, year, region, and variables.
    :param model:
    :param scenario_data:
    :param scenario:
    :param year:
    :param region:
    :param variables:
    :return:
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


def fetch_indices(mapping, regions, variables, A_index, geo):
    """
    Fetch the indices for the given activities in
    the technosphere matrix.
    :param variables:
    :param A_index:
    :param geo:
    :return:
    """

    vars_idx = {}

    for region in regions:

        activities = []

        for variable in variables:
            for dataset in mapping[variable]["dataset"]:
                activities.append(
                    (
                        dataset["name"],
                        dataset["reference product"],
                        dataset["unit"],
                        region,
                    )
                )

        if len(activities) != len(variables):
            print("Warning: mismatch between activities and variables.")
            print(f"Number of variables: {len(variables)}")
            print(f"Number of datasets: {len(activities)}")

        idxs = _get_activity_indices(activities, A_index, geo)

        # Fetch the indices for the given activities in
        # the technosphere matrix

        vars_idx[region] = {
            variable: {
                "idx": idx,
                "dataset": activity,
            }
            for variable, idx, activity in zip(variables, idxs, activities)
        }

    return vars_idx


def generate_A_indices(A_index, reverse_classifications, lca_results_coords):
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
    return np.swapaxes(acts_idx, 0, 1)


def process_region(data: Tuple) -> Union[None, Dict[str, Any]]:
    global impact_categories
    (
        model,
        scenario,
        year,
        region,
        variables,
        vars_idx,
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
        flows,
        demand_cutoff,
    ) = data

    # Generate a list of activity indices for each activity category
    category_idx = []
    for cat in lca_results_coords["act_category"].values:
        category_idx.append(
            [int(A_index[a]) for a in reverse_classifications[cat] if a in A_index]
        )

    act_categories = lca_results_coords["act_category"].values

    if lcia_matrix is not None:
        impact_categories = lca_results_coords["impact_category"].values
        target = np.zeros(
            (len(act_categories), len(list(vars_idx)), len(impact_categories))
        )
    else:
        if flows is not None:
            target = np.zeros((len(act_categories), len(list(vars_idx)), len(flows)))
        else:
            target = np.zeros((len(act_categories), len(list(vars_idx)), len(B_index)))

    for v, variable in enumerate(variables):
        idx, dataset = vars_idx[variable]["idx"], vars_idx[variable]["dataset"]

        # Compute the unit conversion vector for the given activities
        dataset_unit = dataset[2]
        unit_vector = get_unit_conversion_factors(
            scenarios.attrs["units"][variable],
            dataset_unit,
            units_map,
        )

        # Fetch the demand for the given region, model, pathway, and year
        demand = scenarios.sel(
            variables=variable,
            region=region,
            model=model,
            pathway=scenario,
            year=year,
        )

        # If the total demand is zero, return None
        if (
            demand
            / scenarios.sel(
                region=region,
                model=model,
                pathway=scenario,
                year=year,
            ).sum(dim="variables")
        ) < demand_cutoff:
            continue

        # Create the demand vector
        f = create_demand_vector([idx], A, demand, unit_vector)

        # Solve the inventory
        C = solve_inventory(A, B, f)

        if lcia_matrix is not None:
            if lcia_matrix.ndim != 2 or lcia_matrix.shape[0] != B.shape[1]:
                raise ValueError("Incompatible dimensions between B and lcia_matrix")

            # Solve the LCA problem to get the LCIA scores
            D = characterize_inventory(C, lcia_matrix)

            # Sum along the first axis of D to get final result
            D = D.sum(axis=1)

            acts_idx = generate_A_indices(
                A_index,
                reverse_classifications,
                lca_results_coords,
            )

            # Sum over the first axis of D, using acts_idx for advanced indexing
            target[:, v] = D[acts_idx, ...].sum(axis=0)

        else:
            # else, just sum the results of the inventory
            acts_idx = generate_A_indices(
                A_index,
                reverse_classifications,
                lca_results_coords,
            )
            if flows is not None:
                flows_idx = [int(B_index[f]) for f in flows]
                C = C[:, flows_idx]
                target[:, v] = C[acts_idx, ...].sum(axis=0)
            else:
                target[:, v] = C[acts_idx, ...].sum(axis=0)

    # Return a dictionary containing the processed LCA data for the given region
    def get_indices():
        if flows is not None:
            return [" - ".join(a) for a in flows]
        return [" - ".join(a) for a in list(B_index.keys())]

    return {
        "act_category": act_categories,
        "variable": list(vars_idx.keys()),
        "impact_category": (
            impact_categories if lcia_matrix is not None else get_indices()
        ),
        "year": year,
        "region": region,
        "D": target,
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
        self.data, dataframe = validate_datapackage(self.read_datapackage())
        self.mapping = self.get_mapping()
        self.mapping.update(self.get_final_energy_mapping())
        self.scenarios = self.get_scenarios(dataframe)
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

    def get_final_energy_mapping(self):
        """
        Read the final energy mapping file, which is an Excel file
        :return: dict
        """

        def create_dict_for_specific_model(row, model):
            # Construct the key from 'sector', 'variable', and 'fuel'
            key = f"{row['sector']}_{row['variable']}_{row['fuel']}"

            # Check if the specific model's scenario variable is available
            if pd.notna(row[model]):
                # Create the dictionary structure for this row for the specific model
                dict_structure = {
                    key: {
                        "dataset": {
                            "name": row["dataset name"],
                            "reference product": row["dataset reference product"],
                            "unit": row["unit"],
                        },
                        "scenario variable": row[model],
                    }
                }
                return dict_structure
            return None

        def create_dict_with_specific_model(dataframe, model):
            model_dict = {}
            for index, row in dataframe.iterrows():
                row_dict = create_dict_for_specific_model(row, model)
                if row_dict:
                    model_dict.update(row_dict)
            return model_dict

        # Read the Excel file
        df = pd.read_excel(
            DATA_DIR / "final_energy_mapping.xlsx",
        )
        model = self.data.descriptor["scenarios"][0]["name"].split(" - ")[0]

        return create_dict_with_specific_model(df, model)

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

    def get_scenarios(self, scenario_data: pd.DataFrame) -> xr.DataArray:
        """
        Load scenarios from filepaths as pandas DataFrame.
        Concatenate them into an xarray DataArray.
        """

        mapping_vars = [item["scenario variable"] for item in self.mapping.values()]

        # check if all variables in mapping are in scenario_data
        for var in mapping_vars:
            if var not in scenario_data["variables"].values:
                print(f"Variable {var} not found in scenario data.")

        # remove rows which do not have a value under the `variable`
        # column that correspond to any value in self.mapping for `scenario variable`

        scenario_data = scenario_data[scenario_data["variables"].isin(mapping_vars)]

        # convert `year` column to integer
        scenario_data.loc[:, "year"] = scenario_data["year"].astype(int)

        # Convert to xarray DataArray
        data = (
            scenario_data.groupby(["model", "pathway", "variables", "region", "year"])[
                "value"
            ]
            .mean()
            .to_xarray()
        )

        # convert values under "model" column to lower case
        data.coords["model"] = [x.lower() for x in data.coords["model"].values]

        # Replace variable names with values found in self.mapping
        new_names = []
        for variable in data.coords["variables"].values:
            for p_var, val in self.mapping.items():
                if val["scenario variable"] == variable:
                    new_names.append(p_var)
        data.coords["variables"] = new_names

        # Add units
        units = {}
        for variable in data.coords["variables"].values:
            units[variable] = scenario_data[
                scenario_data["variables"]
                == self.mapping[variable]["scenario variable"]
            ].iloc[0]["unit"]

        data.attrs["units"] = units

        return data

    def calculate(
        self,
        methods: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
        characterization: bool = True,
        flows: Optional[List[str]] = None,
        multiprocessing: bool = False,
        data_type: np.dtype = np.float64,
        demand_cutoff: float = 1e-3,
    ) -> None:
        """
        Calculate Life Cycle Assessment (LCA) results for given methods, models, scenarios, regions, and years.

        If no arguments are provided for methods, models, scenarios, regions, or years,
        the function will default to using all available values from the `scenarios` attribute.

        This function processes each combination of model, scenario, and year in parallel
        and stores the results in the `lca_results` attribute.

        :param characterization: Boolean. If True, calculate LCIA results. If False, calculate LCI results.
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
        :param variables: List of variables. If None, all available variables will be used.
        :type variables: Optional[List[str]], default is None
        :param flows: List of biosphere flows. If None, all available flows will be used.
        :type flows: Optional[List[str]], default is None
        :param multiprocessing: Boolean. If True, process each region in parallel.
        :type multiprocessing: bool, default is False
        :param data_type: Data type to use for storing LCA results.
        :type data_type: np.dtype, default is np.float64
        :param demand_cutoff: Float. If the total demand for a given variable is less than this value, the variable is skipped.
        :type demand_cutoff: float, default is 1e-3
        """

        self.scenarios = harmonize_units(self.scenarios, variables)

        # Initialize list to store activities not found in classifications
        missing_class = []

        if characterization is False:
            methods = None

        # Set default values if arguments are not provided
        if methods is None:
            if characterization is True:
                methods = get_lcia_method_names()
        if models is None:
            models = self.scenarios.coords["model"].values
            models = [m.lower() for m in models]
        if scenarios is None:
            scenarios = self.scenarios.coords["pathway"].values
        if regions is None:
            regions = self.scenarios.coords["region"].values
        if years is None:
            years = self.scenarios.coords["year"].values
        if variables is None:
            variables = self.scenarios.coords["variables"].values

        # resize self.scenarios array to fit the given arguments
        self.scenarios = resize_scenario_data(
            self.scenarios, models, scenarios, regions, years, variables
        )

        # refresh self.mapping,
        # remove keys that are not
        # in self.scenarios.variable.values
        self.mapping = {
            k: v
            for k, v in self.mapping.items()
            if k in self.scenarios.coords["variables"].values
        }

        # Iterate over each combination of model, scenario, and year
        for model in models:
            print(f"Calculating LCA results for {model}...")
            geo = Geomap(model=model)
            for scenario in scenarios:
                print(f"--- Calculating LCA results for {scenario}...")
                for year in years:
                    print(f"------ Calculating LCA results for {year}...")

                    # Try to load LCA matrices for the given model, scenario, and year
                    try:
                        A, B, A_index, B_index = get_lca_matrices(
                            self.datapackage, model, scenario, year, data_type=data_type
                        )
                        B = np.asarray(B.todense())

                    except FileNotFoundError:
                        # If LCA matrices can't be loaded, skip to the next iteration
                        print(
                            "LCA matrices not found for the given model, scenario, and year."
                        )
                        continue

                    # Fetch indices
                    vars_info = fetch_indices(
                        self.mapping, regions, variables, A_index, geo
                    )

                    # Remove contribution from activities in other activities
                    #A = remove_double_counting(A, vars_info)

                    # check unclassified activities
                    check_unclassified_activities(A_index, self.classifications)

                    # Create xarray for storing LCA results if not already present
                    if self.lca_results is None:
                        self.lca_results = create_lca_results_array(
                            methods,
                            B_index,
                            years,
                            regions,
                            models,
                            scenarios,
                            self.classifications,
                            self.mapping,
                            flows,
                        )

                    # Fill characterization factor matrix for the given methods
                    if characterization is True:
                        self.lcia_matrix = fill_characterization_factors_matrix(
                            list(B_index.keys()), methods
                        )

                    # Check for activities not found in classifications
                    for act in A_index:
                        if act not in self.classifications:
                            missing_class.append(list(act))

                    if multiprocessing is True:
                        # Prepare data for each region
                        data_for_regions = [
                            (
                                model,
                                scenario,
                                year,
                                region,
                                variables,
                                vars_info[region],
                                self.scenarios,
                                self.mapping,
                                self.units,
                                A,
                                B,
                                A_index,
                                B_index,
                                self.lcia_matrix if characterization else None,
                                self.reverse_classifications,
                                self.lca_results.coords,
                                flows,
                                demand_cutoff,
                            )
                            for region in regions
                        ]

                        # Process each region in parallel
                        with Pool(cpu_count()) as p:
                            results = p.map(process_region, data_for_regions)

                    else:
                        results = []
                        # use pyprind
                        bar = pyprind.ProgBar(len(regions))
                        for region in regions:
                            bar.update()
                            # Iterate over each region
                            results.append(
                                process_region(
                                    (
                                        model,
                                        scenario,
                                        year,
                                        region,
                                        variables,
                                        vars_info[region],
                                        self.scenarios,
                                        self.mapping,
                                        self.units,
                                        A,
                                        B,
                                        A_index,
                                        B_index,
                                        self.lcia_matrix if characterization else None,
                                        self.reverse_classifications,
                                        self.lca_results.coords,
                                        flows,
                                        demand_cutoff,
                                    )
                                )
                            )

                    # Store results in the LCA results xarray
                    for result in results:
                        if result is not None:
                            self.lca_results.loc[
                                {
                                    "act_category": result["act_category"],
                                    "variable": result["variable"],
                                    "impact_category": result["impact_category"],
                                    "year": result["year"],
                                    "region": result["region"],
                                    "model": result["model"],
                                    "scenario": result["scenario"],
                                }
                            ] = result["D"]

    def characterize_planetary_boundaries(
        self,
        models: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        variables: Optional[List[str]] = None,
    ):
        self.calculate(
            models=models,
            scenarios=scenarios,
            regions=regions,
            years=years,
            variables=variables,
            characterization=False,
        )

    def display_results(self, cutoff: float = 0.001) -> xr.DataArray:
        return display_results(self.lca_results, cutoff=cutoff)
