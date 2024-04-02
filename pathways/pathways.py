"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import csv
import logging
import uuid
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import bw2calc as bc
import numpy as np
import pandas as pd
import pyprind
import xarray as xr
import yaml
from bw2calc.monte_carlo import MonteCarloLCA
from datapackage import DataPackage
from numpy import dtype, ndarray
from premise.geomap import Geomap

from .data_validation import validate_datapackage
from .filesystem_constants import DATA_DIR, DIR_CACHED_DB
from .lca import (
    fill_characterization_factors_matrices,
    get_lca_matrices,
    remove_double_counting,
)
from .lcia import get_lcia_method_names
from .utils import (
    _get_activity_indices,
    clean_cache_directory,
    create_lca_results_array,
    display_results,
    get_unit_conversion_factors,
    harmonize_units,
    load_classifications,
    load_numpy_array_from_disk,
    load_units_conversion,
)

warnings.filterwarnings("ignore")


def check_unclassified_activities(technosphere_indices, classifications) -> List:
    """
    Check if there are activities in the technosphere matrix that are not in the classifications.
    :param A:
    :param classifications:
    :return:
    """
    missing_classifications = []
    for act in technosphere_indices:
        if act not in classifications:
            missing_classifications.append(list(act))
    if missing_classifications:

        with open("missing_classifications.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerows(missing_classifications)

    return missing_classifications


def csv_to_dict(filename):
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
            if len(mapping[variable]["dataset"]) > 1:
                print(
                    f"Warning: multiple datasets found for variable {variable}. Using the first one."
                )
            activities.append(
                (
                    mapping[variable]["dataset"][0]["name"],
                    mapping[variable]["dataset"][0]["reference product"],
                    mapping[variable]["dataset"][0]["unit"],
                    region,
                )
            )

        if len(activities) != len(variables):
            print("Warning: mismatch between activities and variables.")
            print(f"Number of variables: {len(variables)}: {variables}")
            print(f"Number of datasets: {len(activities)}: {activities}")
            logging.warning(
                f"Mismatch between activities and variables for region {region}."
                f"Number of variables: {len(variables)}."
                f"Number of datasets: {len(activities)}."
            )

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


def fetch_inventories_locations(A_index: Dict[str, Tuple[str, str, str]]) -> List[str]:
    """
    Fetch the locations of the inventories.
    :param A_index: Dictionary with the indices of the activities in the technosphere matrix.
    :return: List of locations.
    """

    locations = list(set([act[3] for act in A_index]))
    logging.info(f"Unique locations in LCA database: {locations}")

    return locations


def group_technosphere_indices_by_category(
    technosphere_index, reverse_classifications, lca_results_coords
) -> Tuple:
    # Generate a list of activity indices for each activity category
    acts_idx = []
    acts_dict = {}
    for cat in lca_results_coords["act_category"].values:
        x = [
            int(technosphere_index[a])
            for a in reverse_classifications[cat]
            if a in technosphere_index
        ]
        acts_idx.append(x)
        acts_dict[cat] = x

    # Pad each list in acts_idx with -1 to make them all the same length
    max_len = max([len(x) for x in acts_idx])
    acts_idx = np.array(
        [np.pad(x, (0, max_len - len(x)), constant_values=-1) for x in acts_idx]
    )

    # Swap the axes of acts_idx to align with the dimensionality of D
    return acts_idx, acts_dict, np.swapaxes(acts_idx, 0, 1)


def group_technosphere_indices_by_location(technosphere_index, locations) -> Tuple:
    """
    Group the technosphere indices by location.
    :param technosphere_index:
    :param locations:
    :return:
    """

    act_indices_list = []
    act_indices_dict = {}
    for loc in locations:
        x = [int(technosphere_index[a]) for a in technosphere_index if a[-1] == loc]

        act_indices_list.append(x)
        act_indices_dict[loc] = x

    # Pad each list in act_indices_list with -1 to make them all the same length
    max_len = max([len(x) for x in act_indices_list])
    act_indices_array = np.array(
        [np.pad(x, (0, max_len - len(x)), constant_values=-1) for x in act_indices_list]
    )

    return act_indices_list, act_indices_dict, act_indices_array


def process_region(data: Tuple) -> dict[str, ndarray[Any, dtype[Any]] | list[int]]:
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
        dp,
        A_index,
        B_index,
        reverse_classifications,
        lca_results_coords,
        demand_cutoff,
        locations,
        location_to_index,
        rev_A_index,
        lca,
        characterization_matrix,
        debug,
        use_distributions,
    ) = data

    variables_demand = {}
    d = []

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

        variables_demand[variable] = {
            "id": idx,
            "demand": demand.values * float(unit_vector),
        }

        lca.lci(demand={idx: demand.values * float(unit_vector)})

        if use_distributions == 0:
            characterized_inventory = (
                characterization_matrix @ lca.inventory
            ).toarray()

        else:
            # Use distributions for LCA calculations
            # next(lca) is a generator that yields the inventory matrix

            results = np.array(
                [
                    (
                        characterization_matrix
                        @ lca.inventory
                    ).toarray()
                    for _ in zip(range(use_distributions), lca)
                ]
            )

            # calculate quantiles along the first dimension
            characterized_inventory = np.quantile(results, [0.05, 0.5, 0.95], axis=0)

        # vars_info = fetch_indices(mapping, regions, variables, A_index, Geomap(model))
        # characterized_inventory = remove_double_counting(characterized_inventory=characterized_inventory,
        #                                                  vars_info=vars_idx,
        #                                                  activity_idx=idx
        #                                                  )
        d.append(characterized_inventory)

        if debug:
            logging.info(
                f"var.: {variable}, name: {dataset[0][:50]}, "
                f"ref.: {dataset[1]}, unit: {dataset[2][:50]}, idx: {idx},"
                f"loc.: {dataset[3]}, demand: {round(float(demand.values * float(unit_vector)), 2)}, "
                f"unit conv.: {unit_vector}, "
                f"impact: {round(characterized_inventory.sum(axis=-1) / (demand.values * float(unit_vector)), 3)}. "
            )

    id_array = uuid.uuid4()
    np.save(file=DIR_CACHED_DB / f"{id_array}.npy", arr=np.stack(d))

    del d

    # concatenate the list of sparse matrices and
    # add a third dimension and concatenate along it
    return {
        "id_array": id_array,
        "variables": {k: v["demand"] for k, v in variables_demand.items()},
    }


def _calculate_year(args):
    (
        model,
        scenario,
        year,
        regions,
        variables,
        methods,
        demand_cutoff,
        datapackage,
        mapping,
        units,
        lca_results,
        classifications,
        scenarios,
        reverse_classifications,
        debug,
        use_distributions,
    ) = args

    print(f"------ Calculating LCA results for {year}...")
    if debug:
        logging.info(
            f"############################### "
            f"{model}, {scenario}, {year} "
            f"###############################"
        )

    geo = Geomap(model=model)

    # Try to load LCA matrices for the given model, scenario, and year
    try:
        bw_datapackage, technosphere_indices, biosphere_indices = get_lca_matrices(
            datapackage, model, scenario, year
        )

    except FileNotFoundError:
        # If LCA matrices can't be loaded, skip to the next iteration
        if debug:
            logging.warning(f"Skipping {model}, {scenario}, {year}, as data not found.")
        return

    # Fetch indices
    vars_info = fetch_indices(mapping, regions, variables, technosphere_indices, geo)

    # Remove contribution from activities in other activities
    # A = remove_double_counting(A, vars_info)

    # check unclassified activities
    missing_classifications = check_unclassified_activities(
        technosphere_indices, classifications
    )

    if missing_classifications:
        if debug:
            logging.warning(
                f"{len(missing_classifications)} activities are not found in the classifications."
                "See missing_classifications.csv for more details."
            )

    # Initialize list to store activities not found in classifications
    missing_class = []

    # Check for activities not found in classifications
    for act in technosphere_indices:
        if act not in classifications:
            missing_class.append(list(act))

    results = {}

    locations = lca_results.coords["location"].values.tolist()
    location_to_index = {location: index for index, location in enumerate(locations)}
    reverse_technosphere_index = {int(v): k for k, v in technosphere_indices.items()}
    loc_idx = np.array(
        [
            location_to_index[act[-1]]
            for act in reverse_technosphere_index.values()
            if act[-1] in locations
        ]
    )

    acts_category_idx_list, acts_category_idx_dict, acts_category_idx_array = (
        group_technosphere_indices_by_category(
            technosphere_indices,
            reverse_classifications,
            lca_results.coords,
        )
    )

    acts_location_idx_list, acts_location_idx_dict, acts_location_idx_array = (
        group_technosphere_indices_by_location(
            technosphere_indices,
            locations,
        )
    )

    results["other"] = {
        "locations": locations,
        "location_to_index": location_to_index,
        "reverse_technosphere_index": reverse_technosphere_index,
        "activity_location_idx": loc_idx,
        "acts_category_idx_list": acts_category_idx_list,
        "acts_category_idx_dict": acts_category_idx_dict,
        "acts_category_idx_array": acts_category_idx_array,
        "acts_location_idx_list": acts_location_idx_list,
        "acts_location_idx_dict": acts_location_idx_dict,
        "acts_location_idx_array": acts_location_idx_array,
    }

    if use_distributions == 0:
        lca = bc.LCA(
            demand={0: 1},
            data_objs=[
                bw_datapackage,
            ],
        )
        lca.lci(factorize=True)
    else:
        lca = MonteCarloLCA(
            demand={0: 1},
            data_objs=[
                bw_datapackage,
            ],
            use_distributions=True,
        )
        lca.lci()

    characterization_matrix = fill_characterization_factors_matrices(
        biosphere_indices, methods, lca.dicts.biosphere, debug
    )

    if debug:
        logging.info(
            f"Characterization matrix created. Shape: {characterization_matrix.shape}"
        )

    bar = pyprind.ProgBar(len(regions))
    for region in regions:
        bar.update()
        # Iterate over each region
        results[region] = process_region(
            (
                model,
                scenario,
                year,
                region,
                variables,
                vars_info[region],
                scenarios,
                mapping,
                units,
                bw_datapackage,
                technosphere_indices,
                biosphere_indices,
                reverse_classifications,
                lca_results.coords,
                demand_cutoff,
                locations,
                location_to_index,
                reverse_technosphere_index,
                lca,
                characterization_matrix,
                debug,
                use_distributions,
            )
        )

    return results


class Pathways:
    """The Pathways class reads in a datapackage that contains scenario data,
    mapping between scenario variables and LCA datasets, and LCA matrices.

    Parameters
    ----------
    datapackage : str
        Path to the datapackage.json file.
    """

    def __init__(self, datapackage, debug=False):
        self.datapackage = datapackage
        self.data, dataframe = validate_datapackage(self.read_datapackage())
        self.mapping = self.get_mapping()
        self.mapping.update(self.get_final_energy_mapping())
        self.debug = debug
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

        clean_cache_directory()

        if self.debug:
            logging.basicConfig(
                level=logging.DEBUG,
                filename="pathways.log",  # Log file to save the entries
                filemode="a",  # Append to the log file if it exists, 'w' to overwrite
                format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            logging.info("#" * 600)
            logging.info(f"Pathways initialized with datapackage: {datapackage}")

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
                        "dataset": [
                            {
                                "name": row["dataset name"],
                                "reference product": row["dataset reference product"],
                                "unit": row["unit"],
                            }
                        ],
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
                if self.debug:
                    logging.warning(f"Variable {var} not found in scenario data.")

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
        demand_cutoff: float = 1e-3,
        use_distributions: int = 0,
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
        :param use_distributions: Integer. If non zero, use distributions for LCA calculations.
        """

        self.scenarios = harmonize_units(self.scenarios, variables)

        if characterization is False:
            methods = None

        # Set default values if arguments are not provided
        if methods is None and characterization is True:
            methods = get_lcia_method_names()
            if self.debug:
                logging.info(f"Using the following LCIA methods: {methods}")
        if models is None:
            models = self.scenarios.coords["model"].values
            models = [m.lower() for m in models]
            if self.debug:
                logging.info(f"Using the following models: {models}")
        if scenarios is None:
            scenarios = self.scenarios.coords["pathway"].values
            if self.debug:
                logging.info(f"Using the following scenarios: {scenarios}")
        if regions is None:
            regions = self.scenarios.coords["region"].values
            if self.debug:
                logging.info(f"Using the following regions: {regions}")
        if years is None:
            years = self.scenarios.coords["year"].values
            if self.debug:
                logging.info(f"Using the following years: {years}")
        if variables is None:
            variables = self.scenarios.coords["variables"].values
            variables = [str(v) for v in variables]
            if self.debug:
                logging.info(f"Using the following variables: {variables}")

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

        # Create xarray for storing LCA results if not already present
        if self.lca_results is None:
            _, technosphere_index, biosphere_index = get_lca_matrices(
                self.datapackage, models[0], scenarios[0], years[0]
            )
            locations = fetch_inventories_locations(technosphere_index)

            self.lca_results = create_lca_results_array(
                methods=methods,
                years=years,
                regions=regions,
                locations=locations,
                models=models,
                scenarios=scenarios,
                classifications=self.classifications,
                mapping=self.mapping,
                use_distributions=True if use_distributions > 0 else False,
            )

        # Iterate over each combination of model, scenario, and year
        results = {}
        for model in models:
            print(f"Calculating LCA results for {model}...")
            for scenario in scenarios:
                print(f"--- Calculating LCA results for {scenario}...")
                if multiprocessing:
                    args = [
                        (
                            model,
                            scenario,
                            year,
                            regions,
                            variables,
                            methods,
                            demand_cutoff,
                            self.datapackage,
                            self.mapping,
                            self.units,
                            self.lca_results,
                            self.classifications,
                            self.scenarios,
                            self.reverse_classifications,
                            self.debug,
                            use_distributions,
                        )
                        for year in years
                    ]
                    # Process each region in parallel
                    with Pool(cpu_count()) as p:
                        # store th results as a dictionary with years as keys
                        results.update(
                            {
                                (model, scenario, year): result
                                for year, result in zip(
                                    years, p.map(_calculate_year, args)
                                )
                            }
                        )
                else:
                    results = {
                        (model, scenario, year): _calculate_year(
                            (
                                model,
                                scenario,
                                year,
                                regions,
                                variables,
                                methods,
                                demand_cutoff,
                                self.datapackage,
                                self.mapping,
                                self.units,
                                self.lca_results,
                                self.classifications,
                                self.scenarios,
                                self.reverse_classifications,
                                self.debug,
                                use_distributions,
                            )
                        )
                        for year in years
                    }

        # remove None values in results
        results = {k: v for k, v in results.items() if v is not None}

        self.fill_in_result_array(results)

    def fill_in_result_array(self, results: dict):

        # Assuming DIR_CACHED_DB, results, and self.lca_results are already defined

        # Pre-loading data from disk if possible
        cached_data = {
            data["id_array"]: load_numpy_array_from_disk(
                DIR_CACHED_DB / f"{data['id_array']}.npy",
            )
            for coord, result in results.items()
            for region, data in result.items()
            if region != "other"
        }

        # use pyprint to display progress
        bar = pyprind.ProgBar(len(results))
        for coord, result in results.items():
            bar.update()
            model, scenario, year = coord
            acts_category_idx_dict = result["other"]["acts_category_idx_dict"]
            acts_location_idx_dict = result["other"]["acts_location_idx_dict"]

            for region, data in result.items():
                if region == "other":
                    continue

                id_array = data["id_array"]
                variables = data["variables"]

                d = cached_data[id_array]

                for cat, act_cat_idx in acts_category_idx_dict.items():
                    for loc, act_loc_idx in acts_location_idx_dict.items():
                        idx = np.intersect1d(act_cat_idx, act_loc_idx)
                        idx = idx[idx != -1]
                        summed_data = d[..., idx].sum(axis=-1)

                        if idx.size > 0:
                            try:
                                self.lca_results.loc[
                                    {
                                        "region": region,
                                        "model": model,
                                        "scenario": scenario,
                                        "year": year,
                                        "act_category": cat,
                                        "location": loc,
                                        "variable": list(variables.keys()),
                                    }
                                ] = summed_data

                            except:
                                print("summed data shape", summed_data.shape)
                                print(self.lca_results.loc[
                                          {
                                              "region": region,
                                              "model": model,
                                              "scenario": scenario,
                                              "year": year,
                                              "act_category": cat,
                                              "location": loc,
                                              "variable": list(variables.keys()),
                                          }
                                      ].shape)

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
