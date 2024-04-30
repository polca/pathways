"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import logging
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pyprind
import xarray as xr
import yaml
from datapackage import DataPackage

from .data_validation import validate_datapackage
from .filesystem_constants import DATA_DIR, DIR_CACHED_DB
from .lca import _calculate_year, get_lca_matrices
from .lcia import get_lcia_method_names
from .subshares import generate_samples
from .utils import (
    clean_cache_directory,
    create_lca_results_array,
    display_results,
    fetch_inventories_locations,
    harmonize_units,
    load_classifications,
    load_numpy_array_from_disk,
    load_units_conversion,
    resize_scenario_data,
)


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


class Pathways:
    """The Pathways class reads in a datapackage that contains scenario data,
    mapping between scenario variables and LCA datasets, and LCA matrices.

    :param datapackage: Path to the datapackage.zip file.
    :type datapackage: str

    """

    def __init__(self, datapackage, debug=False):
        self.datapackage = datapackage
        self.data, dataframe, self.filepaths = validate_datapackage(
            _read_datapackage(datapackage)
        )
        self.mapping = _get_mapping(self.data)
        try:
            self.mapping.update(self._get_final_energy_mapping())
        except KeyError:
            pass
        self.debug = debug
        self.scenarios = self._get_scenarios(dataframe)
        self.classifications = load_classifications()

        if self.data.get_resource("classifications"):
            self.classifications.update(
                yaml.full_load(self.data.get_resource("classifications").raw_read())
            )

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

    def _get_final_energy_mapping(self):
        """
        Read the final energy mapping file, which is an Excel file
        :return: dict
        """

        def create_dict_for_specific_model(
            row: pd.Series, model: str
        ) -> dict[Any, Any] | None:
            """
            Create a dictionary for a specific model from the row.
            :param row: dict
            :param model: str
            :return: dict
            """
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

        def create_dict_with_specific_model(
            dataframe: pd.DataFrame, model: str
        ) -> dict:
            """
            Create a dictionary for a specific model from the dataframe.
            :param dataframe: pandas.DataFrame
            :param model: str
            :return: dict
            """
            model_dict = {}
            for _, row in dataframe.iterrows():
                row_dict = create_dict_for_specific_model(row, model)
                if row_dict:
                    model_dict.update(row_dict)
            return model_dict

        # Read the Excel file
        mapping_dataframe = pd.read_excel(
            DATA_DIR / "final_energy_mapping.xlsx",
        )
        model = self.data.descriptor["scenarios"][0]["name"].split(" - ")[0]

        return create_dict_with_specific_model(mapping_dataframe, model)

    def _get_scenarios(self, scenario_data: pd.DataFrame) -> xr.DataArray:
        """
        Load scenarios from filepaths as pandas DataFrame.
        Concatenate them into an xarray DataArray.
        :param scenario_data: pd.DataFrame
        :return: xr.DataArray
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
        multiprocessing: bool = False,
        demand_cutoff: float = 1e-3,
        use_distributions: int = 0,
        subshares: bool = False,
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
        :param variables: List of variables. If None, all available variables will be used.
        :type variables: Optional[List[str]], default is None
        :param multiprocessing: Boolean. If True, process each region in parallel.
        :type multiprocessing: bool, default is False
        :param demand_cutoff: Float. If the total demand for a given variable is less than this value, the variable is skipped.
        :type demand_cutoff: float, default is 1e-3
        :param use_distributions: Integer. If non zero, use distributions for LCA calculations.
        :type use_distributions: int, default is 0
        :param subshares: Boolean. If True, calculate subshares.
        :type subshares: bool, default is False
        """

        self.scenarios = harmonize_units(self.scenarios, variables)

        # if no methods are provided, use all those available
        methods = methods or get_lcia_method_names()
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

        try:
            _, technosphere_index, _, uncertain_parameters = get_lca_matrices(
                self.filepaths, models[0], scenarios[0], years[0]
            )
        except Exception as e:
            logging.error(f"Error retrieving LCA matrices: {str(e)}")
            return

        # Create xarray for storing LCA results if not already present
        if self.lca_results is None:
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
                use_distributions=use_distributions > 0,
            )

        # generate share of sub-technologies
        shares = None
        if subshares:
            shares = generate_samples(
                years=self.scenarios.coords["year"].values.tolist(),
                iterations=use_distributions,
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
                            self.filepaths,
                            self.mapping,
                            self.units,
                            self.lca_results,
                            self.classifications,
                            self.scenarios,
                            self.reverse_classifications,
                            self.debug,
                            use_distributions,
                            shares or None,
                            uncertain_parameters,
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
                    results.update(
                        {
                            (model, scenario, year): _calculate_year(
                                (
                                    model,
                                    scenario,
                                    year,
                                    regions,
                                    variables,
                                    methods,
                                    demand_cutoff,
                                    self.filepaths,
                                    self.mapping,
                                    self.units,
                                    self.lca_results,
                                    self.classifications,
                                    self.scenarios,
                                    self.reverse_classifications,
                                    self.debug,
                                    use_distributions,
                                    shares or None,
                                    uncertain_parameters,
                                )
                            )
                            for year in years
                        }
                    )

        # remove None values in results
        results = {k: v for k, v in results.items() if v is not None}

        self._fill_in_result_array(results)


    def _fill_in_result_array(self, results: dict):

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
        progress_bar = pyprind.ProgBar(len(results))
        for coord, result in results.items():
            progress_bar.update()
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

                        if idx.size > 0:
                            summed_data = d[..., idx].sum(axis=-1)
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
                                # transpose quantile dimension to the penultimate dimension
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
                                ] = summed_data.transpose(0, 2, 1)

    def display_results(self, cutoff: float = 0.001) -> xr.DataArray:
        return display_results(self.lca_results, cutoff=cutoff)
