"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

from __future__ import annotations
import logging
import datetime
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import sparse as sp
import xarray as xr
import yaml

from .data_validation import validate_datapackage
from .filesystem_constants import DATA_DIR, USER_LOGS_DIR
from .lca import _calculate_year, get_lca_matrices
from .lcia import get_lcia_method_names
from .stats import log_mc_parameters_to_excel
from .subshares import generate_samples
from .utils import (
    _get_mapping,
    _read_datapackage,
    clean_cache_directory,
    create_lca_results_array,
    display_results,
    export_results_to_parquet,
    fetch_inventories_locations,
    harmonize_units,
    load_classifications,
    load_mapping,
    load_units_conversion,
    resize_scenario_data,
)

logger = logging.getLogger(__name__)


def _fill_in_result_array(
    coords: tuple,
    result: dict,
    use_distributions: int,
    shares: [None, dict],
    methods: list,
) -> np.ndarray:
    """
    Fill in the result array with the results from each region.
    The result array has the following dimensions:
    - iterations (if use_distributions > 0)
    - variables
    - regions
    - locations
    - methods

    :param coords: tuple. (model, scenario, year)
    :param result: dict. The result from _calculate_year function.
    :param use_distributions: int. If > 0, use distributions.
    :param shares: dict or None. The shares of sub-technologies.
    :param methods: list. The LCIA methods.
    :return: np.ndarray. The result array.

    """

    def _load_array(filepath):
        if len(filepath) == 1:
            if Path(filepath[0]).suffix == ".npy":
                return np.load(filepath[0])
            elif Path(filepath[0]).suffix == ".pkl":
                with open(filepath[0], "rb") as f:
                    return pickle.load(f)
            else:
                return sp.load_npz(filepath[0]).todense()
        else:
            if Path(filepath[0]).suffix == ".npy":
                return np.stack([np.load(f) for f in filepath], axis=-1)
            return np.stack([sp.load_npz(f).todense() for f in filepath], axis=-1)

    # Pre-loading data from disk if possible
    iteration_results = {
        region: _load_array(data["iterations_results"])
        for region, data in result.items()
    }

    model, scenario, year = coords

    results = []

    for region, data_ in result.items():

        array = iteration_results[region]

        if use_distributions > 0:
            array = np.quantile(
                array, [0.05, 0.5, 0.95], method="closest_observation", axis=-1
            )
            array = array.transpose(3, 1, 4, 2, 0)
        else:
            array = array.transpose(2, 0, 3, 1)

        results.append(array)

    if use_distributions > 0:
        uncertainty_parameters = {
            region: _load_array(data["uncertainty_params"])
            for region, data in result.items()
        }

        uncertainty_values = {
            region: _load_array(data["iterations_param_vals"])
            for region, data in result.items()
        }

        technosphere_indices = {
            region: _load_array(data["technosphere_indices"])
            for region, data in result.items()
        }

        log_mc_parameters_to_excel(
            model=model,
            scenario=scenario,
            year=year,
            methods=methods,
            result=result,
            uncertainty_parameters=uncertainty_parameters,
            uncertainty_values=uncertainty_values,
            technosphere_indices=technosphere_indices,
            iteration_results=iteration_results,
            shares=shares,
        )

    return np.stack(results, axis=2)


class Pathways:
    """The Pathways class reads in a datapackage that contains scenario data,
    mapping between scenario variables and LCA datasets, and LCA matrices.

    :param datapackage: Path to the datapackage.zip file.
    :type datapackage: str

    """

    def __init__(
        self,
        datapackage,
        geography_mapping: [dict, str] = None,
        activities_mapping: [dict, str] = None,
        ecoinvent_version: str = "3.11",
        debug=True,
    ):
        """
        Initialize the Pathways class.

        :param datapackage: Path to the datapackage.zip file.
        :type datapackage: str
        :param geography_mapping: Optional; path to a YAML file or a dictionary that maps ge
            geographies to higher-level regions.
        :type geography_mapping: dict or str, default is None
        :param activities_mapping: Optional; path to a YAML file or a dictionary that maps
            activities to higher-level classifications.
        :type activities_mapping: dict or str, default is None
        :param ecoinvent_version: Version of the ecoinvent database to use. Default is "3.11".
        :type ecoinvent_version: str, default is "3.11"
        :param debug: If True, enable debug logging. Default is True.
        :type debug: bool, default is True

        """
        self.datapackage = datapackage
        self.data, dataframe, self.filepaths = validate_datapackage(
            _read_datapackage(datapackage)
        )
        self.mapping = _get_mapping(self.data)
        self.ei_version = ecoinvent_version

        self.debug = debug
        self.scenarios = self._get_scenarios(dataframe)
        self.classifications = load_classifications()

        if self.data.get_resource("classifications"):
            self.classifications.update(
                yaml.full_load(self.data.get_resource("classifications").raw_read())
            )

        self.lca_results = None
        self.lcia_methods = get_lcia_method_names(self.ei_version)
        self.units = load_units_conversion()
        self.lcia_matrix = None

        # a mapping of geographies can be added
        # to aggregate locations to a higher level
        # e.g., from countries to regions
        if geography_mapping:
            self.geography_mapping = load_mapping(geography_mapping)
        else:
            self.geography_mapping = None

        if activities_mapping:
            mapping = load_mapping(activities_mapping)
            for k, v in self.classifications.items():
                if v in mapping:
                    self.classifications[k] = mapping[v]

        # create a reverse mapping
        self.reverse_classifications = defaultdict(list)
        for k, v in self.classifications.items():
            self.reverse_classifications[v].append(k)

        # clean cache directory
        clean_cache_directory()

        if self.debug:
            logging.info("#" * 600)
            logging.info(f"Pathways initialized with datapackage: {datapackage}")
            print(f"Log file: {USER_LOGS_DIR / 'pathways.log'}")

    def _get_scenarios(self, scenario_data: pd.DataFrame) -> xr.DataArray:
        """
        Load scenarios from filepaths as pandas DataFrame.
        Concatenate them into an xarray DataArray.
        :param scenario_data: pd.DataFrame
        :return: xr.DataArray
        """

        # check if all variables in mapping are in scenario_data
        for var in self.mapping:
            if var not in scenario_data["variables"].values:
                if self.debug:
                    logging.warning(f"Variable {var} not found in scenario data.")

        # remove rows which do not have a value under the `variable`
        # column that correspond to any value in self.mapping for `scenario variable`

        scenario_data = scenario_data[
            scenario_data["variables"].isin(list(self.mapping.keys()))
        ]

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

        # Add units
        lookup = (
            scenario_data.drop_duplicates("variables")  # in case of duplicates
            .set_index("variables")["unit"]
            .to_dict()
        )

        units = {str(v): lookup.get(str(v)) for v in data.coords["variables"].values}

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
        demand_cutoff: float = 1e-3,
        use_distributions: int = 0,
        subshares: bool = False,
        remove_uncertainty: bool = False,
        seed: int = 0,
        multiprocessing: bool = True,
        double_accounting: Optional[List[str]] = None,
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
        :param demand_cutoff: Float. If the total demand for a given variable is less than this value, the variable is skipped.
        :type demand_cutoff: float, default is 1e-3
        :param use_distributions: Integer. If non-zero, use distributions for LCA calculations.
        :type use_distributions: int, default is 0
        :param subshares: Boolean. If True, calculate subshares.
        :type subshares: bool, default is False
        :param remove_uncertainty: Boolean. If True, remove uncertainty from inventory exchanges.
        :type remove_uncertainty: bool, default is False
        :param seed: Integer. Seed for random number generator.
        :type seed: int, default is 0
        :param double_accounting: List. List of variables for which double accounting processing should be performed.
        :type double_accounting: Optional[List[str]], default is None
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
            k: self.mapping[k] for k in self.scenarios.coords["variables"].values
        }

        try:
            _, technosphere_index, _, uncertain_parameters, _ = get_lca_matrices(
                filepaths=self.filepaths,
                model=models[0],
                scenario=scenarios[0],
                year=years[0],
            )
        except Exception as e:
            logging.error(f"Error retrieving LCA matrices: {str(e)}")
            return

        # Create xarray for storing LCA results if not already present
        if self.lca_results is None:
            locations = fetch_inventories_locations(technosphere_index)

            # if geography mapping is provided, aggregate locations
            if self.geography_mapping:
                locations = list(set(list(self.geography_mapping.values())))
            else:
                self.geography_mapping = {loc: loc for loc in locations}

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
        if subshares is True:
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
                        self.geography_mapping,
                        self.debug,
                        use_distributions,
                        shares,
                        uncertain_parameters,
                        remove_uncertainty,
                        seed,
                        double_accounting,
                    )
                    for year in years
                ]

                if multiprocessing:
                    # Process each region in parallel
                    with Pool(cpu_count(), maxtasksperchild=1000) as p:
                        # store the results as a dictionary with years as keys
                        results.update(
                            {
                                (model, scenario, year): result
                                for year, result in zip(
                                    years, p.map(_calculate_year, args)
                                )
                            }
                        )
                else:
                    for arg in args:
                        results[(arg[0], arg[1], arg[2])] = _calculate_year(arg)

        # remove None values in results
        results = {k: v for k, v in results.items() if v is not None}

        if multiprocessing:
            with Pool(cpu_count(), maxtasksperchild=1000) as p:
                args = [
                    (
                        coords,
                        result,
                        use_distributions,
                        shares,
                        methods,
                    )
                    for coords, result in results.items()
                ]

                r = p.starmap(_fill_in_result_array, args)

                for c, coord in enumerate([c[0] for c in args]):
                    model, scenario, year = coord
                    self.lca_results.loc[
                        dict(
                            model=model,
                            scenario=scenario,
                            year=year,
                        )
                    ] = r[c]
        else:
            for coords, values in results.items():
                model, scenario, year = coords

                logging.info(
                    f"Variables in lca_results: {self.lca_results.coords['variable'].values}"
                )

                self.lca_results.loc[
                    dict(
                        model=model,
                        scenario=scenario,
                        year=year,
                    )
                ] = _fill_in_result_array(
                    coords,
                    values,
                    use_distributions,
                    shares,
                    methods,
                )

    def aggregate_results(self, cutoff: float = 0.001, interpolate: bool = False):
        """
        Aggregate the LCA results by removing values below a certain cutoff.

        :param cutoff: float. The cutoff value below which results will be removed.
        :param interpolate: bool. If True, interpolate missing values.
        :return: None
        """
        self.lca_results = display_results(
            self.lca_results, cutoff=cutoff, interpolate=interpolate
        )

    def export_results(self, filename: str = None) -> str:
        """
        Export the non-zero LCA results to a compressed parquet file.
        :param filename: str. The name of the file to save the results.
        :return: None
        """
        return export_results_to_parquet(self.lca_results, filename)
