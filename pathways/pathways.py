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

from .data_validation import validate_datapackage
from .filesystem_constants import DATA_DIR, DIR_CACHED_DB, STATS_DIR, USER_LOGS_DIR
from .lca import _calculate_year, get_lca_matrices
from .lcia import get_lcia_method_names
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
    load_numpy_array_from_disk,
    load_units_conversion,
    resize_scenario_data,
)

from .stats import log_results_to_excel, log_subshares_to_excel, create_mapping_sheet, run_GSA_delta


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
                filename=USER_LOGS_DIR / "pathways.log",  # Log file to save the entries
                filemode="a",  # Append to the log file if it exists, 'w' to overwrite
                format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            logging.info("#" * 600)
            logging.info(f"Pathways initialized with datapackage: {datapackage}")
            print(f"Log file: {USER_LOGS_DIR / 'pathways.log'}")

    def _get_final_energy_mapping(self):
        """
        Read the final energy mapping file, which is an Excel file
        :return: dict
        """

        def create_dict_for_specific_model(row: pd.Series, model: str) -> [dict, None]:
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
            dataframe: pd.DataFrame, _model: str
        ) -> dict:
            """
            Create a dictionary for a specific model from the dataframe.
            :param dataframe: pandas.DataFrame
            :param _model: str. The specific model to create the dictionary for.
            :return: dict
            """
            model_dict = {}
            for _, row in dataframe.iterrows():
                row_dict = create_dict_for_specific_model(row, _model)
                if row_dict:
                    model_dict.update(row_dict)
            return model_dict

        # Read the Excel file
        mapping_dataframe = pd.read_excel(
            DATA_DIR / "final_energy_mapping.xlsx",
        )
        model = self.data.descriptor["scenarios"][0].split(" - ")[0].strip()

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
        demand_cutoff: float = 1e-3,
        use_distributions: int = 0,
        subshares: bool = False,
        remove_uncertainty: bool = False,
        multiprocessing: bool = True,
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
                        self.debug,
                        use_distributions,
                        shares,
                        uncertain_parameters,
                        remove_uncertainty,
                    )
                    for year in years
                ]

                if multiprocessing:
                    # Process each region in parallel
                    with Pool(cpu_count()) as p:
                        # store the results as a dictionary with years as keys
                        results.update(
                            {
                                (model, scenario, year): result
                                for year, result in zip(years, p.map(_calculate_year, args))
                            }
                        )
                else:
                    for arg in args:
                        results[(model, scenario, arg[2])] = _calculate_year(arg)

        # remove None values in results
        results = {k: v for k, v in results.items() if v is not None}

        self._fill_in_result_array(results, use_distributions, subshares, methods)

    def _fill_in_result_array(self, results: dict, use_distributions: int, subshares: bool, methods: list) -> None:

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

                            except ValueError:
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

            if use_distributions > 0:
                export_path = STATS_DIR / f"{model}_{scenario}_{year}.xlsx"

                # create Excel workbook using openpyxl
                with pd.ExcelWriter(export_path, engine="openpyxl") as writer:

                    df_tot_impacts = log_results_to_excel(
                        total_impacts_by_method=total_impacts_by_method,
                        methods=methods,
                    )

                    # create new worksheet called 'Total impacts'
                    writer.book.create_sheet("Total impacts")
                    # add df to it
                    df_tot_impacts.to_excel(writer, sheet_name="Total impacts", index=True)

                    if subshares:
                        df_shares = log_subshares_to_excel(
                            year=year,
                            shares=shares,
                            total_impacts_df=df_tot_impacts,
                        )

                        # create new worksheet called 'Subshares'
                        writer.book.create_sheet("Subshares")
                        # add df to it
                        df_shares.to_excel(writer, sheet_name="Subshares", index=True)

                    df_mapping = create_mapping_sheet(
                        filepaths=filepaths,
                        model=model,
                        scenario=scenario,
                        year=year,
                        parameter_keys=all_param_keys,
                    )

                    # create new worksheet called 'Mapping'
                    writer.book.create_sheet("Mapping")
                    # add df to it
                    df_mapping.to_excel(writer, sheet_name="Mapping", index=True)

                    df_GSA = run_GSA_delta(
                        methods=methods,
                        df_tot_impacts=df_tot_impacts,
                    )

                    # create new worksheet called 'Delta'
                    writer.book.create_sheet("Delta")
                    # add df to it
                    df_GSA.to_excel(writer, sheet_name="Delta", index=True)

                    print(
                        f"Delta analysis has been saved to the 'Delta' sheets in {export_path.resolve()}"
                    )

    def display_results(self, cutoff: float = 0.001) -> xr.DataArray:
        return display_results(self.lca_results, cutoff=cutoff)

    def export_results(self, filename: str = None) -> None:
        """
        Export the non-zero LCA results to a compressed parquet file.
        :param filename: str. The name of the file to save the results.
        :return: None
        """
        export_results_to_parquet(self.lca_results, filename)
