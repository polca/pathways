"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

from __future__ import annotations
import logging
import datetime
import csv
import io
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

from edges import get_available_methods

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
    """Load per-region result files and stack them into the master results tensor.

    :param coords: Tuple ``(model, scenario, year)`` describing the slice to fill.
    :type coords: tuple[str, str, int]
    :param result: Region-level data returned from :func:`_calculate_year`.
    :type result: dict[str, dict]
    :param use_distributions: Number of Monte Carlo iterations performed.
    :type use_distributions: int
    :param shares: Optional correlated share samples used for subtechnology modeling.
    :type shares: dict | None
    :param methods: Ordered LCIA method names.
    :type methods: list[str]
    :returns: Dense array ready to assign into ``self.lca_results``.
    :rtype: numpy.ndarray
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
    """Instantiate the Pathways workflow around a datapackage bundle.

    :param datapackage: Path to the zipped datapackage file.
    :type datapackage: str | pathlib.Path
    :param geography_mapping: Optional YAML path or mapping dict for regional aggregation.
    :type geography_mapping: str | dict | None
    :param activities_mapping: Optional YAML path or mapping dict for activity reclassification.
    :type activities_mapping: str | dict | None
    :param ecoinvent_version: Target ecoinvent release, used when selecting LCIA data.
    :type ecoinvent_version: str
    :param debug: Enable verbose logging when ``True``.
    :type debug: bool
    :raises FileNotFoundError: If the datapackage or classification resources cannot be read.
    """

    def __init__(
        self,
        datapackage,
        geography_mapping: [dict, str] = None,
        activities_mapping: [dict, str] = None,
        ecoinvent_version: str = "3.11",
        classification_system: str = "CPC",
        debug=True,
    ):
        """Initialize the workflow and load datapackage metadata.

        :param datapackage: Path to the datapackage archive or descriptor.
        :type datapackage: str | pathlib.Path
        :param geography_mapping: Optional aggregation mapping for scenario regions.
        :type geography_mapping: dict | str | None
        :param activities_mapping: Optional reclassification mapping for activities.
        :type activities_mapping: dict | str | None
        :param ecoinvent_version: Ecoinvent version string used when selecting LCIA data.
        :type ecoinvent_version: str
        :param classification_system: Ecoinvent classification system to use.
        :type classification_system: str
        :param debug: Emit verbose logging when ``True``.
        :type debug: bool
        """
        self.datapackage = datapackage
        self.data, dataframe, self.filepaths = validate_datapackage(
            _read_datapackage(datapackage)
        )
        self.mapping = _get_mapping(self.data)
        self.ei_version = ecoinvent_version

        self.debug = debug
        self.scenarios = self._get_scenarios(dataframe)
        self.classification_system = classification_system
        self._load_classifications()

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

        # create a reverse mapping
        self.reverse_classifications = defaultdict(list)
        for k, v in self.classifications.items():
            self.reverse_classifications[v].append(k)

        # add an `undefined` classification
        self.reverse_classifications["undefined"] = []

        # clean cache directory
        clean_cache_directory()

        if self.debug:
            logging.info("#" * 600)
            logging.info(f"Pathways initialized with datapackage: {datapackage}")
            print(f"Log file: {USER_LOGS_DIR / 'pathways.log'}")

    def _load_classifications(self):

        # final structure: {(name, reference product): "code for chosen system"}
        self.classifications = {}

        try:
            resource = self.data.get_resource("classifications")
        except Exception:
            print(
                "[CLASSIFICATIONS] No 'classifications' resource found in datapackage."
            )
            resource = None

        if resource:
            path = resource.descriptor.get("path", "").lower()
            raw = resource.raw_read()

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            if path.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(raw))

                n_rows = 0
                n_used = 0

                for row in reader:
                    n_rows += 1
                    name = (row.get("name") or "").strip()
                    ref = (row.get("reference product") or "").strip()

                    system = (
                        row.get("classification_system") or row.get("system") or ""
                    ).strip()
                    code = (
                        row.get("classification_code") or row.get("code") or ""
                    ).strip()

                    if not name or not ref or not system or not code:
                        continue

                    # Only keep the chosen classification system, e.g. "CPC"
                    if system != self.classification_system:
                        continue

                    key = (name, ref)

                    self.classifications[key] = code
                    n_used += 1

        fallback = load_classifications()  # dict[(name, ref) -> list[(system, code)]]

        added_keys = 0
        skipped_no_match = 0

        for key, cls_list in fallback.items():
            if key in self.classifications:
                # datapackage already gave a code for this key in chosen system
                continue

            # find a code for the chosen system in the fallback list
            code_for_system = None
            for system, code in cls_list:
                if system == self.classification_system:
                    code_for_system = code
                    break

            if code_for_system is None:
                skipped_no_match += 1
                continue

            self.classifications[key] = code_for_system
            added_keys += 1

    def _get_scenarios(self, scenario_data: pd.DataFrame) -> xr.DataArray:
        """Convert the datapackage scenario table into a harmonized ``xarray`` object.

        :param scenario_data: Scenario observations from ``scenario_data`` resource.
        :type scenario_data: pandas.DataFrame
        :returns: Multi-dimensional array indexed by model, pathway, variable, region, and year.
        :rtype: xarray.DataArray
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
        edges_methods: Optional[List[str]] = None,
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
        """Run LCA calculations across selected models, pathways, regions, and years.

        :param methods: Impact assessment methods to report; defaults to all available.
        :type methods: list[str] | None
        :param models: IAM models to include.
        :type models: list[str] | None
        :param scenarios: Pathways to include.
        :type scenarios: list[str] | None
        :param regions: IAM regions to include.
        :type regions: list[str] | None
        :param years: Simulation years to analyze.
        :type years: list[int] | None
        :param variables: Scenario variables supplying demand.
        :type variables: list[str] | None
        :param demand_cutoff: Minimum total demand required to run a functional unit.
        :type demand_cutoff: float
        :param use_distributions: Number of Monte Carlo iterations; ``0`` performs deterministic runs.
        :type use_distributions: int
        :param subshares: Whether to sample sub-technology market share distributions.
        :type subshares: bool
        :param remove_uncertainty: Replace exchange uncertainty parameters with deterministic values.
        :type remove_uncertainty: bool
        :param seed: Random seed forwarded to ``bw2calc`` when sampling.
        :type seed: int
        :param multiprocessing: Whether to parallelize over years using ``multiprocessing.Pool``.
        :type multiprocessing: bool
        :param double_accounting: Optional activity filters for double-accounting diagnostics.
        :type double_accounting: list[str] | None
        :returns: ``None`` (results are stored on ``self.lca_results``).
        :rtype: None
        """

        self.scenarios = harmonize_units(self.scenarios, variables)

        if methods:
            available_methods = get_lcia_method_names()
            for m in methods:
                if m not in available_methods:
                    raise ValueError(f"LCIA method {m} not found in available methods.")

        if edges_methods:
            available_methods = get_available_methods()
            for m in edges_methods:
                if m not in available_methods:
                    raise ValueError(
                        f"Edge LCIA method {m} not found in available `edges` methods."
                    )

        if methods and edges_methods:
            raise ValueError(
                "Please provide either `methods` or `edges_methods`, not both."
            )

        if methods is None and edges_methods is None:
            raise ValueError(
                "Please provide at least one of `methods` or `edges_methods`."
            )

        # if no methods are provided, use all those available

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
                methods=methods or [str(m) for m in edges_methods],
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
                        edges_methods,
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
        """Collapse small contributions and optionally interpolate across years.

        :param cutoff: Minimum contribution kept per activity category.
        :type cutoff: float
        :param interpolate: Interpolate missing years when ``True``.
        :type interpolate: bool
        :returns: ``None`` (mutates ``self.lca_results``).
        :rtype: None
        """
        self.lca_results = display_results(
            self.lca_results, cutoff=cutoff, interpolate=interpolate
        )

    def export_results(self, filename: str = None) -> str:
        """Export non-zero LCA results to a compressed parquet file.

        :param filename: Optional base filename (without extension).
        :type filename: str | None
        :returns: Path to the written ``.gzip`` parquet file.
        :rtype: str
        """
        return export_results_to_parquet(self.lca_results, filename)
