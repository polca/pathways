"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import json

import pandas as pd
import yaml
from datapackage import DataPackage
from pathlib import Path
from csv import reader
import csv
import numpy as np
from scipy import sparse
from .lcia import fill_characterization_factors_matrix, get_lcia_method_names
from .utils import load_classifications, load_units_conversion
import xarray as xr
from collections import defaultdict
from premise.geomap import Geomap
from multiprocessing import Pool, cpu_count
import sys

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


def _get_activity_indices(activities, A_index, geo, region):
    indices = []
    for activity in activities:
        idx = A_index.get((activity[0], activity[1], activity[2], region))
        if idx is not None:
            indices.append(int(idx))
        else:
            possible_locations = geo.iam_to_ecoinvent_location(activity[-1])

            for loc in possible_locations:
                idx = A_index.get((activity[0], activity[1], activity[2], loc))
                if idx is not None:
                    indices.append(int(idx))
                    break
            else:
                for default_loc in ["RoW", "GLO", "RER", "CH"]:
                    idx = A_index.get(
                        (activity[0], activity[1], activity[2], default_loc)
                    )
                    if idx is not None:
                        indices.append(int(idx))
                        break
    return indices


def process_region(data):
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

    #if region == "World":
    #    return None

    # Fetch the demand
    demand = scenarios.sel(
        region=region,
        model=model,
        pathway=scenario,
        year=year,
    )

    if demand.sum() == 0:
        return None

    activities = [
        (
            mapping[x]["dataset"]["name"],
            mapping[x]["dataset"]["reference product"],
            mapping[x]["dataset"]["unit"],
            region,
        )
        for x in demand.coords["variable"].values
    ]

    # Here, you would need to implement the _get_activity_indices method
    # as a standalone function, and adjust the inputs accordingly
    activities_idx = _get_activity_indices(activities, A_index, geo, region)

    assert len(activities_idx) == len(activities)

    units = [scenarios.attrs["units"][mapping[x]["scenario variable"]] for x in mapping]

    unit_conversion = np.ones(len(units))
    for i, unit in enumerate(units):
        unit_conversion[i] = units_map[unit][activities[i][2]]

    f = np.zeros(A.shape[0])
    f[activities_idx] = np.asarray(demand) * unit_conversion

    # remove contributions of activities_idx in other activities
    # Convert your csr_matrix to a lil_matrix
    A_lil = A.tolil()

    # Now, you can modify your lil_matrix without getting the warning
    A_lil[
        np.ix_(
            activities_idx,
            [x for x in range(A_lil.shape[1]) if x not in activities_idx],
        )
    ] = 0

    # Convert it back to csr_matrix if needed
    A = A_lil.tocsr()

    A_inv = spsolve(A, f)[:, np.newaxis]

    C = A_inv * B

    D = C[..., None] * lcia_matrix

    D = D.sum(axis=1)

    acts_idx = []
    for cat in lca_results_coords["act_category"].values:
        acts_idx.append(
            [int(A_index[a]) for a in reverse_classifications[cat] if a in A_index]
        )

    max_len = max([len(x) for x in acts_idx])
    acts_idx = np.array(
        [np.pad(x, (0, max_len - len(x)), constant_values=-1) for x in acts_idx]
    )
    acts_idx = np.swapaxes(acts_idx, 0, 1)

    D = D[acts_idx, ...].sum(axis=0)

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
        # self.lca_A, self.lca_B = self.get_lca_matrices()
        # self.lca_labels = self.get_lca_labels()

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

    def get_lca_matrices(self, model, scenario, year):

        dirpath = (
            Path(self.datapackage).parent / "inventories" / model / scenario / str(year)
        )

        # creates dict of activities <--> indices in A matrix
        A_inds = dict()
        with open(dirpath / "A_matrix_index.csv", "r") as read_obj:
            csv_reader = reader(read_obj, delimiter=";")
            for row in csv_reader:
                A_inds[(row[0], row[1], row[2], row[3])] = row[4]

        # creates dict of bio flow <--> indices in B matrix
        B_inds = dict()
        with open(dirpath / "B_matrix_index.csv", "r") as read_obj:
            csv_reader = reader(read_obj, delimiter=";")
            for row in csv_reader:
                B_inds[(row[0], row[1], row[2], row[3])] = row[4]

        # create a sparse A matrix
        A_coords = np.genfromtxt(dirpath / "A_matrix.csv", delimiter=";", skip_header=1)
        I = A_coords[:, 0].astype(int)
        J = A_coords[:, 1].astype(int)
        A = sparse.csr_matrix((A_coords[:, 2], (J, I)))

        # create a sparse B matrix
        B_coords = np.genfromtxt(dirpath / "B_matrix.csv", delimiter=";", skip_header=1)
        I = B_coords[:, 0].astype(int)
        J = B_coords[:, 1].astype(int)
        B = sparse.csr_matrix(
            (B_coords[:, 2] * -1, (I, J)), shape=(A.shape[0], len(B_inds))
        )

        return A, B, A_inds, B_inds

    def create_lca_results_array(self, methods, years, regions, models, scenarios):
        """
        Create an xarray where to store results.

        :return:
        """

        # the dimensions are `emissions`, `act_category`, `impact_category`, `year`
        # the coordinates are keys from B_inds, values from self.classifications,
        # outputs from get_lcia_methods(), and years from self.scenarios.coords["year"].values

        # create the coordinates

        coords = {
            # "emissions": list_emissions,
            "act_category": list(set(self.classifications.values())),
            "impact_category": methods,
            "year": years,
            "region": regions,
            "model": models,
            "scenario": scenarios,
        }

        # create the DataArray
        return xr.DataArray(
            np.zeros(
                (
                    # len(coords["emissions"]),
                    len(coords["act_category"]),
                    len(methods),
                    len(years),
                    len(regions),
                    len(models),
                    len(scenarios),
                )
            ),
            coords=coords,
            dims=[
                # "emissions",
                "act_category",
                "impact_category",
                "year",
                "region",
                "model",
                "scenario",
            ],
        )

    def calculate(
        self, methods=None, models=None, scenarios=None, regions=None, years=None
    ):
        missing_class = []

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

        if self.lca_results is None:
            self.lca_results = self.create_lca_results_array(
                methods, years, regions, models, scenarios
            )

        for model in models:
            geo = Geomap(model=model)
            for scenario in scenarios:
                for year in years:

                    try:
                        A, B, A_index, B_index = self.get_lca_matrices(
                            model, scenario, year
                        )
                        B = np.asarray(B.todense())
                    except FileNotFoundError:
                        continue



                    self.lcia_matrix = fill_characterization_factors_matrix(
                        list(B_index.keys()), methods
                    )

                    for act in A_index:
                        if act not in self.classifications:
                            missing_class.append(list(act))

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

                    with Pool(cpu_count()) as p:
                        results = p.map(process_region, data_for_regions)

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

    def display_results(self, cutoff=0.01):
        """
        Display results in a dataframe
        But aggregate activity_categories if they represent less than
        `cutoff` percent of the total impact, and call that "other".
        """

        if self.lca_results is None:
            raise ValueError("No results to display")

        # aggregate activity categories
        df = (
            self.lca_results.to_dataframe("value")
            .reset_index()
            .groupby(["model", "scenario", "act_category", "impact_category", "year", "region"])
            .sum()
            .reset_index()
        )

        # get total impact per year
        df = df.merge(
            df.groupby(["model", "scenario", "impact_category", "year", "region"])["value"]
            .sum()
            .reset_index(),
            on=["impact_category", "year", "region"],
            suffixes=["", "_total"],
        )

        # get percentage of total
        df["percentage"] = df["value"] / df["value_total"]

        # aggregate activity categories
        df.loc[df["percentage"] < cutoff, "act_category"] = "other"

        # drop total columns
        df = df.drop(columns=["value_total", "percentage"])

        arr = (
            df.groupby(["model", "scenario", "act_category", "impact_category", "year", "region"])["value"]
            .sum()
            .to_xarray()
        )

        # interpolate years to have a continuous time series
        if len(arr.year) > 1:
            arr = arr.interp(
                year=np.arange(arr.year.min(), arr.year.max() + 1),
                kwargs={"fill_value": "extrapolate"},
                method="linear",
            )

        # convert to xarray
        return arr
