"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import json

import pandas as pd
import xarray as xr
import yaml
from datapackage import DataPackage
from pathlib import Path
from csv import reader
import csv
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve

from .data_validation import validate_datapackage


def csv_to_dict(filename):
    output_dict = {}

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            # Making sure there are at least 5 items in the row
            if len(row) >= 5:
                # The first four items are the key, the fifth item is the value
                key = tuple(row[:4])
                value = row[4]
                output_dict[int(value)] = key

    return output_dict


def get_visible_files(path):
    return [
        file for file in Path(path).iterdir() if not file.name.startswith(".")
    ]


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
        scenario_data = pd.DataFrame(scenario_data, columns=self.data.get_resource("scenarios").headers)

        # remove rows which do not have a value under the `variable`
        # column that correpsonds to any value in self.mapping for `scenario variable`
        scenario_data = scenario_data[
            scenario_data["variable"].isin([item["scenario variable"] for item in self.mapping.values()])]

        # Convert to xarray DataArray
        data = (
            scenario_data
            .groupby(["model", "pathway", "variable", "region", "year"])["value"]
            .mean()
            .to_xarray()
        )

        # Add metadata
        data.attrs["contributors"] = self.data.descriptor["contributors"]
        data.attrs["description"] = self.data.descriptor["description"]

        # Replace variable names with values found in self.mapping
        new_names = []
        for variable in data.coords["variable"].values:
            for p_var, val in self.mapping.items():
                if val["scenario variable"] == variable:
                    new_names.append(p_var)
        data.coords["variable"] = new_names

        return data

    def get_lca_matrices(self, model, scenario, year):

        dirpath = Path(self.datapackage).parent / "inventories" / model / scenario / str(year)

        # creates dict of activities <--> indices in A matrix
        A_inds = dict()
        with open(dirpath / "A_matrix_index.csv", 'r') as read_obj:
            csv_reader = reader(read_obj, delimiter=";")
            for row in csv_reader:
                A_inds[(row[0], row[1], row[2], row[3])] = row[4]

        A_inds_rev = {int(v): k for k, v in A_inds.items()}

        # creates dict of bio flow <--> indices in B matrix
        B_inds = dict()
        with open(dirpath / "B_matrix_index.csv", 'r') as read_obj:
            csv_reader = reader(read_obj, delimiter=";")
            for row in csv_reader:
                B_inds[(row[0], row[1], row[2], row[3])] = row[4]

        B_inds_rev = {int(v): k for k, v in B_inds.items()}

        # create a sparse A matrix
        A_coords = np.genfromtxt(dirpath / "A_matrix.csv", delimiter=";", skip_header=1)
        I = A_coords[:, 0].astype(int)
        J = A_coords[:, 1].astype(int)
        A = sparse.csr_matrix((A_coords[:, 2], (J, I)))

        # create a sparse B matrix
        B_coords = np.genfromtxt(dirpath / "B_matrix.csv", delimiter=";", skip_header=1)
        I = B_coords[:, 0].astype(int)
        J = B_coords[:, 1].astype(int)
        B = sparse.csr_matrix((B_coords[:, 2] * -1, (I, J)), shape=(A.shape[0], len(B_inds)))

        return A, B, A_inds, B_inds

    def calculate(self):
        """Calculate the LCA results for the scenarios."""
        # iterate through the scenarios
        for model in self.scenarios.coords["model"].values:
            for scenario in self.scenarios.coords["pathway"].values:
                for year in self.scenarios.coords["year"].values:

                    print(model, scenario, year)

                    A, B, A_index, B_index = self.get_lca_matrices(model, scenario, year)

                    print(A.shape, B.shape)

                    for var in self.scenarios.coords["variable"].values:
                        for region in self.scenarios.coords["region"].values:
                            # get the value for the current scenario
                            value = self.scenarios.sel(
                                model=model,
                                pathway=scenario,
                                variable=var,
                                region=region,
                                year=year
                            ).values

                            # get the corresponding LCA activity
                            # by looking into `self.mapping`

                            # get the activity details
                            name = self.mapping[var]["dataset"]["name"]
                            reference_product = self.mapping[var]["dataset"]["reference product"]
                            unit = self.mapping[var]["dataset"]["unit"]

                            if (name, reference_product, unit, region) in A_index:

                                key = (name, reference_product, unit, region)
                                idx = A_index[key]
                                print(key, idx)
                                f = np.zeros(A.shape[0])
                                f[int(idx)] = 1
                                A_inv = spsolve(np.nan_to_num(A), f)
                                print(A_inv.shape)
                                # print()
                                # C = A_inv * B
                                # print(C.sum())
