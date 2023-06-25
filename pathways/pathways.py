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

from .data_validation import validate_datapackage


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
        scenario_data = scenario_data[scenario_data["variable"].isin([item["scenario variable"] for item in self.mapping])]

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
        data.coords["variable"] = [item["variable"] for item in self.mapping]

        return data
