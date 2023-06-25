"""
This module defines the class Pathways, which reads in a datapackage
that contains scenario data, mapping between scenario variables and
LCA datasets, and LCA matrices.
"""

import json
from .data_validation import validate_datapackage
from datapackage import DataPackage
import xarray as xr
import pandas as pd
import yaml

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
        filepath = self.data["mappings"]["path"]
        with open(filepath, "r") as f:
            return yaml.safe_load(f)

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
        scenarios = {}
        for scenario in self.data["scenarios"]:
            scenarios[scenario] = self.read_scenario_data(scenario)

        # Concatenate into xarray DataArray
        data = xr.concat(scenarios, dim="scenario")

        # Add metadata
        data.attrs["metadata"] = self.data.descriptor["metadata"]

        # Replace variable names with values found in self.mapping
        # under `pathways variable`
        for variable in data.coords["variable"].values:
            for item in self.mapping:
                if variable == item["scenario variable"]:
                    data.coords["variable"].loc[variable] = item["pathways variable"]

        return data