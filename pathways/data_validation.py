"""
This module contains functions that validate the data contained
in the datapackage.json file.
"""

import json
import datapackage
from pathlib import Path
import pandas as pd
import yaml


def validate_datapackage(datapackage: datapackage.DataPackage):
    """
    Validate the datapackage.json file.
    The datapackage must be valid according to the Frictionless Data.
    It must also contain the following resources:
        - scenarios
        - matrices
        - mappings
    And must also contain some metadata, such as:
        - authors
        - description

    """

    # Validate datapackage according to the Frictionless Data specifications
    try:
        datapackage.validate()
    except datapackage.errors.ValidationError as e:
        raise ValueError(f"Invalid datapackage: {e}")

    # Check that the datapackage contains the required resources
    required_resources = ["scenarios", "matrices", "mappings"]
    for resource in required_resources:
        if resource not in datapackage.resources:
            raise ValueError(f"Missing resource: {resource}")

    # Check that the datapackage contains the required metadata
    required_metadata = ["authors", "description"]
    for metadata in required_metadata:
        if metadata not in datapackage.descriptor:
            raise ValueError(f"Missing metadata: {metadata}")

    # Check that the scenario data is valid
    for scenario in datapackage.resources["scenarios"].read():
        validate_scenario_data(scenario["path"])

    # Check that the mapping is valid
    validate_mapping(datapackage.resources["mappings"].path)

    return datapackage


def read_scenario_data(filepath: str) -> pd.DataFrame:
    """
    Read the scenario data.
    :param filepath:
    :return: pd.DataFrame
    """

    # Read the data
    if filepath.endswith(".csv"):
        data = pd.read_csv(filepath)
    else:
        data = pd.read_excel(filepath)

    return data


def validate_scenario_data(filepath: str) -> bool:
    """
    This function validates the scenario data.
    `filepath` is a relative path within the datapackage.
    The filepath must be either a CSV or Excel file.
    The file must contain the following columns:
        - scenario: string
        - variable: string
        - region: string
        - year: integer
        - value: float

    :param filepath: Relative path to the scenario data file.
    :return: True if the data is valid, False otherwise.
    """

    # Ensure that filepath is a string
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
    # Ensures that filepath is either a CSV or Excel file
    if not filepath.endswith(".csv") and not filepath.endswith(".xlsx"):
        raise ValueError("filepath must be a CSV or Excel file")
    # Ensure that the file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = read_scenario_data(filepath)

    # Check that the file contains the required columns
    required_columns = ["scenario", "variable", "region", "year", "value"]
    required_data_types = {
        "scenario": str,
        "variable": str,
        "region": str,
        "year": int,
        "value": float
    }

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing mandatory column: {column}")

        if not data[column].dtype == required_data_types[column]:
            raise TypeError(f"Invalid data type for column {column}. "
                            f"Must be {required_data_types[column]}")

    return True


def validate_mapping(filepath: str):
    """
    Validates the mapping between scenario variables and LCA datasets.
    The mapping must be a YAML file.
    Its structure should be like this:

    - pathways variable: string
      dataset: string
      scenario variable: string

    :param filepath: relative path to the mapping file.
    :return: boolean
    """

    # Ensure that filepath is a string
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")
    # Ensures that filepath is a YAML file
    if not filepath.endswith(".yaml"):
        raise ValueError("filepath must be a YAML file")
    # Ensure that the file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read the data
    with open(filepath, "r") as f:
        mapping = yaml.safe_load(f)

    # Check that the data has the required structure
    required_keys = ["pathways variable", "dataset", "scenario variable"]
    for item in mapping:
        for key in required_keys:
            if key not in item:
                raise ValueError(f"Missing key: {key}")

    # Check that all values for `scenario variable` are unique
    scenario_variables = [item["scenario variable"] for item in mapping]
    if len(scenario_variables) != len(set(scenario_variables)):
        raise ValueError("All values for `scenario variable` must be unique")

    # Check that all values for `pathways variable` are unique
    pathways_variables = [item["pathways variable"] for item in mapping]
    if len(pathways_variables) != len(set(pathways_variables)):
        raise ValueError("All values for `pathways variable` must be unique")

    # Check that all values for `dataset` are unique
    datasets = [item["dataset"] for item in mapping]
    if len(datasets) != len(set(datasets)):
        raise ValueError("All values for `dataset` must be unique")

    # Check that all values for `scenario variable` are present in the scenario data
    scenario_data = read_scenario_data("data/scenarios.csv")
    scenario_variables = set(scenario_variables)
    if not scenario_variables.issubset(set(scenario_data["variable"].unique())):
        raise ValueError("All values for `scenario variable` must be present in the scenario data")
