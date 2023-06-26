"""
This module contains functions that validate the data contained
in the datapackage.json file.
"""

import json
from pathlib import Path

import datapackage
from datapackage import DataPackageException, validate
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

    # Validate datapackage according
    # to the Frictionless Data specifications
    try:
        validate(datapackage.descriptor)
    except DataPackageException as e:
        if e.multiple:
            for error in e.errors:
                print(f"Invalid datapackage: {error}")
        else:
            raise ValueError(f"Invalid datapackage: {e}")

    # Check that the datapackage contains the required resources
    required_resources = ["scenarios", "exchanges", "labels", "mapping"]
    for resource in required_resources:
        try:
            datapackage.get_resource(resource)
        except DataPackageException:
            raise ValueError(f"Missing resource: {resource}")

    # Check that the datapackage contains the required metadata
    required_metadata = ["contributors", "description"]
    for metadata in required_metadata:
        if metadata not in datapackage.descriptor:
            raise ValueError(f"Missing metadata: {metadata}")

    # Check that the scenario data is valid
    validate_scenario_data(datapackage.get_resource("scenarios"))

    # Check that the mapping is valid
    validate_mapping(datapackage.get_resource("mapping"), datapackage)

    # Check that the LCA data is valid
    #validate_lca_data(datapackage)

    return datapackage


def validate_scenario_data(resource: datapackage.Resource) -> bool:
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

    :param resource: Datapackage resource.
    :return: True if the data is valid, False otherwise.
    """

    # Check that the file contains the required columns
    required_columns = ["model", "pathway", "variable", "region", "year", "value"]

    data = resource.read()
    headers = resource.headers
    df = pd.DataFrame(data, columns=headers)

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing mandatory column: {column}")

    return True

def validate_mapping(resource: datapackage.Resource, datapackage: datapackage.DataPackage):
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

    mapping = yaml.safe_load(resource.raw_read())

    # Check that the data has the required structure
    required_keys = ["dataset", "scenario variable"]
    for k, v in mapping.items():
        if not set(required_keys).issubset(set(v.keys())):
            raise ValueError(f"Invalid mapping: missing keys for {k}")

    # Check that all values for `scenario variable` are unique
    scenario_variables = [item["scenario variable"] for item in mapping.values()]
    if len(scenario_variables) != len(set(scenario_variables)):
        raise ValueError("All values for `scenario variable` must be unique")

    # Check that all values for `scenario variable` are present in the scenario data
    scenario_data = datapackage.get_resource("scenarios").read()
    scenario_data = pd.DataFrame(scenario_data, columns=datapackage.get_resource("scenarios").headers)
    scenario_variables = set(scenario_variables)
    if not scenario_variables.issubset(set(scenario_data["variable"].unique())):
        raise ValueError(
            "All values for `scenario variable` must be present in the scenario data"
            f"Missing variables: {scenario_variables - set(scenario_data['variable'].unique())}"
        )


def validate_lca_data(datapackage):
    """
    Ensure that the required LCA data is present and valid.
    :param datapackage:
    :return:
    """

    # Check that resources `exchanges` and `labels` are present
    required_resources = ["exchanges", "labels"]
    for resource in required_resources:
        try:
            datapackage.get_resource(resource).read()
        except DataPackageException:
            raise ValueError(f"Missing resource: {resource}")
