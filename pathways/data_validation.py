"""
This module contains functions that validate the data contained
in the datapackage.json file.
"""

import logging

import datapackage
import pandas as pd
import yaml
from datapackage import DataPackageException, validate

from .filesystem_constants import USER_LOGS_DIR

logging.basicConfig(
    level=logging.DEBUG,
    filename=USER_LOGS_DIR / "pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def validate_datapackage(
    data_package: datapackage.DataPackage,
) -> (datapackage.DataPackage, pd.DataFrame, list):
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
        validate(data_package.descriptor)
    except DataPackageException as e:
        # we want to remove errors relating to the YAMl file
        for error in e.errors:
            if ".yaml" in str(error):
                e.errors.remove(error)

        if e.multiple:
            for error in e.errors:
                if "not one of" in str(error):
                    e.errors.remove(error)
                else:
                    print(f"Invalid datapackage: {error}")
        else:
            raise ValueError(f"Invalid datapackage: {e}")

    # Check that the datapackage contains the required resources
    required_resources = ["scenario_data", "exchanges", "labels", "mapping"]
    for resource in required_resources:
        try:
            data_package.get_resource(resource)
        except DataPackageException:
            raise ValueError(f"Missing resource: {resource}")

    # Check that the datapackage contains the required metadata
    required_metadata = ["contributors", "description"]
    for metadata in required_metadata:
        if metadata not in data_package.descriptor:
            raise ValueError(f"Missing metadata: {metadata}")

    # extract the scenario data
    data = data_package.get_resource("scenario_data").read()
    headers = data_package.get_resource("scenario_data").headers
    dataframe = pd.DataFrame(data, columns=headers)

    # Check that the scenario data is valid
    validate_scenario_data(dataframe)

    # Check that the mapping is valid
    validate_mapping(data_package.get_resource("mapping"))

    # fetch filepaths to resources
    filepaths = []
    for resource in data_package.resources:
        if "matrix" in resource.descriptor["name"]:
            filepaths.append(resource.source)

    return data_package, dataframe, filepaths


def validate_scenario_data(dataframe: pd.DataFrame) -> bool:
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

    :param dataframe: pandas DataFrame containing the scenario data.
    :return: True if the data is valid, False otherwise.
    """

    # Check that the file contains the required columns
    required_columns = ["model", "pathway", "variables", "region", "year", "value"]

    for column in required_columns:
        if column not in dataframe.columns:
            raise ValueError(f"Missing mandatory column: {column}")

    return True


def validate_mapping(resource: datapackage.Resource):
    """
    Validates the mapping between scenario variables and LCA datasets.
    The mapping must be a YAML file.
    Its structure should be like this:

    - pathways variable: string
      dataset: string
      scenario variable: string

    :param resource: datapackage.Resource
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
        print(
            "All values for `scenario variable` must be unique. "
            f"Duplicate values: {set([x for x in scenario_variables if scenario_variables.count(x) > 1])}"
        )
