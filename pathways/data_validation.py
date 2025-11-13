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

logger = logging.getLogger(__name__)


def validate_datapackage(
    data_package: datapackage.DataPackage,
) -> (datapackage.DataPackage, pd.DataFrame, list):
    """Validate a Frictionless datapackage and extract its scenario content.

    :param data_package: Loaded datapackage descriptor to validate.
    :type data_package: datapackage.DataPackage
    :raises ValueError: If Frictionless validation fails, mandatory resources are
        missing, or required metadata is absent.
    :returns: The validated datapackage, the scenario data table, and CSV matrix
        file paths.
    :rtype: tuple[datapackage.DataPackage, pandas.DataFrame, list[str]]
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
    """Ensure the scenario sheet contains the mandatory columns.

    :param dataframe: Scenario observations extracted from the datapackage.
    :type dataframe: pandas.DataFrame
    :raises ValueError: If a required column is missing.
    :returns: ``True`` when all required fields are present.
    :rtype: bool
    """

    # Check that the file contains the required columns
    required_columns = ["model", "pathway", "variables", "region", "year", "value"]

    for column in required_columns:
        if column not in dataframe.columns:
            raise ValueError(f"Missing mandatory column: {column}")

    return True


def validate_mapping(resource: datapackage.Resource):
    """Validate the YAML mapping between scenario variables and LCA datasets.

    :param resource: Mapping resource as exposed by the datapackage.
    :type resource: datapackage.Resource
    :raises ValueError: If mandatory mapping keys are missing.
    :returns: ``None`` (the function raises when validation fails).
    :rtype: None
    """

    mapping = yaml.safe_load(resource.raw_read())

    # Check that the data has the required structure
    required_keys = [
        "dataset",
    ]
    for k, v in mapping.items():
        if not set(required_keys).issubset(set(v.keys())):
            raise ValueError(f"Invalid mapping: missing keys for {k}")
