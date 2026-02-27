from unittest.mock import Mock, patch

import pytest
from datapackage import DataPackageException

from pathways.data_validation import validate_datapackage


def test_validate_datapackage_raises_on_non_ignored_errors():
    package = Mock()
    package.descriptor = {}

    err = DataPackageException(
        "invalid",
        errors=["resource foo is invalid", "another serious error"],
    )

    with patch("pathways.data_validation.validate", side_effect=err):
        with pytest.raises(ValueError, match="Invalid datapackage"):
            validate_datapackage(package)


def test_validate_datapackage_ignores_expected_errors_and_continues():
    package = Mock()
    package.descriptor = {"contributors": [], "description": "x"}
    package.resources = []
    package.get_resource.side_effect = DataPackageException("missing")

    err = DataPackageException(
        "invalid",
        errors=["foo.yaml issue", "value not one of expected"],
    )

    with patch("pathways.data_validation.validate", side_effect=err):
        with pytest.raises(ValueError, match="Missing resource"):
            validate_datapackage(package)
