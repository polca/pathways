import json
from unittest.mock import mock_open, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pathways.lcia import (
    fill_characterization_factors_matrices,
    format_lcia_method_exchanges,
    get_lcia_method_names,
)


def test_get_lcia_method_names_success():
    mock_data = '[{"name": ["IPCC", "2021", "Global Warming Potential"]}, {"name": ["ReCiPe", "2016", "Midpoint"]} ]'
    expected_result = [
        "IPCC - 2021 - Global Warming Potential",
        "ReCiPe - 2016 - Midpoint",
    ]
    with patch("builtins.open", mock_open(read_data=mock_data)):
        with patch("json.load", return_value=json.loads(mock_data)):
            method_names = get_lcia_method_names()
            assert (
                method_names == expected_result
            ), "Method names not correctly formatted"


def test_format_lcia_method_exchanges():
    method_input = {
        "exchanges": [
            {"name": "CO2", "categories": ["air"], "amount": 1},
            {
                "name": "CH4",
                "categories": ["air", "low population density, long-term"],
                "amount": 25,
            },
        ]
    }
    expected_output = {
        ("CO2", "air", "unspecified"): 1,
        ("CH4", "air", "low population density, long-term"): 25,
    }
    assert (
        format_lcia_method_exchanges(method_input) == expected_output
    ), "Exchange formatting incorrect"


@pytest.fixture
def mock_lcia_methods_data():
    """Returns mock LCIA methods similar to what get_lcia_methods would return."""
    return {
        "IPCC 2021 - Global Warming Potential": {
            ("CO2", "air", "unspecified"): 1,
            ("CH4", "air", "low population density, long-term"): 25,
        }
    }


@pytest.fixture
def mock_biosphere_data():
    """Returns mock biosphere dictionary and matrix dict for testing."""
    biosphere_dict = {
        ("CO2", "air", "unspecified"): 0,
        ("CH4", "air", "low population density, long-term"): 1,
    }
    biosphere_matrix_dict = {
        0: 0,
        1: 1,
    }  # Mapping of biosphere_dict indices to matrix indices
    return biosphere_matrix_dict, biosphere_dict


def test_fill_characterization_factors_matrices(
    mock_lcia_methods_data, mock_biosphere_data
):
    methods = ["IPCC 2021 - Global Warming Potential"]
    biosphere_matrix_dict, biosphere_dict = mock_biosphere_data

    with patch("pathways.lcia.get_lcia_methods", return_value=mock_lcia_methods_data):
        matrix = fill_characterization_factors_matrices(
            methods, biosphere_matrix_dict, biosphere_dict, debug=False
        )

    assert isinstance(matrix, csr_matrix), "Output is not a CSR matrix"
    assert matrix.shape == (
        len(methods),
        len(biosphere_matrix_dict),
    ), "Matrix shape is incorrect"

    # Verifying content of the matrix
    expected_data = np.array([1, 25])
    np.testing.assert_array_equal(
        matrix.data, expected_data, "Matrix data does not match expected values"
    )
    np.testing.assert_array_equal(
        matrix.indices, np.array([0, 1]), "Matrix indices do not match expected values"
    )
    np.testing.assert_array_equal(
        matrix.indptr, np.array([0, 2]), "Matrix indices does not match expected values"
    )
