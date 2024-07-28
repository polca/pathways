from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from pathways.lca import load_matrix_and_index, read_indices_csv


def test_read_indices_csv_success():
    mock_csv_data = "activity;product;location;unit;1\nanother_activity;another_product;another_location;another_unit;2"
    expected_dict = {
        ("activity", "product", "location", "unit"): 1,
        ("another_activity", "another_product", "another_location", "another_unit"): 2,
    }
    with patch("builtins.open", mock_open(read_data=mock_csv_data)):
        result = read_indices_csv(Path("dummy_path.csv"))
        assert result == expected_dict


def test_load_matrix_and_index(tmp_path):
    mock_csv_data = (
        "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip"
        "\n1;0;3.5;3;4;5;6;7;8;0;0"
        "\n1;1;0.5;3;4;5;6;7;8;0;1"
    )
    expected_output = (
        np.array([3.5, 0.5]),
        np.array([(0, 1), (1, 1)], dtype=[("row", "i4"), ("col", "i4")]),
        np.array([False, True]),
        np.array(
            [(3, 4.0, 5.0, 6.0, 7.0, 8.0, False), (3, 4.0, 5.0, 6.0, 7.0, 8.0, False)],
            dtype=[
                ("uncertainty_type", "i4"),
                ("loc", "f4"),
                ("scale", "f4"),
                ("shape", "f4"),
                ("minimum", "f4"),
                ("maximum", "f4"),
                ("negative", "?"),
            ],
        ),
    )

    # Write mock CSV data to a temporary file
    temp_file = tmp_path / "temp.csv"
    temp_file.write_text(mock_csv_data)

    # Call the function with the path to the temporary file
    data_array, indices_array, flip_array, distributions_array = load_matrix_and_index(
        temp_file
    )

    # Check that the output matches the expected output
    # but they have different dtypes

    assert np.allclose(data_array, expected_output[0])
    assert np.array_equal(indices_array, expected_output[1])
    assert np.array_equal(flip_array, expected_output[2])
    assert np.array_equal(distributions_array, expected_output[3])
