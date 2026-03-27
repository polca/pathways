import json
from unittest.mock import Mock

import numpy as np
import pytest
import sparse as sp

from pathways.pathways import _fill_in_result_array
from pathways.utils import _get_mapping, _group_technosphere_indices


def test_group_technosphere_indices():
    indices = {("activity1", "location1"): 0, ("activity2", "location2"): 1}
    group_by = lambda x: x[1]  # Group by location
    group_values = ["location1", "location2"]
    expected = {"location1": [0], "location2": [1]}
    result = _group_technosphere_indices(indices, group_by, group_values)
    assert result == expected, "Grouping does not match expected output"


def test_get_mapping():
    mock_data = Mock()
    mock_data.get_resource.return_value.raw_read.return_value = """
    variable1:
      dataset: [details]
    """
    expected_mapping = {"variable1": {"dataset": ["details"]}}
    assert (
        _get_mapping(mock_data) == expected_mapping
    ), "Mapping does not match expected dictionary"


def test_fill_in_result_array_handles_single_mc_iteration_file(tmp_path, monkeypatch):
    iter_array = sp.COO.from_numpy(np.arange(24, dtype=float).reshape(2, 1, 3, 4))
    iter_path = tmp_path / "iter_results.npz"
    sp.save_npz(iter_path, iter_array)

    uncertainty_indices_path = tmp_path / "uncertainty_indices.npy"
    np.save(uncertainty_indices_path, np.empty((0,), dtype=int))

    uncertainty_values_path = tmp_path / "uncertainty_values.npy"
    np.save(uncertainty_values_path, np.empty((0, 1), dtype=float))

    technosphere_indices_path = tmp_path / "technosphere_indices.json"
    technosphere_indices_path.write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(
        "pathways.pathways.log_mc_parameters_to_excel",
        lambda **kwargs: None,
    )

    result = {
        "EUR": {
            "iterations_results": [str(iter_path)],
            "uncertainty_params": [str(uncertainty_indices_path)],
            "iterations_param_vals": [str(uncertainty_values_path)],
            "technosphere_indices": [str(technosphere_indices_path)],
        }
    }

    array = _fill_in_result_array(
        coords=("remind", "ssp2", 2005),
        result=result,
        use_distributions=100,
        shares=None,
        methods=["climate change"],
    )

    assert array.shape == (3, 2, 1, 4, 1, 3)
    assert np.array_equal(array[..., 0], array[..., 1])
    assert np.array_equal(array[..., 1], array[..., 2])
