from unittest.mock import Mock

import pytest

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
