"""Matrix selection must ignore years embedded in ancestor directory names."""

import pytest

from pathways.lca import get_lca_matrices


@pytest.mark.parametrize("year", [2020, 2050])
def test_matrix_year_is_exact_directory(tmp_path, year):
    root = tmp_path / "sweet_sure-2050-switzerland" / "run-2020"
    files = []
    for matrix_year in [2020, 2050]:
        directory = root / "inventories" / "remind" / "SSP2-test" / str(matrix_year)
        directory.mkdir(parents=True)
        contents = {
            "A_matrix_index.csv": (
                "name;reference product;unit;location;index\n"
                f"plant {matrix_year};electricity;kilowatt hour;CH;0\n"
            ),
            "B_matrix_index.csv": (
                "name;compartment;subcompartment;unit;index\n"
                "Carbon dioxide, fossil;air;unspecified;kilogram;0\n"
            ),
            "A_matrix.csv": "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip\n0;0;1;0;0;0;0;0;0;0;0\n",
            "B_matrix.csv": "row;col;value;uncertainty type;loc;scale;shape;minimum;maximum;negative;flip\n0;0;2;0;0;0;0;0;0;0;0\n",
        }
        for name, content in contents.items():
            path = directory / name
            path.write_text(content)
            files.append(str(path))
    _, activity_index, biosphere_index, _, _ = get_lca_matrices(
        filepaths=files, model="remind", scenario="SSP2-test", year=year
    )
    assert list(activity_index) == [
        (f"plant {year}", "electricity", "kilowatt hour", "CH")
    ]
    assert list(biosphere_index) == [("Carbon dioxide, fossil", "air", "unspecified")]
