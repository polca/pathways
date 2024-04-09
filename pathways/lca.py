import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bw_processing as bwp
import numpy as np
from bw_processing import Datapackage
from scipy import sparse
from scipy.sparse import csr_matrix

from .lcia import get_lcia_methods

# Attempt to import pypardiso's spsolve function.
# If it isn't available, we fall back on scipy's spsolve.
try:
    from pypardiso import spsolve

    print("Solver: pypardiso")
except ImportError:
    from scikits.umfpack import spsolve

    print("Solver: scikits.umfpack")

logging.basicConfig(
    level=logging.DEBUG,
    filename="pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_indices_csv(file_path: Path) -> Dict[Tuple[str, str, str, str], str]:
    """
    Reads a CSV file and returns its contents as a dictionary.

    Each row of the CSV file is expected to contain four string values followed by an index.
    These are stored in the dictionary as a tuple of the four strings mapped to the index.

    :param file_path: The path to the CSV file.
    :type file_path: Path

    :return: A dictionary mapping tuples of four strings to indices.
    :rtype: Dict[Tuple[str, str, str, str], str]
    """
    indices = dict()
    with open(file_path) as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=";")
        for row in csv_reader:
            indices[(row[0], row[1], row[2], row[3])] = int(row[4])
    return indices


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a CSV file and returns its contents as a CSR sparse matrix.

    :param file_path: The path to the CSV file.
    :type file_path: Path
    :return: A tuple containing the data, indices, and sign of the data.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Load the data from the CSV file
    array = np.genfromtxt(file_path, delimiter=";", skip_header=1)

    # give `indices_array` a list of tuples of indices
    indices_array = np.array(
        list(zip(array[:, 1].astype(int), array[:, 0].astype(int))),
        dtype=bwp.INDICES_DTYPE,
    )

    data_array = array[:, 2]

    # make a boolean scalar array to store the sign of the data
    flip_array = array[:, -1].astype(bool)

    distributions_array = np.array(
        list(
            zip(
                array[:, 3].astype(int),  # uncertainty type
                array[:, 4].astype(float),  # loc
                array[:, 5].astype(float),  # scale
                array[:, 6].astype(float),  # shape
                array[:, 7].astype(float),  # minimum
                array[:, 8].astype(float),  # maximum
                array[:, 9].astype(bool),  # negative
            )
        ),
        dtype=bwp.UNCERTAINTY_DTYPE,
    )

    return data_array, indices_array, flip_array, distributions_array


def get_lca_matrices(
    datapackage: str,
    model: str,
    scenario: str,
    year: int,
) -> Tuple[Datapackage, Dict, Dict]:
    """
    Retrieve Life Cycle Assessment (LCA) matrices from disk.

    :param datapackage: The path to the datapackage.
    :type datapackage: str
    :param model: The name of the model.
    :type model: str
    :param scenario: The name of the scenario.
    :type scenario: str
    :param year: The year of the scenario.
    :type year: int
    :rtype: Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict, Dict]
    """

    # find the correct filepaths in filepaths
    # the correct filepath are the strings that contains
    # the model, scenario and year
    def filter_filepaths(suffix: str, contains: List[str]):
        return [
            Path(fp)
            for fp in filepaths
            if all(kw in fp for kw in contains)
            and Path(fp).suffix == suffix
            and Path(fp).exists()
        ]

    def select_filepath(keyword: str, fps):
        matches = [fp for fp in fps if keyword in fp.name]
        if not matches:
            raise FileNotFoundError(f"Expected file containing '{keyword}' not found.")
        return matches[0]

    fps = filter_filepaths(".csv", [model, scenario, str(year)])
    if len(fps) != 4:
        raise ValueError(f"Expected 4 filepaths, got {len(fps)}")

    fp_A_inds = select_filepath("A_matrix_index", fps)
    fp_B_inds = select_filepath("B_matrix_index", fps)
    technosphere_inds = read_indices_csv(fp_A_inds)
    biosphere_inds = read_indices_csv(fp_B_inds)

    dp = bwp.create_datapackage()

    fp_A = select_filepath("A_matrix", [fp for fp in fps if "index" not in fp.name])
    fp_B = select_filepath("B_matrix", [fp for fp in fps if "index" not in fp.name])

    # Load matrices and add them to the datapackage
    for matrix_name, fp in [("technosphere_matrix", fp_A), ("biosphere_matrix", fp_B)]:
        data, indices, sign, distributions = load_matrix_and_index(fp)
        dp.add_persistent_vector(
            matrix=matrix_name,
            indices_array=indices,
            data_array=data,
            flip_array=sign if matrix_name == "technosphere_matrix" else None,
            distributions_array=distributions,
        )

    return dp, technosphere_inds, biosphere_inds


def fill_characterization_factors_matrices(
    methods: list, biosphere_matrix_dict: dict, biosphere_dict: dict, debug=False
) -> csr_matrix:
    """
    Create one CSR matrix for all LCIA method, with the last dimension being the index of the method
    :param methods: contains names of the methods to use.
    :param biosphere_matrix_dict: dictionary with biosphere flows and their indices
    :param debug: if True, log debug information
    :return: a sparse matrix with the characterization factors
    """

    lcia_data = get_lcia_methods(methods=methods)

    # Prepare data for efficient creation of the sparse matrix
    data = []
    rows = []
    cols = []
    cfs = []

    for m, method in enumerate(methods):
        method_data = lcia_data[method]
        for flow_name in method_data:
            if flow_name in biosphere_dict:
                idx = biosphere_dict[flow_name]
                if idx in biosphere_matrix_dict:
                    data.append(method_data[flow_name])
                    rows.append(biosphere_matrix_dict[idx])
                    cols.append(m)
                    cfs.append((method, flow_name, idx, method_data[flow_name]))

    # Efficiently create the sparse matrix
    matrix = sparse.csr_matrix(
        (data, (cols, rows)),
        shape=(len(methods), len(biosphere_matrix_dict)),
        dtype=np.float64,
    )

    if debug:
        # sort l by method and flow
        cfs = sorted(cfs, key=lambda x: (x[0], x[1]))
        for x in cfs:
            method, flow, f, value = x
            logging.info(
                f"LCIA method: {method}, Flow: {flow}, Index: {f}, Value: {value}"
            )

    return matrix


def remove_double_counting(
    characterized_inventory: csr_matrix, vars_info: dict, activity_idx: int
) -> csr_matrix:
    """
    Remove double counting from a characterized inventory matrix for all activities except
    the activity being evaluated, across all methods.

    :param characterized_inventory: Characterized inventory matrix with rows for different methods and columns for different activities.
    :param vars_info: Dictionary with information about which indices to zero out.
    :param activity_idx: Index of the activity being evaluated, which should not be zeroed out.
    :return: Characterized inventory with double counting removed for all but the evaluated activity.
    """

    print("Removing double counting")
    if isinstance(characterized_inventory, np.ndarray):
        characterized_inventory = csr_matrix(characterized_inventory)
    elif not isinstance(characterized_inventory, csr_matrix):
        raise TypeError(
            "characterized_inventory must be a csr_matrix or a numpy array."
        )

    # Gather all indices for which we want to avoid double counting, except the evaluated activity
    list_of_idx_to_remove = []
    for region in vars_info:
        for variable in vars_info[region]:
            idx = vars_info[region][variable]
            if idx != activity_idx:
                list_of_idx_to_remove.append(idx)

    # Convert to lil_matrix for more efficient element-wise operations - CHECK IF THIS IS ACTUALLY FASTER
    characterized_inventory = characterized_inventory.tolil()

    # Zero out the specified indices for all methods, except for the activity being evaluated
    for idx in list_of_idx_to_remove:
        characterized_inventory[:, idx] = 0

    return characterized_inventory.tocsr()
