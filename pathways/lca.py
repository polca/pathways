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
# If it isn't available, fall back on scipy's spsolve.
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
            indices[(row[0], row[1], row[2], row[3])] = row[4]
    return indices


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a CSV file and returns its contents as a CSR sparse matrix.

    :param file_path: The path to the CSV file.
    :type file_path: Path
    :return: A tuple containing the data, indices, and sign of the data.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    # Load the data from the CSV file
    coords = np.genfromtxt(file_path, delimiter=";", skip_header=1)

    # give `indices_array` a list of tuples of indices
    indices_array = np.array(
        list(zip(coords[:, 1].astype(int), coords[:, 0].astype(int))),
        dtype=bwp.INDICES_DTYPE,
    )

    data_array = coords[:, 2]
    # make a boolean scalar array to store the sign of the data
    sign_array = np.where(data_array < 0, True, False)
    # make values of data_array all positive
    data_array = np.abs(data_array)

    return data_array, indices_array, sign_array


def get_lca_matrices(
    datapackage: str,
    model: str,
    scenario: str,
    year: int,
) -> Tuple[Datapackage, Dict, Dict]:
    """
    Retrieve Life Cycle Assessment (LCA) matrices from disk.

    ...

    :rtype: Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict, Dict]
    """
    dirpath = (
        Path(datapackage).parent / "inventories" / model.lower() / scenario / str(year)
    )

    # check that files exist
    if not dirpath.exists():
        raise FileNotFoundError(f"Directory {dirpath} does not exist.")

    A_inds = read_indices_csv(dirpath / "A_matrix_index.csv")
    B_inds = read_indices_csv(dirpath / "B_matrix_index.csv")

    # create brightway datapackage
    dp = bwp.create_datapackage()

    a_data, a_indices, a_sign = load_matrix_and_index(
        dirpath / "A_matrix.csv",
    )

    b_data, b_indices, b_sign = load_matrix_and_index(
        dirpath / "B_matrix.csv",
    )

    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=a_indices,
        data_array=a_data,
        flip_array=a_sign,
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        indices_array=b_indices,
        data_array=b_data * -1,
        flip_array=b_sign,
    )

    return dp, A_inds, B_inds


def fill_characterization_factors_matrices(
    biosphere_flows: dict, methods, biosphere_dict, debug=False
) -> csr_matrix:
    """
    Create one CSR matrix for all LCIA method, with the last dimension being the index of the method
    :param biosphere_flows:
    :param methods: contains names of the methods to use.
    :return:
    """

    lcia_data = get_lcia_methods(methods=methods)
    biosphere_flows = {k[:3]: v for k, v in biosphere_flows.items()}
    reversed_biosphere_flows = {int(v): k for k, v in biosphere_flows.items()}

    matrix = sparse.csr_matrix(
        (len(methods), len(biosphere_dict)),
        dtype=np.float64,
    )

    if debug:
        logging.info(f"LCIA matrix shape: {matrix.shape}")

    l = []

    for m, method in enumerate(methods):
        method_data = lcia_data[method]
        for flow_idx, f in biosphere_dict.items():
            if flow_idx in reversed_biosphere_flows:
                flow = reversed_biosphere_flows[flow_idx]
                if flow in method_data:
                    matrix[m, f] = method_data[flow]
                    l.append((method, flow, f, method_data[flow]))
    if debug:
        # sort l by method and flow
        l = sorted(l, key=lambda x: (x[0], x[1]))
        for x in l:
            method, flow, f, value = x
            logging.info(
                f"LCIA method: {method}, Flow: {flow}, Index: {f}, Value: {value}"
            )

    return matrix


def remove_double_counting(A: csr_matrix, vars_info: dict) -> csr_matrix:
    """
    Remove double counting from a technosphere matrix.
    :param A: Technosphere matrix
    :param vars_info: Dictionary with information about variables
    :return: Technosphere matrix with double counting removed
    """

    # Modify A in COO format for efficiency
    # To avoid double-counting, set all entries in A corresponding
    # to activities not in activities_idx to zero

    A_coo = A.tocoo()

    list_of_idx = []

    for region in vars_info:
        for variable in vars_info[region]:
            idx = vars_info[region][variable]["idx"]
            if idx not in list_of_idx:
                list_of_idx.append(idx)
                row_mask = np.isin(A_coo.row, idx)
                col_mask = np.isin(A_coo.col, idx)
                A_coo.data[row_mask & ~col_mask] = 0  # zero out rows

    A_coo.eliminate_zeros()
    return A_coo.tocsr()
