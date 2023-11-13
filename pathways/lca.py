import csv
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy
import scipy.sparse
import xarray as xr
from scipy import sparse
from scipy.sparse import csr_matrix

# Attempt to import pypardiso's spsolve function.
# If it isn't available, fall back on scipy's spsolve.
try:
    from pypardiso import spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve


def create_demand_vector(
    activities_idx: List[int],
    A: scipy.sparse,
    demand: xr.DataArray,
    unit_conversion: np.ndarray,
) -> np.ndarray:
    """
    Create a demand vector with the given activities indices, sparse matrix A, demand values, and unit conversion factors.

    This function multiplies the given demand with the unit conversion factors and assigns the result to the positions in
    the vector corresponding to the activities indices. All other positions in the vector are set to zero.

    :param activities_idx: Indices of activities for which to create demand. These indices correspond to positions in the vector and matrix A.
    :type activities_idx: List[int]

    :param A: Sparse matrix used to determine the size of the demand vector.
    :type A: scipy.sparse.csr_matrix

    :param demand: Demand values for the activities, provided as a DataArray from the xarray package.
    :type demand: xr.DataArray

    :param unit_conversion: Unit conversion factors corresponding to each activity in activities_idx.
    :type unit_conversion: numpy.ndarray

    :return: The demand vector, represented as a 1-dimensional numpy array.
    :rtype: numpy.ndarray
    """

    # Initialize the demand vector with zeros, with length equal to the number of rows in A
    f = np.zeros(A.shape[0])

    # Assign demand values to the positions in the vector corresponding to the activities indices
    # Demand values are converted to the appropriate units before assignment
    f[activities_idx] = float(demand) * float(unit_conversion)

    return f


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
    num_indices: int,
    sign: int = 1,
    transpose: bool = False,
    extra_indices: Optional[int] = None,
    data_type: np.dtype = np.float32,
) -> csr_matrix:
    """
    Reads a CSV file and returns its contents as a CSR sparse matrix.

    :param file_path: The path to the CSV file.
    :type file_path: Path
    :param num_indices: The number of indices in the data.
    :type num_indices: int
    :param transpose: Whether or not to transpose the matrix. Default is False.
    :type transpose: bool
    :param extra_indices: Optional parameter to adjust the shape of the matrix. Default is None.
    :type extra_indices: Optional[int]

    :return: The data from the CSV file as a CSR sparse matrix.
    :rtype: csr_matrix
    """
    # Load the data from the CSV file
    coords = np.genfromtxt(file_path, delimiter=";", skip_header=1)

    # Extract the row and column indices and the data
    I = coords[:, 0].astype(int)
    J = coords[:, 1].astype(int)
    data = coords[:, 2] * sign

    # Determine the shape of the matrix
    if transpose:
        shape = (num_indices, I.max() + 1)
        idx = (J, I)
    else:
        shape = (
            extra_indices if extra_indices is not None else I.max() + 1,
            num_indices,
        )
        idx = (I, J)

    # Create and return the CSR matrix
    matrix = csr_matrix((data, idx), shape=shape, dtype=data_type)
    return matrix


def get_lca_matrices(
    datapackage: str, model: str, scenario: str, year: int, data_type: np.dtype = np.float32
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict, Dict]:
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

    A = load_matrix_and_index(dirpath / "A_matrix.csv", len(A_inds), transpose=True, data_type=data_type)
    B = load_matrix_and_index(
        dirpath / "B_matrix.csv", len(B_inds), sign=-1, extra_indices=len(A_inds), data_type=data_type
    )

    return A, B, A_inds, B_inds


def remove_double_counting(A: csr_matrix, vars_info: dict) -> csr_matrix:
    """
    Remove double counting from a technosphere matrix.
    :param A:
    :param activities_idx:
    :return:
    """

    # Modify A in COO format for efficiency
    # To avoid double-counting, set all entries in A corresponding
    # to activities not in activities_idx to zero

    A_coo = A.tocoo()

    for region in vars_info:
        for variable in vars_info[region]:
            idx = vars_info[region][variable]["idx"]
            row_mask = np.isin(A_coo.row, idx)
            col_mask = np.isin(A_coo.col, idx)
            A_coo.data[row_mask & ~col_mask] = 0  # zero out rows

    A_coo.eliminate_zeros()
    return A_coo.tocsr()


def characterize_inventory(C, lcia_matrix) -> np.ndarray:
    """
    Characterize an inventory with an LCIA matrix.
    :param C: Solved inventory
    :param lcia_matrix: Characterization matrix
    :return: Characterized inventory
    """

    # Multiply C with lcia_matrix to get D
    return C[..., None] * lcia_matrix


def solve_inventory(A: csr_matrix, B: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Solve the inventory problem for a set of activities, given technosphere and biosphere matrices, demand vector,
    LCIA matrix, and the indices of activities to consider.

    This function uses either the pypardiso or scipy library to solve the linear system, depending on the availability
    of the pypardiso library. The solutions are then used to calculate LCIA scores.

    ...

    :rtype: numpy.ndarray

    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    if A.shape[0] != f.size:
        raise ValueError("Incompatible dimensions between A and f")

    if B.shape[0] != A.shape[0]:
        raise ValueError("Incompatible dimensions between A and B")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Solve the system Ax = f for x using sparse solver
        A_inv = spsolve(A, f)[:, np.newaxis]

    # Compute product of A_inv and B
    C = A_inv * B

    return C
