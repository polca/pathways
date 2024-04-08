import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bw_processing as bwp
import numpy as np
import yaml
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


def get_matrix_arrays(
    dirpath: Path,
    matrix_type: str,
) -> list:
    """
    Retrieve matrix arrays from disk.

    ...

    :rtype: List of np.ndarrays
    """

    matrix_filename = f"{matrix_type}_matrix.csv"

    data, indices, sign, distributions = load_matrix_and_index(
        dirpath / matrix_filename,
    )

    return [data, indices, sign, distributions]


def get_indices(
    dirpath: Path,
) -> Tuple[Dict, Dict]:
    """
    Build the technosphere and biosphere indices.

    ...

    :rtype: Tuple[Dict, Dict]
    """

    A_indices = read_indices_csv(dirpath / "A_matrix_index.csv")
    B_indices = read_indices_csv(dirpath / "B_matrix_index.csv")

    return A_indices, B_indices


def get_lca_matrices(
    A_arrays: list,
    B_arrays: list,
) -> Datapackage:
    """
    Build the technosphere and biosphere matrices from matrix arrays.

    ...

    :rtype: Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict, Dict]
    """

    # create brightway datapackage
    dp = bwp.create_datapackage()

    a_data, a_indices, a_sign, a_distributions = A_arrays
    b_data, b_indices, b_distributions = B_arrays

    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        indices_array=a_indices,
        data_array=a_data,
        flip_array=a_sign,
        distributions_array=a_distributions,
    )

    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        indices_array=b_indices,
        data_array=b_data,
        distributions_array=b_distributions,
    )

    return dp


def get_subshares_matrix(
    A_array: list,
    indices: Dict,
    year: int,
) -> Datapackage:
    """
    Add subshares samples to an LCA object.
    :
    """

    dp_correlated = bwp.create_datapackage()

    a_data, a_indices, a_sign, _ = A_array

    # adjusted_data = adjust_matrix_based_on_shares(data_array, indices_array, indices, year)

    dp_correlated.add_persistent_array(
        matrix="technosphere_matrix",
        indices_array=a_indices,
        data_array=a_data,
        flip_array=a_sign,
    )

    return dp_correlated


def adjust_matrix_based_on_shares(data_array, indices_array, shares_dict, year):
    """
    Adjust the technosphere matrix based on shares.
    :param data_array:
    :param indices_array:
    :param shares_dict:
    :param year:
    :return:
    """
    ##### RIGHT NOW I AM ONLY MULTIPLYING BY THE SHARES IN 2020 - TO-DO: ADD THE SAMPLES
    #                                                             TO-DO: RETURN THE LAST ARRAY NEEDED TO BUILD THE BW.PACKAGE

    index_lookup = {(row["row"], row["col"]): i for i, row in enumerate(indices_array)}

    modified_data = []
    modified_indices = []

    # Determine unique product indices from shares_dict to identify which shouldn't be automatically updated/added
    unique_product_indices_from_dict = set()
    for _, regions in shares_dict.items():
        for _, techs in regions.items():
            for _, details in techs.items():
                if "idx" in details:
                    unique_product_indices_from_dict.add(details["idx"])

    # Helper function to find index using the lookup dictionary
    def find_index(activity_idx, product_idx):
        return index_lookup.get((activity_idx, product_idx))

    for tech_category, regions in shares_dict.items():
        for region, techs in regions.items():
            all_tech_indices = [
                techs[tech]["idx"] for tech in techs if techs[tech]["idx"] is not None
            ]
            all_product_indices = set()

            # Optimization: Use np.isin for efficient filtering
            tech_indices = np.isin(indices_array["row"], all_tech_indices)
            all_product_indices.update(indices_array["col"][tech_indices])

            for product_idx in all_product_indices:
                # Vectorized operation to calculate total_output for each product_idx
                relevant_indices = [
                    find_index(tech_idx, product_idx)
                    for tech_idx in all_tech_indices
                    if find_index(tech_idx, product_idx) is not None
                    and tech_idx != product_idx
                ]
                total_output = np.sum(data_array[relevant_indices])

                for tech, details in techs.items():
                    share = details.get(year, {}).get("value", 0)
                    idx = details["idx"]
                    if idx is None or share == 0:
                        continue

                    # Calculate new amount
                    new_amount = total_output * share
                    index = find_index(idx, product_idx)

                    # Adjust value or add new exchange
                    if (
                        index is not None
                        and product_idx not in unique_product_indices_from_dict
                    ):  # Exclude diagonal and undesired exchanges
                        data_array[index] = new_amount
                        # Append to modified_indices regardless of whether it's a new addition or an adjustment
                        modified_indices.append((idx, product_idx))
                        modified_data.append(new_amount)
                    elif (
                        product_idx not in unique_product_indices_from_dict
                    ):  # Exclude diagonal and undesired exchanges
                        modified_data.append(new_amount)
                        modified_indices.append((idx, product_idx))

    modified_data_array = np.array(modified_data, dtype=np.float64)
    modified_indices_array = np.array(modified_indices, dtype=bwp.INDICES_DTYPE)
    # modified_flip_array = np.array(modified_flip, dtype=flip_array.dtype)

    return modified_data_array, modified_indices_array


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


# def remove_double_counting(
#     characterized_inventory: csr_matrix, vars_info: dict, activity_idx: int
# ) -> csr_matrix:
#     """
#     Remove double counting from a characterized inventory matrix for all activities except
#     the activity being evaluated, across all methods.
#
#     :param characterized_inventory: Characterized inventory matrix with rows for different methods and columns for different activities.
#     :param vars_info: Dictionary with information about which indices to zero out.
#     :param activity_idx: Index of the activity being evaluated, which should not be zeroed out.
#     :return: Characterized inventory with double counting removed for all but the evaluated activity.
#     """
#
#     print("Removing double counting")
#     if isinstance(characterized_inventory, np.ndarray):
#         characterized_inventory = csr_matrix(characterized_inventory)
#     elif not isinstance(characterized_inventory, csr_matrix):
#         raise TypeError(
#             "characterized_inventory must be a csr_matrix or a numpy array."
#         )
#
#     # Gather all indices for which we want to avoid double counting, except the evaluated activity
#     list_of_idx_to_remove = []
#     for region in vars_info:
#         for variable in vars_info[region]:
#             idx = vars_info[region][variable]
#             if idx != activity_idx:
#                 list_of_idx_to_remove.append(idx)
#
#     # Convert to lil_matrix for more efficient element-wise operations - CHECK IF THIS IS ACTUALLY FASTER
#     characterized_inventory = characterized_inventory.tolil()
#
#     # Zero out the specified indices for all methods, except for the activity being evaluated
#     for idx in list_of_idx_to_remove:
#         characterized_inventory[:, idx] = 0
#
#     return characterized_inventory.tocsr()


def remove_double_counting(A: csr_matrix, vars_info: dict) -> csr_matrix:
    """
    Remove double counting from a technosphere matrix.
    :param A: Technosphere matrix
    :param vars_info: Dictionary with information about which indices to zero out.
    :return: Technosphere matrix with double counting removed.
    """

    A_coo = A.tocoo()

    list_of_idx = []

    for region in vars_info:
        for variable in vars_info[region]:
            idx = vars_info[region][variable]["idx"]
            if idx not in list_of_idx:
                list_of_idx.append(idx)
                row_mask = np.isin(A_coo.row, idx)
                col_mask = np.isin(A_coo.col, idx)
                A_coo.data[row_mask & ~col_mask] = 0

    A_coo.eliminate_zeros()
    return A_coo.tocsr()
