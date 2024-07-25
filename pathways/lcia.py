"""
This module contains functions to list, and LCIA methods and fill the LCIA characterization matrix.
"""

import json
import logging

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

from .filesystem_constants import DATA_DIR

LCIA_METHODS = DATA_DIR / "lcia_ei310.json"


def get_lcia_method_names():
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    return [" - ".join(x["name"]) for x in data]


def format_lcia_method_exchanges(method):
    """
        Format LCIA method data to fit such structure:
        (name, unit, type, category, subcategory, amount, uncertainty type, uncertainty amount)
    -
        :param method: LCIA method
        :return: list of tuples
    """

    return {
        (
            x["name"],
            x["categories"][0],
            x["categories"][1] if len(x["categories"]) > 1 else "unspecified",
        ): x["amount"]
        for x in method["exchanges"]
    }


def get_lcia_methods(methods: list = None):
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    if methods:
        data = [x for x in data if " - ".join(x["name"]) in methods]

    return {" - ".join(x["name"]): format_lcia_method_exchanges(x) for x in data}


def fill_characterization_factors_matrices(
    methods: list, biosphere_matrix_dict: dict, biosphere_dict: dict, debug=False
) -> csr_matrix:
    """
    Create one CSR matrix for all LCIA method, with the last dimension being the index of the method
    :param methods: contains names of the LCIA methods to use (e.g., ["IPCC 2021, Global wArming Potential"]).
    :param biosphere_matrix_dict: dictionary with biosphere flows and their indices in bw2calc's matrix
    :param biosphere_dict: dictionary with biosphere flows and their indices in the biosphere matrix (not bw2calc's matrix)
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
