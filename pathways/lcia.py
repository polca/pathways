"""
This module contains functions to list, and LCIA methods and fill the LCIA characterization matrix.
"""

import json
import logging

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

from .filesystem_constants import DATA_DIR

LCIA_METHODS_EI310 = DATA_DIR / "lcia_ei310.json"
LCIA_METHODS_EI311 = DATA_DIR / "lcia_ei311.json"


def get_lcia_method_names(ei_version="3.11"):
    """List LCIA method names bundled with the specified ecoinvent version.

    :param ei_version: Ecoinvent release identifier (e.g. ``"3.11"``).
    :type ei_version: str
    :returns: Ordered method names formatted as ``"family - method"``.
    :rtype: list[str]
    """

    if ei_version != "3.11":
        filepath = LCIA_METHODS_EI311
    else:
        filepath = LCIA_METHODS_EI310

    with open(filepath, "r") as f:
        data = json.load(f)

    return [" - ".join(x["name"]) for x in data]


def format_lcia_method_exchanges(method):
    """Map an LCIA method's exchanges to impact amounts keyed by flow identity.

    :param method: LCIA method object as loaded from the JSON descriptor.
    :type method: dict
    :returns: Mapping from ``(flow name, category, subcategory)`` to characterization values.
    :rtype: dict[tuple[str, str, str], float]
    """

    return {
        (
            x["name"],
            x["categories"][0],
            x["categories"][1] if len(x["categories"]) > 1 else "unspecified",
        ): x["amount"]
        for x in method["exchanges"]
    }


def get_lcia_methods(methods: list = None, ei_version="3.11"):
    """Load selected LCIA methods and format their exchanges.

    :param methods: Optional subset of method names to extract.
    :type methods: list[str] | None
    :param ei_version: Ecoinvent release identifier to read.
    :type ei_version: str
    :returns: Mapping of method names to formatted exchange dictionaries.
    :rtype: dict[str, dict[tuple[str, str, str], float]]
    """

    if ei_version != "3.11":
        filepath = LCIA_METHODS_EI311
    else:
        filepath = LCIA_METHODS_EI310

    with open(filepath, "r") as f:
        data = json.load(f)

    if methods:
        data = [x for x in data if " - ".join(x["name"]) in methods]

    return {" - ".join(x["name"]): format_lcia_method_exchanges(x) for x in data}


def fill_characterization_factors_matrices(
    methods: list, biosphere_matrix_dict: dict, biosphere_dict: dict, debug=False
) -> csr_matrix:
    """Assemble a CSR matrix with characterization factors for multiple LCIA methods.

    :param methods: Ordered method names to include.
    :type methods: list[str]
    :param biosphere_matrix_dict: Mapping of biosphere flows to rows in the bw2calc matrix.
    :type biosphere_matrix_dict: dict[int, int]
    :param biosphere_dict: Mapping of flow descriptors to biosphere indices.
    :type biosphere_dict: dict[tuple[str, str, str], int]
    :param debug: Flag to emit detailed logging about matched factors.
    :type debug: bool
    :returns: CSR matrix with shape ``(len(methods), len(biosphere_matrix_dict))``.
    :rtype: scipy.sparse.csr_matrix
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
