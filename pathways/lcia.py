import numpy as np
from scipy import sparse
from . import DATA_DIR
import json
import xarray as xr


LCIA_METHODS = DATA_DIR / "lcia_data.json"


def get_lcia_method_names():
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    return [" - ".join(x["name"]) for x in data]


def format_lcia_method_exchanges(method):
    """
    Format LCIA method data to fit such structure:
    (name, unit, type, category, subcategory, amount, uncertainty type, uncertainty amount)

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


def get_lcia_methods():
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    return {" - ".join(x["name"]): format_lcia_method_exchanges(x) for x in data}


def fill_characterization_factors_matrix(biosphere_flows: list, methods) -> np.ndarray:
    """
    Create a characterization matrix base don the list of biosphere flows
    given.
    :param biosphere_flows:
    :param methods: contains names of the methods to use.
    :return:
    """

    lcia_data = get_lcia_methods()

    # create a numpy array filled with zeros
    # of size equal to biosphere_flows and lcia methods

    cf_matrix = np.zeros((len(biosphere_flows), len(methods)))

    # fill the matrix
    for i, flow in enumerate(biosphere_flows):
        for j, method in enumerate(methods):
            try:
                cf_matrix[i, j] = lcia_data[method][flow[:3]]
            except KeyError:
                continue

    return cf_matrix
