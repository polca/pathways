import json

import numpy as np
import xarray as xr
from scipy import sparse

from . import DATA_DIR

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


def get_lcia_methods(methods: list = None):
    """Get a list of available LCIA methods."""
    with open(LCIA_METHODS, "r") as f:
        data = json.load(f)

    if methods:
        data = [x for x in data if " - ".join(x["name"]) in methods]

    return {" - ".join(x["name"]): format_lcia_method_exchanges(x) for x in data}



