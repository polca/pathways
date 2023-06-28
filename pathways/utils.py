import yaml
import numpy as np

from . import DATA_DIR

CLASSIFICATIONS = DATA_DIR / "activities_classifications.yaml"
UNITS_CONVERSION = DATA_DIR / "units_conversion.yaml"

def load_classifications():
    """Load the activities classifications."""

    with open(CLASSIFICATIONS, "r") as f:
        data = yaml.full_load(f)

    # ensure that NO is not interpreted as False
    new_keys = []
    old_keys = []
    for key, value in data.items():
        # check if last element of key is not nan
        if not isinstance(key[-1], str):
            new_entry = key[:-1] + ("NO",)
            new_keys.append(new_entry)
            old_keys.append(key)

    for new_key, old_key in zip(new_keys, old_keys):
        data[new_key] = data.pop(old_key)

    return data

def load_units_conversion():
    """Load the units conversion."""

    with open(UNITS_CONVERSION, "r") as f:
        data = yaml.full_load(f)

    return data


