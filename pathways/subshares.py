import logging
from collections import defaultdict

import bw2calc
import bw_processing as bwp
import numpy as np
import yaml
from premise.geomap import Geomap
from scipy.interpolate import interp1d
from stats_arrays import *

from pathways.filesystem_constants import DATA_DIR, USER_LOGS_DIR
from pathways.utils import get_activity_indices

SUBSHARES = DATA_DIR / "technologies_shares.yaml"

logging.basicConfig(
    level=logging.DEBUG,
    filename=USER_LOGS_DIR / "pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_subshares() -> dict:
    """
    Load a YAML file and return its content as a Python dictionary.
    :return: A dictionary with the categories, technologies and market shares data.
    """
    with open(SUBSHARES) as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError("Subshares data should be a dictionary.")

    return check_subshares(check_uncertainty_params(data))


def check_uncertainty_params(data):
    """
    Check if the uncertainty parameters are valid,
    according to stats_array specifications
    """

    MANDATORY_UNCERTAINTY_FIELDS = {
        0: {"loc"},
        1: {"loc"},
        2: {"loc", "scale"},
        3: {"loc", "scale"},
        4: {"minimum", "maximum"},
        5: {"loc", "minimum", "maximum"},
    }

    UNCERTAINTY = {
        "undefined": UndefinedUncertainty.id,
        "lognormal": LognormalUncertainty.id,
        "normal": NormalUncertainty.id,
        "uniform": UniformUncertainty.id,
        "triangular": TriangularUncertainty.id,
    }

    for group, technologies in data.items():
        for technology in technologies.values():
            if "share" in technology:
                for year, params in technology["share"].items():
                    params["uncertainty_type"] = UNCERTAINTY.get(
                        params.get("uncertainty_type", "undefined"),
                        UndefinedUncertainty.id,
                    )

                    if not all(
                        key in params
                        for key in MANDATORY_UNCERTAINTY_FIELDS[
                            params["uncertainty_type"]
                        ]
                    ):
                        logging.warning(
                            f"Missing mandatory uncertainty parameters for '{year}' in '{group}'"
                        )
    return data


def check_subshares(data: dict) -> dict:
    """
    Adjusts the values in 'data' for each year ensuring the sum equals 1
    for each category, excluding technologies without a name.
    It dynamically identifies years (integer keys) that contain a 'value'
    subkey and adjusts them.

    :param data: A dictionary with categories as keys, each category is
    a dictionary of subcategories containing a list of technology dictionaries.
    :return: A dictionary with the adjusted values.
    """

    for category, technologies in data.items():
        technologies_to_remove = []
        totals = defaultdict(float)
        for technology, params in technologies.items():
            name = params.get("name")
            if name in {"null", "Null", None} or not name.strip():
                logging.warning(
                    f"Technology '{technology}' in category '{category}' is being removed due to invalid name '{name}'."
                )
                technologies_to_remove.append(technology)
                continue
            if "share" in params:
                for year, share in params["share"].items():
                    if "loc" in share:
                        totals[year] += share["loc"]
            else:
                logging.warning(
                    f"Technology '{technology}' in category '{category}' does not have a 'share' key"
                )

        for tech in technologies_to_remove:
            del technologies[tech]

        for year, total_value in totals.items():
            if not np.isclose(total_value, 1.00, rtol=1e-3):
                logging.warning(
                    f"Total of '{year}' values in category '{category}' does not add up to 1.00 (Total: {total_value}). Adjusting values."
                )
                for technology, params in technologies.items():
                    if (
                        "share" in params
                        and year in params["share"]
                        and "loc" in params["share"][year]
                    ):
                        # Adjust the share value
                        params["share"][year]["loc"] /= total_value

    return data


def find_technology_indices(
    regions: list, technosphere_indices: dict, geo: Geomap
) -> dict:
    """
    Fetch the indices in the technosphere matrix for the specified technologies and regions.
    The function dynamically adapts to any integer year keys in the data, and populates details
    for each such year under each technology in each region.

    :param regions: List of region identifiers.
    :param technosphere_indices: Dictionary mapping activities to indices in the technosphere matrix.
    :param geo: Geomap object used for location mappings.
    :return: Dictionary keyed by technology categories, each containing a nested dictionary of regions
            to technology indices and year attributes.
    """
    technologies_dict = load_subshares()
    indices_dict = {}

    for region in regions:
        for tech_category, techs in technologies_dict.items():
            category_dict = indices_dict.setdefault(tech_category, {})

            for tech, info in techs.items():
                regional_indices = category_dict.setdefault(region, {})

                activity_key = create_activity_key(info, region)
                activity_index = get_activity_indices(
                    [activity_key], technosphere_indices, geo
                )[0]

                if activity_index is None:
                    print(
                        f"Warning: No activity index found for technology '{tech}' in region '{region}'."
                    )
                    continue

                tech_data = regional_indices.setdefault(tech, {"idx": activity_index})
                tech_data["share"] = info.get("share", {})

    return indices_dict


def create_activity_key(tech: dict, region: str) -> tuple:
    """
    Creates a tuple representing an activity with its technology specifications and region.
    This function forms part of a critical step in linking technology-specific data
    to a spatial database structure.

    :param tech: Dictionary containing technology details.
    :param region: String representing the region.
    :return: Tuple of technology name, reference product, unit, and region.
    """
    return tech.get("name"), tech.get("reference product"), tech.get("unit"), region


def get_subshares_matrix(
    correlated_array: list,
) -> bwp.datapackage.Datapackage:
    """
    Add subshares samples to a bw_processing.datapackage object.
    :param correlated_array: List containing the subshares samples.
    """

    dp_correlated = bwp.create_datapackage()
    a_data_samples, a_indices, a_sign = correlated_array

    dp_correlated.add_persistent_array(
        matrix="technosphere_matrix",
        indices_array=a_indices,
        data_array=a_data_samples,
        flip_array=a_sign,
    )

    return dp_correlated


def adjust_matrix_based_on_shares(
    lca: bw2calc.MultiLCA,
    shares_dict: dict,
    subshares: dict,
    year: int,
):
    """
    Adjust the technosphere matrix based on shares.
    :param lca: bw2calc.LCA object.
    :param shares_dict: Dictionary containing the shares data.
    :param subshares: Dictionary containing the subshares data.
    :param year: Integer representing the year.
    :return: Tuple containing the data, indices, and signs.
    """

    final_amounts = defaultdict(float)

    # get coordinates of nonzero values in the technosphere matrix
    nz_row, nz_col = lca.technosphere_matrix.nonzero()

    def get_nz_col_indices(index):
        rows = np.where(np.isin(nz_row, index))
        cols = nz_col[rows]
        # return only cols for which lca.technosphere_matrix[index, cols] < 0
        mask = lca.technosphere_matrix[index, cols].toarray() < 0
        return cols[mask[0]]

    for tech_category, regions in shares_dict.items():
        for region, technologies in regions.items():
            for technology in technologies.values():
                technology["consumer_idx"] = get_nz_col_indices(technology["idx"])

    for tech_category, regions in shares_dict.items():
        for region, technologies in regions.items():
            for name, tech in technologies.items():
                for consumer in tech["consumer_idx"]:
                    initial_amount = lca.technosphere_matrix[tech["idx"], consumer]

                    subshare = subshares[tech_category][year][name]

                    # Distribute the initial amount based on the subshare of the current technology
                    split_amount = initial_amount * subshare * -1

                    # Add the split amount to the combined amount for this technology and consumer
                    final_amounts[(tech["idx"], consumer)] += split_amount

                    for other_supplier, other_supplier_tech in technologies.items():
                        if other_supplier != name:
                            additional_amount = (
                                initial_amount
                                * subshares[tech_category][year][other_supplier]
                                * -1
                            )
                            final_amounts[
                                (other_supplier_tech["idx"], consumer)
                            ] += additional_amount

    # Prepare the list of indices and amounts for the modified technosphere matrix
    list_indices = []
    list_amounts = []

    # Now, append the combined amounts to list_amounts and list_indices
    for (tech_idx, consumer_idx), total_amount in final_amounts.items():
        list_indices.append((tech_idx, consumer_idx))
        list_amounts.append(tuple(total_amount))

        logging.info(
            f"Final combined amount for tech index {tech_idx} to consumer index {consumer_idx}: {total_amount}"
        )

    indices = np.array(list_indices, dtype=bwp.INDICES_DTYPE)
    data = np.array(list_amounts)
    signs = np.ones_like(indices, dtype=bool)

    return data, indices, signs


def default_dict_factory():
    return defaultdict(dict)


def load_and_normalize_shares(
    ranges: dict,
    iterations: int,
) -> dict:
    """
    Load and normalize shares for parameters to sum to 1 while respecting their specified ranges.
    :param ranges: A dictionary with categories, technologies and market shares data.
    :param iterations: Number of iterations for random generation.
    :return: A dict with normalized shares for each technology and year.
    """
    # shares = defaultdict(lambda: defaultdict(dict)) # Gives problems for pickling in multiprocessing
    shares = defaultdict(default_dict_factory)

    for technology_group, technologies in ranges.items():
        for technology, params in technologies.items():
            for y, share in params["share"].items():
                uncertainty_base = UncertaintyBase.from_dicts(share)
                random_generator = MCRandomNumberGenerator(
                    params=uncertainty_base,
                )
                shares[technology_group][y][technology] = np.squeeze(
                    np.array([random_generator.next() for _ in range(iterations)])
                )

    totals = defaultdict(lambda: np.array([]))

    for technology_group, years in shares.items():
        for year, technologies in years.items():
            arrays = [
                shares[technology_group][year][technology]
                for technology in technologies
            ]
            if len(arrays) > 0:
                if year not in totals[(technology_group, year)]:
                    totals[(technology_group, year)] = np.sum(arrays, axis=0)
                else:
                    totals[(technology_group, year)] += np.sum(arrays, axis=0)

    for technology_group, technologies in ranges.items():
        for technology, params in technologies.items():
            for y, share in params["share"].items():
                normalized = (
                    shares[technology_group][y][technology]
                    / totals[(technology_group, y)]
                )
                normalized = np.clip(
                    normalized,
                    ranges[technology_group][technology]["share"][y].get("minimum", 0),
                    ranges[technology_group][technology]["share"][y].get("maximum", 1),
                )
                shares[technology_group][y][technology] = normalized

    # Ensure that `shares` is a dictionary
    if not isinstance(shares, dict):
        raise ValueError("Normalized shares should be a dictionary.")

    return shares


def interpolate_shares(shares: dict, years: list) -> None:
    """
    Interpolates missing years in the shares data.
    :param shares: A dictionary with categories, technologies and market shares data.
    :param years: List of years for which to interpolate shares.
    :return: None
    """
    for technology_group in shares:
        all_years = sorted(shares[technology_group])
        for year in years:
            if year in shares[technology_group]:
                continue
            lower_year = max((y for y in all_years if y <= year), default=None)
            upper_year = min((y for y in all_years if y >= year), default=None)
            if lower_year and upper_year:
                interpolate_for_year(
                    shares, technology_group, lower_year, upper_year, year
                )


def interpolate_for_year(
    shares: dict,
    technology_group: str,
    lower_year: int,
    upper_year: int,
    target_year: int,
) -> None:
    """
    Interpolates shares for a specific year.
    :param shares: A dictionary with categories, technologies and market shares data.
    :param technology_group: A string representing the technology group.
    :param lower_year: An integer representing the lower year.
    :param upper_year: An integer representing the upper year.
    :param target_year: An integer representing the target year.
    :return: None
    """
    for technology in shares[technology_group][lower_year]:

        shares_lower = shares[technology_group][lower_year][technology]
        shares_upper = shares[technology_group][upper_year][technology]

        if shares_lower.ndim == 1:
            shares_lower = shares_lower.reshape(1, -1)
        if shares_upper.ndim == 1:
            shares_upper = shares_upper.reshape(1, -1)

        # Stack arrays vertically and ensure it is 2xN (2 rows, N columns)
        s = np.vstack((shares_lower, shares_upper)).T

        f = interp1d([lower_year, upper_year], s, kind="linear")
        shares[technology_group][target_year][technology] = f(target_year)


def generate_samples(
    years: list,
    iterations: int = 10,
) -> dict:
    """
    Generates and adjusts randomly selected shares for parameters to sum to 1
    while respecting their specified ranges, and interpolates missing years.

    :param years: List of years for which to generate/interpolate shares.
    :param iterations: Number of iterations for random generation.
    :return: A dict with adjusted and interpolated shares for each technology and year.
    """
    ranges = load_subshares()
    shares = load_and_normalize_shares(
        ranges,
        iterations,
    )
    interpolate_shares(shares, years)
    return shares
