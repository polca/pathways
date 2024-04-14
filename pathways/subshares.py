from collections import defaultdict

import bw2calc
import bw_processing
import bw_processing as bwp
import numpy as np
import yaml
import logging
from bw_processing import Datapackage
from premise.geomap import Geomap
from stats_arrays import *
from scipy.interpolate import interp1d

from pathways.filesystem_constants import DATA_DIR
from pathways.utils import get_activity_indices

SUBSHARES = DATA_DIR / "technologies_shares.yaml"

logging.basicConfig(
    level=logging.DEBUG,
    filename="pathways.log",  # Log file to save the entries
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

    return adjust_subshares(check_uncertainty_params(data))


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
                    params["uncertainty_type"] = UNCERTAINTY.get(params.get("uncertainty_type", "undefined"), UndefinedUncertainty.id)


                    if not all(
                            key in params for key in MANDATORY_UNCERTAINTY_FIELDS[params["uncertainty_type"]]
                    ):
                        logging.warning(
                            f"Missing mandatory uncertainty parameters for '{year}' in '{group}'"

                        )
    return data


def adjust_subshares(data: dict) -> dict:
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
        totals = defaultdict(float)
        for technology, params in technologies.items():
            if "share" in params:
                for year, share in params["share"].items():
                    if "loc" in share:
                        totals[year] += share["loc"]
            else:
                logging.warning(f"Technology '{technology}' in category '{category}' does not have a 'share' key")

        # if any values of totals is not equal to 1, we log it
        for year, total_value in totals.items():
            if not np.isclose(total_value, 1.00, rtol=1e-3):
                logging.warning(
                    f"Total of '{year}' values in category '{category}' does not add up to 1.00 (Total: {total_value})"
                )
    return data


def find_technology_indices(regions: list, technosphere_indices: dict, geo: Geomap) -> dict:
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
                    continue

                tech_data = regional_indices.setdefault(
                    tech, {"idx": activity_index}
                )
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
        lca: bw2calc.LCA,
        shares_dict: dict,
        use_distributions: int,
        year: int,
):
    """
    Adjust the technosphere matrix based on shares.
    :param lca: bw2calc.LCA object.
    :param shares_dict: Dictionary containing the shares data.
    :param use_distributions: Number of iterations.
    :param year: Integer representing the year.
    :return:
    """

    modified_data = []
    modified_indices = []
    modified_signs = []

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
                technology["consumer_idx"] = get_nz_col_indices(technology['idx'])
            correlated_samples(ranges=technologies, year=year, iterations=use_distributions)

    for tech_category, regions in shares_dict.items():
        for region, technologies in regions.items():
            for tech in technologies.values():
                for consumer in tech["consumer_idx"]:
                    logging.debug(f"Consumer: {consumer} receiving from {tech['idx']}: {lca.technosphere_matrix[tech['idx'], consumer]}")
                total_amount = lca.technosphere_matrix[np.ix_(
                    np.array([tech["idx"] for tech in technologies.values()]),
                    np.hstack([tech["consumer_idx"] for tech in technologies.values()]),
                )].sum()

                logging.debug(f"Tech: {technologies}, Total amount: {total_amount}")


    for tech, details in tech.items():
        if year != 2020:
            if details[2050]["distribution"] == "uniform":
                tech_group_ranges[tech] = (
                    details[2050]["min"],
                    details[year]["max"],
                )
                tech_group_defaults[tech] = details.get(2020, {}).get(
                    "value", 0
                )
            else:
                print(
                    "At this point, only uniform distributions are supported. Exiting."
                )
                exit(1)

    if year != 2020 and tech_group_ranges:
        group_shares = correlated_samples(
            tech_group_ranges, tech_group_defaults
        )
        print("Tech group", tech_category, "shares: ", group_shares)
    else:
        group_shares = {
            tech: details.get(year, {}).get("value", 0)
            for tech, details in tech.items()
        }
        print("Tech group", tech_category, "shares: ", group_shares)

    for product_idx in all_product_indices:
        relevant_indices = [
            lca.dicts.product[idx]
            for idx in all_tech_indices
            if lca.dicts.product[idx] is not None
        ]
        total_output = np.sum(
            lca.technosphere_matrix[product_idx, relevant_indices]
        )

        for tech, share in group_shares.items():
            if (
                    tech in tech
                    and "idx" in tech[tech]
                    and tech[tech]["idx"] is not None
            ):
                idx = tech[tech]["idx"]

                if year == 2020:
                    share_value = details.get(year, {}).get("value", 0)
                    new_amounts = np.array(
                        [total_output * share_value]
                    ).reshape((1, -1))
                else:
                    new_amounts = np.array(
                        [total_output * share for _ in range(use_distributions)]
                    ).reshape((1, -1))
                index = find_index(idx, product_idx)

                if (
                        index is not None
                        and product_idx not in unique_product_indices_from_dict
                ):
                    modified_indices.append((idx, product_idx))
                    modified_data.append(new_amounts)
                    modified_signs.append(sign_array[index])
                elif (
                        index is None
                        and product_idx not in unique_product_indices_from_dict
                ):
                    modified_data.append(new_amounts)
                    modified_indices.append((idx, product_idx))
                    modified_signs.append(True)

    # modified_data_array = np.array(modified_data, dtype=object)
    modified_data_array = np.concatenate(modified_data, axis=0)
    modified_indices_array = np.array(modified_indices, dtype=bwp.INDICES_DTYPE)
    modified_signs_array = np.array(modified_signs, dtype=bool)

    return [modified_data_array, modified_indices_array, modified_signs_array]


def correlated_samples(ranges: dict, year: int, iterations=10):
    """
    Adjusts randomly selected shares for parameters to sum to 1
    while respecting their specified ranges.

    :param ranges: Dict with parameter names as keys and (min, max) tuples as values.
    :param defaults: Dict with default values for each parameter.
    :param iterations: Number of iterations to attempt to find a valid distribution.
    :return: A dict with the adjusted shares for each parameter.
    """

    shares = defaultdict(dict)
    for technology, params in ranges.items():
        for y, share in params["share"].items():
            u = UncertaintyBase.from_dicts(share)
            r = MCRandomNumberGenerator(u)
            shares[y][technology] = np.array([r.next() for _ in range(iterations)])

    for y in shares:
        total = np.hstack([shares[y][technology] for technology in shares[y]])
        # normalize by the sum of all shares
        shares[y] = {technology: np.clip(
            np.squeeze(share) / total.sum(axis=1),
            ranges[technology]["share"].get("min", 0),
            ranges[technology]["share"].get("max", 1),
        ) for technology, share in shares[y].items()}

    # interpolate the values to `year`
    # find the lowest year
    lowest_year = min(shares.keys())
    # find the highest year
    highest_year = max(shares.keys())

    # update the ranges with the new values
    for technology, params in ranges.items():
        if year == lowest_year:
            shares["iterations"][technology] = shares[lowest_year][technology]
        elif year == highest_year:
            shares["iterations"][technology] = shares[highest_year][technology]
        else:
            interpolator = interp1d(
                [lowest_year, highest_year],
                np.array([
                    shares[lowest_year][technology],
                    shares[highest_year][technology]
                ]),
                axis=0
            )
            shares["iterations"][technology] = interpolator(year)

    return shares

