import bw_processing as bwp
import yaml
import numpy as np
from bw_processing import Datapackage

from pathways.filesystem_constants import DATA_DIR
from pathways.utils import get_activity_indices

SUBSHARES = DATA_DIR / "technologies_shares.yaml"


def load_subshares() -> dict:
    """
    Load a YAML file and return its content as a Python dictionary.
    :return: A dictionary with the categories, technologies and market shares data.
    """
    with open(SUBSHARES, "r") as stream:
        data = yaml.safe_load(stream)

    adjust_subshares(data)
    return data


def adjust_subshares(data):
    """
    Adjust the subshares data to ensure that the sum of the 2020 values is equal to 1, after neglecting the technologies
    with no name.
    :param data: Dictionary with the categories, technologies and market shares data.
    :return: Adjusted dictionary.
    """
    for category, technologies in data.items():

        # Initialize totals
        total_2020_value = 0
        total_adjustable_value = 0

        # First pass to calculate totals
        for subcategory, tech_list in technologies.items():
            for tech in tech_list:
                if 2020 in tech:
                    value = tech[2020].get("value", 0)
                    total_2020_value += value
                    if tech.get("name") is not None:
                        total_adjustable_value += value

        # Skip adjustment if no values or all values are named
        if total_2020_value == 0 or total_adjustable_value == 0:
            continue

        adjustment_factor = total_2020_value / total_adjustable_value

        # Second pass to adjust values
        adjusted_total = 0
        for subcategory, tech_list in technologies.items():
            for tech in tech_list:
                if 2020 in tech and tech.get("name") is not None:
                    tech[2020]["value"] = tech[2020]["value"] * adjustment_factor
                    adjusted_total += tech[2020]["value"]

        # Check if the adjusted total is close to 1.00, allowing a small margin for floating-point arithmetic
        if not np.isclose(adjusted_total, 1.00, rel_tol=1e-9):
            print(
                f"Warning: Total of adjusted '2020' values in category '{category}' does not add up to 1.00 (Total: {adjusted_total})"
            )


def subshares_indices(regions, A_index, geo):
    """
    Fetch the indices in the technosphere matrix from the activities in technologies_shares.yaml in
    the given regions.
    :param regions: List of regions
    :param A_index: Dictionary with the indices of the activities in the technosphere matrix.
    :param geo: Geomap object.
    :return: dictionary of technology categories and their indices.
    """
    technologies_dict = load_subshares()

    indices_dict = {}

    for region in regions:
        for tech_category, techs in technologies_dict.items():
            if tech_category not in indices_dict:
                indices_dict[tech_category] = {}
            for tech_type, tech_list in techs.items():
                for tech in tech_list:
                    name = tech.get("name")
                    ref_prod = tech.get("reference product")
                    unit = tech.get("unit")
                    activity = (name, ref_prod, unit, region)
                    activity_index = get_activity_indices([activity], A_index, geo)[0]

                    value_2020 = tech.get(2020, {}).get("value")
                    min_2050 = tech.get(2050, {}).get("min")
                    max_2050 = tech.get(2050, {}).get("max")
                    distribution_2050 = tech.get(2050, {}).get("distribution")

                    if region not in indices_dict[tech_category]:
                        indices_dict[tech_category][region] = {}
                    indices_dict[tech_category][region][tech_type] = {
                        "idx": activity_index,
                        2020: {"value": value_2020},
                        2050: {
                            "min": min_2050,
                            "max": max_2050,
                            "distribution": distribution_2050,
                        },
                    }

    return indices_dict


def get_subshares_matrix(
        correlated_array: list,
) -> Datapackage:
    """
    Add subshares samples to an LCA object.
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


def adjust_matrix_based_on_shares(A_arrays, shares_dict, use_distributions, year):
    """
    Adjust the technosphere matrix based on shares.
    :param data_array:
    :param indices_array:
    :param shares_dict:
    :param year:
    :return:
    """

    data_array, indices_array, sign_array, _ = A_arrays
    index_lookup = {(row["row"], row["col"]): i for i, row in enumerate(indices_array)}

    modified_data = []
    modified_indices = []
    modified_signs = []

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
                details["idx"] for _, details in techs.items() if "idx" in details
            ]
            all_product_indices = set(
                indices_array["col"][np.isin(indices_array["row"], all_tech_indices)]
            )

            tech_group_ranges = {}
            tech_group_defaults = {}

            for tech, details in techs.items():
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
                group_shares = correlated_uniform_samples(
                    tech_group_ranges, tech_group_defaults
                )
                print("Tech group", tech_category, "shares: ", group_shares)
            else:
                group_shares = {
                    tech: details.get(year, {}).get("value", 0)
                    for tech, details in techs.items()
                }
                print("Tech group", tech_category, "shares: ", group_shares)

            for product_idx in all_product_indices:
                relevant_indices = [
                    find_index(idx, product_idx)
                    for idx in all_tech_indices
                    if find_index(idx, product_idx) is not None
                ]
                total_output = np.sum(data_array[relevant_indices])

                for tech, share in group_shares.items():
                    if (
                            tech in techs
                            and "idx" in techs[tech]
                            and techs[tech]["idx"] is not None
                    ):
                        idx = techs[tech]["idx"]

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


def correlated_uniform_samples(ranges, defaults, iterations=1000):
    """
    Adjusts randomly selected shares for parameters to sum to 1 while respecting their specified ranges.

    :param ranges: Dict with parameter names as keys and (min, max) tuples as values.
    :param defaults: Dict with default values for each parameter.
    :param iterations: Number of iterations to attempt to find a valid distribution.
    :return: A dict with the adjusted shares for each parameter.
    """
    for _ in range(iterations):
        shares = {
            param: np.random.uniform(low, high) for param, (low, high) in ranges.items()
        }
        total_share = sum(shares.values())
        shares = {param: share / total_share for param, share in shares.items()}
        if all(
                ranges[param][0] <= share <= ranges[param][1]
                for param, share in shares.items()
        ):
            return shares

    print("Failed to find a valid distribution after {} iterations".format(iterations))
    return defaults
