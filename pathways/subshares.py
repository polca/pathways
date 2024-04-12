import bw2calc
import bw_processing
import bw_processing as bwp
import yaml
import numpy as np
from bw_processing import Datapackage
from premise.geomap import Geomap

from pathways.filesystem_constants import DATA_DIR
from pathways.utils import get_activity_indices

SUBSHARES = DATA_DIR / "technologies_shares.yaml"


def load_subshares() -> dict:
    """
    Load a YAML file and return its content as a Python dictionary.
    :return: A dictionary with the categories, technologies and market shares data.
    """
    with open(SUBSHARES) as stream:
        data = yaml.safe_load(stream)

    data = adjust_subshares(data)
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

    values_to_adjust = []
    for category, technologies in data.items():
        years = identify_years(technologies)
        for year in years:
            total_value, total_adjustable_value = compute_totals(technologies, year)
            if total_value == 0 or total_adjustable_value == 0:
                continue
            values_to_adjust.append((technologies, total_value, total_adjustable_value, category, year))

    for technologies, total_value, total_adjustable_value, category, year in values_to_adjust:
        technologies = adjust_values(technologies, total_value, total_adjustable_value, category, year)
        data[category] = technologies

    return data


def identify_years(technologies: dict) -> set:
    """
    Identify the years in the technologies dictionary that contain a 'value' subkey.
    :param technologies: A dictionary of subcategories containing a list of technology dictionaries.
    :return: A set of years.
    """
    years = set()
    for subcategory, tech_list in technologies.items():
        for tech in tech_list:
            year_keys = [key for key in tech.keys() if isinstance(key, int) and 'value' in tech[key]]
            years.update(year_keys)
    return years


def compute_totals(technologies: dict, year: int) -> tuple:
    """
    Compute the total value and total adjustable value for the given year.
    :param technologies: A dictionary of subcategories containing a list of technology dictionaries.
    :param year: int. A year to compute totals for.
    :return: A tuple with the total value and total adjustable value.
    """
    total_value = 0
    total_adjustable_value = 0
    for subcategory, tech_list in technologies.items():
        for tech in tech_list:
            if year in tech:
                value = tech[year].get('value', 0)
                total_value += value
                if tech.get("name") is not None:
                    total_adjustable_value += value
    return total_value, total_adjustable_value


def adjust_values(
        technologies: dict,
        total_value: float,
        total_adjustable_value: float,
        category: str,
        year: int
) -> dict:
    """
    Adjust the values in the technologies dictionary for the given year.
    :param technologies:
    :param total_value:
    :param total_adjustable_value:
    :param category:
    :param year:
    :return:
    """
    adjustment_factor = total_value / total_adjustable_value
    adjusted_total = 0
    for subcategory, tech_list in technologies.items():
        for tech in tech_list:
            if year in tech and tech.get("name") is not None:
                old_value = tech[year]["value"]
                tech[year]["value"] = old_value * adjustment_factor
                adjusted_total += tech[year]["value"]

    if not np.isclose(adjusted_total, 1.00, rtol=1e-9):
        print(
            f"Warning: Total of adjusted '{year}' values in category '{category}' "
            f"does not add up to 1.00 (Total: {adjusted_total})")

    return technologies


def subshares_indices(regions: list, technosphere_indices: dict, geo: Geomap) -> dict:
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

            for tech_type, tech_list in techs.items():
                regional_indices = category_dict.setdefault(region, {})

                for tech in tech_list:
                    activity_key = create_activity_key(tech, region)
                    activity_index = get_activity_indices([activity_key], technosphere_indices, geo)[0]

                    tech_data = regional_indices.setdefault(tech_type, {"idx": activity_index})

                    # Populate dynamic year data
                    for key, value in tech.items():
                        if isinstance(key, int):  # Year identified
                            tech_data[key] = value

    return indices_dict


def create_activity_key(tech, region):
    """
    Creates a tuple representing an activity with its technology specifications and region.
    This function forms part of a critical step in linking technology-specific data
    to a spatial database structure.

    :param tech: Dictionary containing technology details.
    :param region: String representing the region.
    :return: Tuple of technology name, reference product, unit, and region.
    """
    return (
        tech.get("name"),
        tech.get("reference product"),
        tech.get("unit"),
        region
    )


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


def adjust_matrix_based_on_shares(lca: bw2calc.LCA, shares_dict, use_distributions, year):
    """
    Adjust the technosphere matrix based on shares.
    :param data_array:
    :param indices_array:
    :param shares_dict:
    :param year:
    :return:
    """

    modified_data = []
    modified_indices = []
    modified_signs = []

    # Determine unique product indices from
    # shares_dict to identify those that shouldn't
    # be automatically updated/added
    unique_product_indices_from_dict = set()
    for _, regions in shares_dict.items():
        for _, techs in regions.items():
            for _, details in techs.items():
                if "idx" in details:
                    unique_product_indices_from_dict.add(details["idx"])

    for tech_category, regions in shares_dict.items():
        for region, techs in regions.items():
            all_tech_indices = [
                details["idx"] for _, details in techs.items() if "idx" in details
            ]

            # find column indices in lca.technosphere_matrix
            # for which the row indices are in all_tech_indices

            nonzeros = lca.technosphere_matrix.nonzero()
            nonzeros_row, nonzeros_column = nonzeros

            # we want the indices from nonzeros_column if an index
            # from nonzeros_row is in all_tech_indices

            a = np.where(np.isin(nonzeros_row, all_tech_indices))

            indices_array = np.array(
                (nonzeros_row[a], nonzeros_column[a]), dtype=bwp.INDICES_DTYPE
            )







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
                group_shares = correlated_samples(
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
                    lca.dicts.product[idx]
                    for idx in all_tech_indices
                    if lca.dicts.product[idx] is not None
                ]
                total_output = np.sum(lca.technosphere_matrix[product_idx, relevant_indices])

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


def correlated_samples(ranges: dict, defaults: dict, iterations=1000):
    """
    Adjusts randomly selected shares for parameters to sum to 1
    while respecting their specified ranges.

    :param ranges: Dict with parameter names as keys and (min, max) tuples as values.
    :param defaults: Dict with default values for each parameter.
    :param iterations: Number of iterations to attempt to find a valid distribution.
    :return: A dict with the adjusted shares for each parameter.
    """
    for _ in range(iterations):
        shares = {
            param: np.random.uniform(low, high)
            for param, (low, high) in ranges.items()
        }
        total_share = sum(shares.values())
        shares = {param: share / total_share for param, share in shares.items()}
        if all(
                ranges[param][0] <= share <= ranges[param][1]
                for param, share in shares.items()
        ):
            return shares

    print(f"Failed to find a valid distribution after {iterations} iterations")
    return defaults
