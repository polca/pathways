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

logger = logging.getLogger(__name__)


def load_subshares() -> dict:
    """Load technology share definitions from the bundled YAML file.

    :returns: Validated share configuration keyed by category and technology.
    :rtype: dict
    :raises ValueError: If the YAML file does not contain a dictionary.
    """
    with open(SUBSHARES) as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, dict):
        raise ValueError("Subshares data should be a dictionary.")

    return check_subshares(check_uncertainty_params(data))


def check_uncertainty_params(data):
    """Normalize uncertainty descriptors and ensure mandatory fields are present.

    :param data: Parsed technology share configuration.
    :type data: dict
    :returns: Updated configuration with numeric uncertainty identifiers.
    :rtype: dict
    :raises ValueError: When required uncertainty fields are missing (logged warnings accompany adjustments).
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
    """Validate share totals per year and prune invalid technology definitions.

    :param data: Share configuration keyed by technology group.
    :type data: dict
    :returns: Adjusted configuration with normalized ``loc`` values summing to 1.
    :rtype: dict
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
    """Resolve technosphere indices for each technology and region combination.

    :param regions: IAM regions to consider.
    :type regions: list[str]
    :param technosphere_indices: Mapping from activity descriptors to matrix indices.
    :type technosphere_indices: dict[tuple[str, str, str, str], int]
    :param geo: Geomap helper for geographic lookups.
    :type geo: premise.geomap.Geomap
    :returns: Nested dictionary ``{category: {region: {technology: {...}}}}`` with indices and share metadata.
    :rtype: dict[str, dict[str, dict[str, dict[str, Any]]]]
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
    """Construct the activity tuple used to query technosphere indices.

    :param tech: Technology descriptor from the YAML file.
    :type tech: dict
    :param region: IAM region name.
    :type region: str
    :returns: Tuple ``(name, reference product, unit, region)``.
    :rtype: tuple[str, str, str, str]
    """
    return tech.get("name"), tech.get("reference product"), tech.get("unit"), region


def get_subshares_matrix(
    correlated_array: list,
) -> bwp.datapackage.Datapackage:
    """Build a bw_processing datapackage containing correlated share arrays.

    :param correlated_array: Sequence of sampled data, indices, and sign flags.
    :type correlated_array: list[numpy.ndarray]
    :returns: Datapackage ready to merge with core technosphere matrices.
    :rtype: bw_processing.Datapackage
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
    """Redistribute technosphere exchanges according to sampled technology shares.

    :param lca: Running ``bw2calc.MultiLCA`` object.
    :type lca: bw2calc.MultiLCA
    :param shares_dict: Nested dictionary of technology indices per region.
    :type shares_dict: dict[str, dict[str, dict]]
    :param subshares: Share samples keyed by technology group and year.
    :type subshares: dict
    :param year: Scenario year used to select share values.
    :type year: int
    :returns: Tuple of sampled amounts, indices, and sign flags for the technosphere matrix.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
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
    """Helper returning ``defaultdict(dict)`` for nested share storage.

    :returns: Fresh ``defaultdict(dict)`` instance.
    :rtype: collections.defaultdict
    """

    return defaultdict(dict)


def load_and_normalize_shares(
    ranges: dict,
    iterations: int,
) -> dict:
    """Sample technology shares within uncertainty bounds and normalize totals.

    :param ranges: Validated share configuration from :func:`load_subshares`.
    :type ranges: dict
    :param iterations: Number of Monte Carlo draws per technology.
    :type iterations: int
    :returns: Nested dictionary of sampled share arrays normalized per year.
    :rtype: dict
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
    """Fill missing years in share samples using linear interpolation.

    :param shares: Sampled share dictionary grouped by technology and year.
    :type shares: dict
    :param years: Years that should exist after interpolation.
    :type years: list[int]
    :returns: ``None`` (``shares`` is modified in place).
    :rtype: None
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
    """Interpolate share samples for a specific year between two bounding years.

    :param shares: Sampled share dictionary grouped by technology and year.
    :type shares: dict
    :param technology_group: Name of the technology group being interpolated.
    :type technology_group: str
    :param lower_year: Earliest available year used for interpolation.
    :type lower_year: int
    :param upper_year: Latest available year used for interpolation.
    :type upper_year: int
    :param target_year: Year to populate with interpolated samples.
    :type target_year: int
    :returns: ``None`` (updates ``shares`` in place).
    :rtype: None
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
    """Generate share samples across requested years, including interpolation.

    :param years: Years that should be present in the returned share dictionary.
    :type years: list[int]
    :param iterations: Number of Monte Carlo iterations to draw.
    :type iterations: int
    :returns: Normalized and interpolated share samples.
    :rtype: dict
    """
    ranges = load_subshares()
    shares = load_and_normalize_shares(
        ranges,
        iterations,
    )
    interpolate_shares(shares, years)
    return shares
