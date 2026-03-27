import logging
import hashlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import bw2calc
import bw_processing as bwp
import numpy as np
import yaml
from premise.geomap import Geomap
from stats_arrays import *

from pathways.filesystem_constants import DATA_DIR
from pathways.utils import get_activity_indices

SUBSHARES = DATA_DIR / "technologies_shares.yaml"
GROUP_METADATA_KEYS = {"_targets"}

logger = logging.getLogger(__name__)


def _is_group_metadata_key(key: Any) -> bool:
    return str(key) in GROUP_METADATA_KEYS


def _iter_group_technologies(group_data: dict):
    for key, value in group_data.items():
        if _is_group_metadata_key(key):
            continue
        yield key, value


def _normalize_targets(targets: Any) -> list[dict]:
    if targets is None:
        return []

    if isinstance(targets, dict):
        targets = [targets]

    if not isinstance(targets, list):
        raise ValueError("`_targets` should be a list of activity descriptors.")

    normalized = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("Each `_targets` entry should be a dictionary.")

        required_fields = ("name", "reference product", "unit")
        missing = [field for field in required_fields if not target.get(field)]
        if missing:
            raise ValueError(
                "`_targets` entries must define "
                + ", ".join(required_fields)
                + f". Missing: {', '.join(missing)}."
            )

        normalized.append(deepcopy(target))

    return normalized


def load_subshares(
    subshares: dict | str | Path | None = None,
    groups: list[str] | None = None,
    regions: list[str] | None = None,
) -> dict:
    """Load technology-share definitions from the bundled YAML or a custom source.

    ``share`` blocks can be defined either as a legacy year mapping:

    .. code-block:: yaml

        share:
          2020: {loc: 0.8}
          2050: {minimum: 0.5, maximum: 0.9, uncertainty_type: uniform}

    or as a region-aware mapping:

    .. code-block:: yaml

        share:
          default:
            2020: {loc: 0.8}
          EUR:
            2020: {loc: 0.7}
            2050: {minimum: 0.4, maximum: 0.8, uncertainty_type: uniform}

    :param subshares: Custom YAML path or already-parsed configuration. ``None`` uses
        the bundled default file.
    :type subshares: dict | str | pathlib.Path | None
    :param groups: Optional list of technology groups to keep.
    :type groups: list[str] | None
    :param regions: Optional list of regions to resolve, using ``default`` as fallback.
    :type regions: list[str] | None
    :returns: Validated share configuration keyed by category, technology, and region.
    :rtype: dict
    """
    data = _load_subshares_source(subshares)

    if not isinstance(data, dict):
        raise ValueError("Subshares data should be a dictionary.")

    data = _normalize_share_structure(data)
    data = _filter_subshare_groups(data, groups)
    data = check_uncertainty_params(data)
    data = _resolve_regions(data, regions)

    return check_subshares(data)


def _load_subshares_source(subshares: dict | str | Path | None) -> dict:
    if subshares is None:
        with open(SUBSHARES, encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    if isinstance(subshares, (str, Path)):
        with open(subshares, encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    if isinstance(subshares, dict):
        return deepcopy(subshares)

    raise TypeError(
        "`subshares` must be None, a dict, or a path to a YAML configuration file."
    )


def _normalize_share_structure(data: dict) -> dict:
    normalized = {}

    for group, technologies in data.items():
        if not isinstance(technologies, dict):
            raise ValueError(
                f"Subshare group '{group}' should contain a dictionary of technologies."
            )

        normalized[group] = {}
        for technology, params in technologies.items():
            if _is_group_metadata_key(technology):
                if technology == "_targets":
                    normalized[group][technology] = _normalize_targets(params)
                continue

            if not isinstance(params, dict):
                raise ValueError(
                    f"Technology '{technology}' in group '{group}' should be a dictionary."
                )

            tech_data = deepcopy(params)
            share_data = tech_data.get("share", {})

            if not isinstance(share_data, dict):
                raise ValueError(
                    f"'share' for technology '{technology}' in group '{group}' should be a dictionary."
                )

            if _is_year_mapping(share_data):
                tech_data["share"] = {"default": _normalize_year_mapping(share_data)}
            else:
                regional_shares = {}
                for region, year_mapping in share_data.items():
                    if not isinstance(year_mapping, dict):
                        raise ValueError(
                            f"Share definition for region '{region}' in technology '{technology}' "
                            f"of group '{group}' should be a dictionary."
                        )

                    if not _is_year_mapping(year_mapping):
                        raise ValueError(
                            f"Share definition for region '{region}' in technology '{technology}' "
                            f"of group '{group}' should map years to uncertainty parameters."
                        )

                    regional_shares[str(region)] = _normalize_year_mapping(year_mapping)

                tech_data["share"] = regional_shares

            normalized[group][technology] = tech_data

    return normalized


def _is_year_mapping(data: dict) -> bool:
    return bool(data) and all(_is_year_key(key) for key in data)


def _is_year_key(key: Any) -> bool:
    try:
        int(key)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_year_mapping(data: dict) -> dict:
    normalized = {}
    for year, params in data.items():
        if not isinstance(params, dict):
            raise ValueError(
                f"Share definition for year '{year}' should be a dictionary of uncertainty parameters."
            )
        normalized[int(year)] = deepcopy(params)
    return normalized


def _filter_subshare_groups(data: dict, groups: list[str] | None) -> dict:
    if not groups:
        return data

    missing_groups = [group for group in groups if group not in data]
    if missing_groups:
        raise ValueError(
            "Requested subshare groups were not found in the configuration: "
            + ", ".join(sorted(missing_groups))
        )

    return {group: data[group] for group in groups}


def _resolve_regions(data: dict, regions: list[str] | None) -> dict:
    if not regions:
        return data

    resolved = {}

    for group, technologies in data.items():
        resolved[group] = {}
        for technology, params in technologies.items():
            if _is_group_metadata_key(technology):
                resolved[group][technology] = deepcopy(params)
                continue

            share_blocks = params.get("share", {})
            regional_shares = {}

            for region in regions:
                if region in share_blocks:
                    regional_shares[region] = deepcopy(share_blocks[region])
                elif "default" in share_blocks:
                    regional_shares[region] = deepcopy(share_blocks["default"])
                else:
                    logger.warning(
                        "No share definition found for technology '%s' in group '%s' and region '%s'.",
                        technology,
                        group,
                        region,
                    )

            tech_data = {k: deepcopy(v) for k, v in params.items() if k != "share"}
            tech_data["share"] = regional_shares
            resolved[group][technology] = tech_data

    return resolved


def check_uncertainty_params(data: dict) -> dict:
    """Normalize uncertainty descriptors and ensure mandatory fields are present."""

    mandatory_uncertainty_fields = {
        0: {"loc"},
        1: {"loc"},
        2: {"loc", "scale"},
        3: {"loc", "scale"},
        4: {"minimum", "maximum"},
        5: {"loc", "minimum", "maximum"},
    }

    uncertainty = {
        "undefined": UndefinedUncertainty.id,
        "lognormal": LognormalUncertainty.id,
        "normal": NormalUncertainty.id,
        "uniform": UniformUncertainty.id,
        "triangular": TriangularUncertainty.id,
    }

    for group, technologies in data.items():
        for technology_name, technology in technologies.items():
            if _is_group_metadata_key(technology_name):
                continue
            for region, years in technology.get("share", {}).items():
                for year, params in years.items():
                    params["uncertainty_type"] = uncertainty.get(
                        params.get("uncertainty_type", "undefined"),
                        UndefinedUncertainty.id,
                    )

                    if not all(
                        key in params
                        for key in mandatory_uncertainty_fields[
                            params["uncertainty_type"]
                        ]
                    ):
                        logging.warning(
                            "Missing mandatory uncertainty parameters for '%s' in '%s' (%s).",
                            year,
                            group,
                            region,
                        )
    return data


def check_subshares(data: dict) -> dict:
    """Validate share totals per year and prune invalid technology definitions."""

    for category, technologies in data.items():
        technologies_to_remove = []
        totals = defaultdict(float)

        for technology, params in _iter_group_technologies(technologies):
            name = params.get("name")
            if (
                not isinstance(name, str)
                or not name.strip()
                or name.strip().lower() == "null"
            ):
                logging.warning(
                    "Technology '%s' in category '%s' is being removed due to invalid name '%s'.",
                    technology,
                    category,
                    name,
                )
                technologies_to_remove.append(technology)
                continue

            if "share" not in params:
                logging.warning(
                    "Technology '%s' in category '%s' does not have a 'share' key.",
                    technology,
                    category,
                )
                continue

            for region, years in params["share"].items():
                for year, share in years.items():
                    if "loc" in share:
                        totals[(region, year)] += share["loc"]

        for technology in technologies_to_remove:
            del technologies[technology]

        for (region, year), total_value in totals.items():
            if np.isclose(total_value, 0.0):
                raise ValueError(
                    f"Total 'loc' value is zero for category '{category}', region '{region}', year '{year}'."
                )

            if not np.isclose(total_value, 1.0, rtol=1e-3):
                logging.warning(
                    "Total of '%s' values in category '%s' for region '%s' does not add up to 1.00 "
                    "(Total: %s). Adjusting values.",
                    year,
                    category,
                    region,
                    total_value,
                )
                for _, params in _iter_group_technologies(technologies):
                    if (
                        "share" in params
                        and region in params["share"]
                        and year in params["share"][region]
                        and "loc" in params["share"][region][year]
                    ):
                        params["share"][region][year]["loc"] /= total_value

    return data


def find_technology_indices(
    regions: list,
    technosphere_indices: dict,
    geo: Geomap,
    subshares: dict | str | Path | None = None,
    groups: list[str] | None = None,
) -> dict:
    """Resolve technosphere indices for each technology and region combination."""
    technologies_dict = load_subshares(
        subshares=subshares, groups=groups, regions=regions
    )
    indices_dict = {}

    for region in regions:
        for tech_category, techs in technologies_dict.items():
            category_dict = indices_dict.setdefault(
                tech_category,
                {"technologies": {}, "targets": {}},
            )
            regional_indices = category_dict["technologies"].setdefault(region, {})

            for tech, info in _iter_group_technologies(techs):
                if region not in info.get("share", {}):
                    continue

                activity_key = create_activity_key(info, region)
                activity_index = get_activity_indices(
                    [activity_key], technosphere_indices, geo
                )[0]

                if activity_index is None:
                    print(
                        f"Warning: No activity index found for technology '{tech}' in region '{region}'."
                    )
                    continue

                regional_indices[tech] = {"idx": activity_index}

            targets = []
            for target in techs.get("_targets", []):
                target_region = target.get("location", region)
                activity_key = create_activity_key(target, target_region)
                activity_index = get_activity_indices(
                    [activity_key], technosphere_indices, geo
                )[0]

                if activity_index is None:
                    print(
                        "Warning: No activity index found for target "
                        f"'{target.get('name')}' in region '{region}'."
                    )
                    continue

                targets.append({"idx": activity_index, "activity": activity_key})

            category_dict["targets"][region] = targets

    return indices_dict


def create_activity_key(tech: dict, region: str | None) -> tuple:
    """Construct the activity tuple used to query technosphere indices."""
    return tech.get("name"), tech.get("reference product"), tech.get("unit"), region


def get_subshares_matrix(
    correlated_array: list,
) -> bwp.datapackage.Datapackage | None:
    """Build a bw_processing datapackage containing correlated share arrays."""

    dp_correlated = bwp.create_datapackage()
    if isinstance(correlated_array, dict):
        technosphere_array = correlated_array.get("technosphere")
        biosphere_array = correlated_array.get("biosphere")
    else:
        technosphere_array = correlated_array
        biosphere_array = None

    added_arrays = False

    if technosphere_array is not None:
        a_data_samples, a_indices, a_sign = technosphere_array
        if a_indices.size > 0 and a_data_samples.size > 0:
            dp_correlated.add_persistent_array(
                matrix="technosphere_matrix",
                indices_array=a_indices,
                data_array=a_data_samples,
                flip_array=a_sign,
            )
            added_arrays = True

    if biosphere_array is not None:
        b_data_samples, b_indices = biosphere_array
        if b_indices.size > 0 and b_data_samples.size > 0:
            dp_correlated.add_persistent_array(
                matrix="biosphere_matrix",
                indices_array=b_indices,
                data_array=b_data_samples,
            )
            added_arrays = True

    if not added_arrays:
        logger.info("No matrix exchanges matched the configured subshare groups.")
        return None

    return dp_correlated


def _infer_iteration_count(subshares: dict, year: int) -> int | None:
    for group_regions in subshares.values():
        for region_years in group_regions.values():
            if year not in region_years:
                continue
            year_shares = region_years[year]
            if not year_shares:
                continue
            return len(np.atleast_1d(next(iter(year_shares.values()))))
    return None


def _build_correlated_array(
    updates: dict[tuple[int, int], np.ndarray],
    iteration_count: int | None,
    with_flip: bool,
):
    indices = np.array(list(updates), dtype=bwp.INDICES_DTYPE)
    if updates:
        data = np.vstack(
            [
                np.atleast_1d(np.asarray(values, dtype=float))
                for values in updates.values()
            ]
        )
    else:
        data = np.empty((0, iteration_count or 0), dtype=float)

    if with_flip:
        signs = np.ones(indices.shape, dtype=bool)
        return data, indices, signs

    return data, indices


def adjust_matrix_based_on_shares(
    lca: bw2calc.MultiLCA,
    shares_dict: dict,
    subshares: dict,
    year: int,
):
    """Redistribute technosphere exchanges according to sampled technology shares."""

    technosphere_updates: dict[tuple[int, int], np.ndarray] = {}
    biosphere_updates: dict[tuple[int, int], np.ndarray] = {}
    iteration_count = _infer_iteration_count(subshares, year)

    nz_row, nz_col = lca.technosphere_matrix.nonzero()
    bio_nz_row, bio_nz_col = lca.biosphere_matrix.nonzero()

    def set_update(store, row_idx: int, col_idx: int, values, replace: bool = False):
        array = np.atleast_1d(np.asarray(values, dtype=float))
        key = (int(row_idx), int(col_idx))
        if replace or key not in store:
            store[key] = array.copy()
        else:
            store[key] = store[key] + array

    def get_nz_col_indices(index):
        rows = np.where(np.isin(nz_row, index))
        cols = nz_col[rows]
        mask = lca.technosphere_matrix[index, cols].toarray() < 0
        return cols[mask[0]]

    def zero_target_columns(targets: list[dict]):
        if iteration_count is None:
            return

        zero_vector = np.zeros(iteration_count, dtype=float)
        for target in targets:
            target_idx = target["idx"]

            tech_rows = nz_row[nz_col == target_idx]
            for row_idx in tech_rows:
                if int(row_idx) == int(target_idx):
                    continue
                set_update(
                    technosphere_updates,
                    row_idx,
                    target_idx,
                    zero_vector,
                    replace=True,
                )

            bio_rows = bio_nz_row[bio_nz_col == target_idx]
            for row_idx in bio_rows:
                set_update(
                    biosphere_updates,
                    row_idx,
                    target_idx,
                    zero_vector,
                    replace=True,
                )

    for tech_category, group_indices in shares_dict.items():
        for region, technologies in group_indices.get("technologies", {}).items():
            for technology in technologies.values():
                technology["consumer_idx"] = get_nz_col_indices(technology["idx"])

    for tech_category, group_indices in shares_dict.items():
        group_shares = subshares.get(tech_category, {})
        technologies_by_region = group_indices.get("technologies", {})
        targets_by_region = group_indices.get("targets", {})
        group_has_targets = any(targets_by_region.values())

        for region, technologies in technologies_by_region.items():
            regional_year_shares = group_shares.get(region, {})
            if year not in regional_year_shares:
                continue

            year_shares = regional_year_shares[year]
            if iteration_count is None and year_shares:
                iteration_count = len(np.atleast_1d(next(iter(year_shares.values()))))

            if group_has_targets:
                targets = targets_by_region.get(region, [])
                if not targets or iteration_count is None:
                    continue

                zero_target_columns(targets)

                for target in targets:
                    target_idx = target["idx"]
                    target_output_amount = float(
                        abs(lca.technosphere_matrix[target_idx, target_idx])
                    )
                    if np.isclose(target_output_amount, 0.0):
                        target_output_amount = 1.0

                    for name, tech in technologies.items():
                        if name not in year_shares:
                            continue

                        set_update(
                            technosphere_updates,
                            tech["idx"],
                            target_idx,
                            target_output_amount * year_shares[name],
                            replace=True,
                        )
                continue

            for name, tech in technologies.items():
                if name not in year_shares:
                    continue

                for consumer in tech["consumer_idx"]:
                    initial_amount = lca.technosphere_matrix[tech["idx"], consumer]
                    split_amount = abs(initial_amount) * year_shares[name]
                    set_update(
                        technosphere_updates,
                        tech["idx"],
                        consumer,
                        split_amount,
                    )

                    for other_supplier, other_supplier_tech in technologies.items():
                        if other_supplier == name or other_supplier not in year_shares:
                            continue

                        additional_amount = (
                            abs(initial_amount) * year_shares[other_supplier]
                        )
                        set_update(
                            technosphere_updates,
                            other_supplier_tech["idx"],
                            consumer,
                            additional_amount,
                        )

    for (tech_idx, consumer_idx), total_amount in technosphere_updates.items():
        logging.info(
            "Final combined amount for tech index %s to consumer index %s: %s",
            tech_idx,
            consumer_idx,
            total_amount,
        )

    return {
        "technosphere": _build_correlated_array(
            technosphere_updates,
            iteration_count,
            with_flip=True,
        ),
        "biosphere": _build_correlated_array(
            biosphere_updates,
            iteration_count,
            with_flip=False,
        ),
    }


def _project_to_bounded_simplex(
    values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    target: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> np.ndarray:
    if np.any(lower_bounds > upper_bounds):
        raise ValueError("Lower share bounds cannot exceed upper share bounds.")

    if lower_bounds.sum() > target + tol or upper_bounds.sum() < target - tol:
        raise ValueError(
            "Share bounds do not admit a feasible solution summing to the requested total."
        )

    lower_tau = np.min(values - upper_bounds)
    upper_tau = np.max(values - lower_bounds)

    for _ in range(max_iter):
        tau = 0.5 * (lower_tau + upper_tau)
        projected = np.clip(values - tau, lower_bounds, upper_bounds)
        total = projected.sum()

        if abs(total - target) <= tol:
            return projected

        if total > target:
            lower_tau = tau
        else:
            upper_tau = tau

    projected = np.clip(
        values - 0.5 * (lower_tau + upper_tau), lower_bounds, upper_bounds
    )
    residual = target - projected.sum()

    if abs(residual) > 1e-8:
        slack = upper_bounds - projected if residual > 0 else projected - lower_bounds
        candidates = slack > tol
        if not np.any(candidates):
            raise ValueError(
                "Unable to normalize sampled shares within the configured bounds."
            )

        weights = slack[candidates]
        weights = weights / weights.sum()
        projected[candidates] += residual * weights
        projected = np.clip(projected, lower_bounds, upper_bounds)

    if not np.isclose(projected.sum(), target, atol=1e-8):
        raise ValueError("Normalized shares do not sum to the requested total.")

    return projected


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "::".join([str(seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.md5(payload).digest()[:4], "big", signed=False)


def load_and_normalize_shares(
    ranges: dict,
    iterations: int,
    seed: int = 0,
) -> dict:
    """Sample technology shares within uncertainty bounds and normalize totals."""
    if iterations <= 0:
        raise ValueError(
            "Subshare sampling requires `iterations` to be greater than zero."
        )

    shares = {}
    for technology_group, technologies in ranges.items():
        for technology, params in _iter_group_technologies(technologies):
            for region, years in params["share"].items():
                for year, share in years.items():
                    uncertainty_base = UncertaintyBase.from_dicts(share)
                    random_generator = MCRandomNumberGenerator(
                        params=uncertainty_base,
                        seed=_stable_seed(
                            seed, technology_group, region, technology, year
                        ),
                    )

                    samples = np.atleast_1d(
                        np.squeeze(
                            np.array(
                                [random_generator.next() for _ in range(iterations)],
                                dtype=float,
                            )
                        )
                    )

                    shares.setdefault(technology_group, {}).setdefault(
                        region, {}
                    ).setdefault(year, {})[technology] = samples

    for technology_group, regions in shares.items():
        for region, years in regions.items():
            for year, technologies in years.items():
                tech_names = list(technologies)
                samples_matrix = np.vstack(
                    [
                        np.atleast_1d(np.asarray(technologies[technology], dtype=float))
                        for technology in tech_names
                    ]
                )
                lower_bounds = np.array(
                    [
                        ranges[technology_group][technology]["share"][region][year].get(
                            "minimum", 0.0
                        )
                        for technology in tech_names
                    ],
                    dtype=float,
                )
                upper_bounds = np.array(
                    [
                        ranges[technology_group][technology]["share"][region][year].get(
                            "maximum", 1.0
                        )
                        for technology in tech_names
                    ],
                    dtype=float,
                )

                normalized = np.zeros_like(samples_matrix, dtype=float)
                for sample_idx in range(samples_matrix.shape[1]):
                    column = samples_matrix[:, sample_idx]
                    total = column.sum()
                    if np.isclose(total, 0.0):
                        column = np.full(column.shape, 1.0 / len(column))
                    else:
                        column = column / total

                    normalized[:, sample_idx] = _project_to_bounded_simplex(
                        column,
                        lower_bounds,
                        upper_bounds,
                    )

                for idx, technology in enumerate(tech_names):
                    technologies[technology] = normalized[idx]

    return shares


def interpolate_shares(shares: dict, years: list) -> None:
    """Fill missing years in share samples using linear interpolation."""
    for technology_group, regions in shares.items():
        for region, yearly_shares in regions.items():
            all_years = sorted(yearly_shares)
            for year in sorted(int(y) for y in years):
                if year in yearly_shares:
                    continue

                lower_year = max((y for y in all_years if y <= year), default=None)
                upper_year = min((y for y in all_years if y >= year), default=None)

                if lower_year is not None and upper_year is not None:
                    interpolate_for_year(
                        shares,
                        technology_group,
                        region,
                        lower_year,
                        upper_year,
                        year,
                    )


def interpolate_for_year(
    shares: dict,
    technology_group: str,
    region: str,
    lower_year: int,
    upper_year: int,
    target_year: int,
) -> None:
    """Interpolate share samples for a specific year between two bounding years."""
    if lower_year == upper_year:
        shares[technology_group][region][target_year] = deepcopy(
            shares[technology_group][region][lower_year]
        )
        return

    span = upper_year - lower_year
    weight = (target_year - lower_year) / span
    interpolated = {}

    for technology in shares[technology_group][region][lower_year]:
        shares_lower = np.atleast_1d(
            np.asarray(
                shares[technology_group][region][lower_year][technology], dtype=float
            )
        )
        shares_upper = np.atleast_1d(
            np.asarray(
                shares[technology_group][region][upper_year][technology], dtype=float
            )
        )
        interpolated[technology] = shares_lower + weight * (shares_upper - shares_lower)

    shares[technology_group][region][target_year] = interpolated


def _renormalize_interpolated_shares(shares: dict) -> None:
    for technology_group, regions in shares.items():
        for region, years in regions.items():
            for year, technologies in years.items():
                tech_names = list(technologies)
                samples_matrix = np.vstack(
                    [
                        np.atleast_1d(np.asarray(technologies[technology], dtype=float))
                        for technology in tech_names
                    ]
                )
                totals = samples_matrix.sum(axis=0)
                zero_mask = np.isclose(totals, 0.0)

                if np.any(zero_mask):
                    samples_matrix[:, zero_mask] = 1.0 / len(tech_names)
                    totals = samples_matrix.sum(axis=0)

                normalized = samples_matrix / totals

                for idx, technology in enumerate(tech_names):
                    technologies[technology] = normalized[idx]


def year_has_subshare_variation(shares: dict | None, year: int) -> bool:
    """Return ``True`` when sampled subshares vary across iterations for a year."""
    if not shares:
        return False

    for group_regions in shares.values():
        for region_years in group_regions.values():
            if year not in region_years:
                continue

            for sampled_values in region_years[year].values():
                sampled_array = np.atleast_1d(np.asarray(sampled_values, dtype=float))
                if sampled_array.size > 1 and not np.allclose(
                    sampled_array, sampled_array[0]
                ):
                    return True

    return False


def generate_samples(
    years: list,
    iterations: int = 10,
    regions: list[str] | None = None,
    seed: int = 0,
    subshares: dict | str | Path | None = None,
    groups: list[str] | None = None,
) -> dict:
    """Generate share samples across requested years, including interpolation."""
    requested_regions = list(regions) if regions else ["default"]
    ranges = load_subshares(
        subshares=subshares,
        groups=groups,
        regions=requested_regions,
    )
    shares = load_and_normalize_shares(
        ranges,
        iterations,
        seed=seed,
    )
    interpolate_shares(shares, years)
    _renormalize_interpolated_shares(shares)

    if regions is None:
        return {
            technology_group: region_shares["default"]
            for technology_group, region_shares in shares.items()
            if "default" in region_shares
        }

    return shares
