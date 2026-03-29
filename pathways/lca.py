"""
This module contains functions to calculate the Life Cycle Assessment (LCA) results for a given model, scenario, and year.

"""

from __future__ import annotations
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any

import bw2calc as bc
import bw_processing as bwp
import numpy as np
import pyprind
import sparse as sp
from bw_processing import Datapackage
from premise.geomap import Geomap
from scipy import sparse
from scipy.sparse import issparse

from scipy import sparse as sps
import numpy as np
import sparse as spnd

from .filesystem_constants import DIR_CACHED_DB, DATA_DIR, STATS_DIR
from .jacobi_gmres_multi_lca import JacobiGMRESMultiLCA
from .lcia import fill_characterization_factors_matrices
from .subshares import (
    adjust_matrix_based_on_shares,
    find_technology_indices,
    get_subshares_matrix,
    year_has_subshare_variation,
)
from .edges_matrix import (
    create_edges_characterization_matrix,
    export_edges_contributors,
)
from .utils import (
    CustomFilter,
    _group_technosphere_indices,
    check_unclassified_activities,
    fetch_indices,
    get_unit_conversion_factors,
    read_indices_csv,
    read_categories_from_yaml,
    get_combined_filters,
    apply_filters,
)
from .stats import log_double_accounting, log_double_accounting_flows

logger = logging.getLogger(__name__)
MATRIX_ARRAY_CACHE_VERSION = 2
MULTILCA_SOLVERS = {"direct", "jacobi-gmres"}


def _matrix_array_cache_dir() -> Path:
    """Return the persistent subdirectory used for parsed matrix caches."""
    cache_dir = Path(DIR_CACHED_DB) / "matrix_arrays"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _matrix_source_identifier(file_path: Path) -> str:
    """Build a stable identifier for a matrix source across datapackage re-extractions."""
    resolved = Path(file_path).expanduser().resolve()

    if "inventories" in resolved.parts:
        start = resolved.parts.index("inventories")
        return "/".join(resolved.parts[start:])

    return str(resolved)


def _matrix_source_digest(file_path: Path) -> str:
    """Hash the source CSV contents for cache validation across temporary extracts."""
    digest = hashlib.blake2b(digest_size=16)
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_array_cache_path(file_path: Path) -> Path:
    """Build a stable cache path for parsed matrix arrays."""
    identifier = _matrix_source_identifier(file_path)
    digest = hashlib.sha1(identifier.encode("utf-8")).hexdigest()
    source = Path(file_path)
    safe_stem = f"{source.parent.name}_{source.stem}"
    return _matrix_array_cache_dir() / f"{safe_stem}_{digest}.npz"


def _load_matrix_array_cache(
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load parsed matrix arrays from cache when metadata still matches the source CSV."""
    source = Path(file_path).expanduser().resolve()
    cache_path = _matrix_array_cache_path(source)

    if not cache_path.exists():
        return None

    stat = source.stat()
    source_identifier = _matrix_source_identifier(source)

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            cache_version = int(cached["cache_version"][0])
            cached_identifier = str(cached["source_identifier"][0])
            source_size = int(cached["source_size"][0])
            cached_digest = str(cached["source_digest"][0])

            if cache_version != MATRIX_ARRAY_CACHE_VERSION:
                return None
            if cached_identifier != source_identifier or source_size != stat.st_size:
                return None

            if cached_digest != _matrix_source_digest(source):
                return None

            return (
                cached["data_array"],
                cached["indices_array"],
                cached["flip_array"],
                cached["distributions_array"],
            )
    except (OSError, ValueError, KeyError, IndexError):
        cache_path.unlink(missing_ok=True)
        return None


def _write_matrix_array_cache(
    file_path: Path,
    data_array: np.ndarray,
    indices_array: np.ndarray,
    flip_array: np.ndarray,
    distributions_array: np.ndarray,
) -> None:
    """Persist parsed matrix arrays for faster warm loads on later runs."""
    source = Path(file_path).expanduser().resolve()
    cache_path = _matrix_array_cache_path(source)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{uuid.uuid4().hex}.tmp.npz")
    stat = source.stat()
    source_identifier = _matrix_source_identifier(source)
    source_digest = _matrix_source_digest(source)

    try:
        np.savez(
            tmp_path,
            cache_version=np.array([MATRIX_ARRAY_CACHE_VERSION], dtype=np.int64),
            source_identifier=np.array([source_identifier]),
            source_size=np.array([stat.st_size], dtype=np.int64),
            source_digest=np.array([source_digest]),
            data_array=data_array,
            indices_array=indices_array,
            flip_array=flip_array,
            distributions_array=distributions_array,
        )
        tmp_path.replace(cache_path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def _effective_distribution_count(
    requested_iterations: int,
    uncertain_parameters: list[tuple] | None,
    shares: dict | None,
    year: int,
) -> int:
    """Reduce Monte Carlo iterations when a year has no active uncertainty sources."""
    if requested_iterations <= 0:
        return 0

    has_matrix_uncertainty = bool(uncertain_parameters)
    has_subshare_uncertainty = year_has_subshare_variation(shares, year)

    if has_matrix_uncertainty or has_subshare_uncertainty:
        return requested_iterations

    return 1


def _get_multilca_class(solver: str):
    """Return the ``MultiLCA`` implementation to use for the requested solver."""
    if solver == "direct":
        return bc.MultiLCA
    if solver == "jacobi-gmres":
        return JacobiGMRESMultiLCA
    raise ValueError(
        f"Unknown solver '{solver}'. Expected one of {sorted(MULTILCA_SOLVERS)}."
    )


def _create_multilca(
    *,
    demands: dict[str, dict[int, float]],
    data_objs: list,
    use_distributions: bool,
    seed: int,
    solver: str,
    iterative_rtol: float,
    iterative_atol: float,
    iterative_restart: int | None,
    iterative_maxiter: int | None,
    iterative_use_guess: bool,
    use_arrays: bool = False,
):
    """Instantiate the requested ``MultiLCA`` backend."""
    lca_cls = _get_multilca_class(solver)

    kwargs = dict(
        demands=demands,
        method_config={"impact_categories": []},
        data_objs=data_objs,
        use_distributions=use_distributions,
        seed_override=seed,
    )
    if use_arrays:
        kwargs["use_arrays"] = True

    if lca_cls is JacobiGMRESMultiLCA:
        kwargs.update(
            rtol=iterative_rtol,
            atol=iterative_atol,
            restart=iterative_restart,
            maxiter=iterative_maxiter,
            use_guess=iterative_use_guess,
        )

    return lca_cls(**kwargs)


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a sparse matrix representation and uncertainties from a CSV export.

    :param file_path: CSV file containing row, column, value, and distribution columns.
    :type file_path: pathlib.Path
    :returns: Tuple of data values, index pairs, sign flags, and distribution metadata.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    cached = _load_matrix_array_cache(file_path)
    if cached is not None:
        return cached

    # Load the data from the CSV file
    array = np.genfromtxt(file_path, delimiter=";", skip_header=1, ndmin=2)
    if array.size == 0 or array.shape[0] == 0:
        raise ValueError(f"Matrix file {file_path} has no data rows.")
    if array.shape[1] < 11:
        raise ValueError(
            f"Matrix file {file_path} must contain at least 11 semicolon-separated columns; "
            f"found {array.shape[1]}."
        )

    # give `indices_array` a list of tuples of indices
    indices_array = np.array(
        list(zip(array[:, 1].astype(int), array[:, 0].astype(int))),
        dtype=bwp.INDICES_DTYPE,
    )

    data_array = array[:, 2]

    # make a boolean scalar array to store the sign of the data
    flip_array = array[:, -1].astype(bool)

    distributions_array = np.array(
        list(
            zip(
                array[:, 3].astype(int),  # uncertainty type
                array[:, 4].astype(float),  # loc
                array[:, 5].astype(float),  # scale
                array[:, 6].astype(float),  # shape
                array[:, 7].astype(float),  # minimum
                array[:, 8].astype(float),  # maximum
                array[:, 9].astype(bool),  # negative
            )
        ),
        dtype=bwp.UNCERTAINTY_DTYPE,
    )

    _write_matrix_array_cache(
        file_path=file_path,
        data_array=data_array,
        indices_array=indices_array,
        flip_array=flip_array,
        distributions_array=distributions_array,
    )

    return data_array, indices_array, flip_array, distributions_array


def get_lca_matrices(
    filepaths: list,
    model: str,
    scenario: str,
    year: int,
    mapping: Dict = None,
    regions: List[str] = None,
    variables: List[str] = None,
    geo: Geomap = None,
    remove_uncertainty: bool = False,
) -> tuple[
    Datapackage,
    dict[tuple[str, str, str, str], int],
    dict[tuple, int],
    list[tuple] | None,
    dict | None,
]:
    """Retrieve the technosphere and biosphere matrices plus indices for a scenario.

    :param filepaths: Candidate CSV file paths bundled in the datapackage.
    :type filepaths: list[str]
    :param model: Name of the IAM model to filter for.
    :type model: str
    :param scenario: Pathway identifier to match in filenames.
    :type scenario: str
    :param year: Scenario year encoded in the filenames.
    :type year: int
    :param mapping: Optional scenario-to-dataset mapping for variable lookup.
    :type mapping: dict | None
    :param regions: IAM regions used to pre-fetch activity indices.
    :type regions: list[str] | None
    :param variables: Scenario variable names to pre-fetch.
    :type variables: list[str] | None
    :param geo: Geomap helper for location matching.
    :type geo: premise.geomap.Geomap | None
    :param remove_uncertainty: When ``True``, zero out distribution parameters.
    :type remove_uncertainty: bool
    :returns: Datapackage with LCI matrices, technosphere/biosphere indices, uncertain parameter tuples, and variable index metadata.
    :rtype: tuple[bw_processing.Datapackage, dict, dict, list[tuple] | None, dict | None]
    :raises FileNotFoundError: If expected matrix files cannot be located.
    :raises ValueError: When the set of candidate files does not match expectations.
    """

    # find the correct filepaths in filepaths
    # the correct filepath are the strings that contains
    # the model, scenario and year
    def filter_filepaths(suffix: str, contains: List[str]):
        return [
            Path(fp)
            for fp in filepaths
            if all(kw in fp.replace(" ", "") for kw in contains)
            and Path(fp).suffix == suffix
            and Path(fp).exists()
        ]

    def select_filepath(keyword: str, fps):
        matches = [fp for fp in fps if keyword in fp.name]
        if not matches:
            raise FileNotFoundError(f"Expected file containing '{keyword}' not found.")
        return matches[0]

    fps = filter_filepaths(
        suffix=".csv",
        contains=[model, str(year)] + scenario.replace(" ", "").split("-"),
    )
    if len(fps) != 4:
        raise ValueError(
            f"Expected 4 filepaths, got {len(fps)} when looking at {filepaths} for terms: {model}, {scenario}, {year}"
        )

    fp_technosphere_inds = select_filepath("A_matrix_index", fps)
    fp_biosphere_inds = select_filepath("B_matrix_index", fps)
    technosphere_inds = read_indices_csv(fp_technosphere_inds)
    biosphere_inds = read_indices_csv(fp_biosphere_inds)
    # remove the last element of the tuple, which is the index
    biosphere_inds = {k[:-1]: v for k, v in biosphere_inds.items()}

    # Fetch indices
    if geo is not None:
        vars_info = fetch_indices(mapping, regions, variables, technosphere_inds, geo)
    else:
        vars_info = None

    dp = bwp.create_datapackage()

    fp_A = select_filepath("A_matrix", [fp for fp in fps if "index" not in fp.name])
    fp_B = select_filepath("B_matrix", [fp for fp in fps if "index" not in fp.name])

    # Load matrices and add them to the datapackage
    uncertain_parameters = None
    for matrix_name, fp in [("technosphere_matrix", fp_A), ("biosphere_matrix", fp_B)]:
        data, indices, sign, distributions = load_matrix_and_index(fp)

        # remove uncertainty data
        if remove_uncertainty is True:
            distributions = np.array(
                [
                    (0, None, None, None, None, None, False)
                    for _ in range(len(distributions))
                ],
                dtype=bwp.UNCERTAINTY_DTYPE,
            )

        if matrix_name == "technosphere_matrix":
            uncertain_parameters = find_uncertain_parameters(distributions, indices)

        dp.add_persistent_vector(
            matrix=matrix_name,
            indices_array=indices,
            data_array=data,
            flip_array=sign if matrix_name == "technosphere_matrix" else None,
            distributions_array=distributions,
        )

    return dp, technosphere_inds, biosphere_inds, uncertain_parameters, vars_info


def find_uncertain_parameters(
    distributions_array: np.ndarray, indices_array: np.ndarray
) -> list[tuple]:
    """Identify technosphere elements that carry non-default uncertainty metadata.

    :param distributions_array: Structured array of uncertainty descriptors.
    :type distributions_array: numpy.ndarray
    :param indices_array: Row/column index pairs aligned with the distributions.
    :type indices_array: numpy.ndarray
    :returns: List of index tuples referring to uncertain technosphere elements.
    :rtype: list[tuple[int, int]]
    """
    uncertain_indices = np.where(distributions_array["uncertainty_type"] != 0)[0]
    uncertain_parameters = [tuple(indices_array[idx]) for idx in uncertain_indices]

    return uncertain_parameters


def remove_double_accounting(
    lca: bc.MultiLCA,
    activities_to_exclude: List[int],
    exceptions: List[int],
    technosphere_indices: Dict[tuple, int] = None,
    debug: bool = False,
) -> tuple[bc.MultiLCA, Dict]:
    """Remove double counting from a technosphere matrix.

    Zeroes out outputs FROM specified activities (e.g., electricity production/markets)
    TO activities OUTSIDE the energy system. Flows WITHIN the energy system
    (between activities in activities_to_exclude) are preserved to maintain the
    complete supply chain for functional units.

    For example, if electricity production and markets are in activities_to_exclude
    and a heat pump activity is in exceptions (as a functional unit):
    - HV production → HV market: KEPT (both are energy activities)
    - HV market → LV transformation: KEPT (both are energy activities)
    - LV market → heat pump (FU): KEPT (FU is in exceptions)
    - LV market → battery production: ZEROED (battery is outside energy system)

    :param lca: bw2calc.MultiLCA object with computed technosphere matrix.
    :type lca: bw2calc.MultiLCA
    :param activities_to_exclude: List of activity indices representing the energy system
        (production, transformation, markets). Outputs from these to non-energy,
        non-exception activities will be zeroed.
    :type activities_to_exclude: list[int]
    :param exceptions: List of activity indices (typically functional units) that should
        continue to receive outputs from energy activities.
    :type exceptions: list[int]
    :param technosphere_indices: Optional mapping from (name, product, unit, location) to index
        for logging purposes.
    :type technosphere_indices: dict[tuple, int] | None
    :param debug: Enable detailed logging when True.
    :type debug: bool
    :returns: Tuple of (modified LCA object, zeroed flows statistics dict).
    :rtype: tuple[bw2calc.MultiLCA, dict]
    """
    tm_original = lca.technosphere_matrix.copy()
    tm_modified = tm_original.tocoo()

    # Convert to sets for O(1) lookup
    activities_to_exclude_set = set(activities_to_exclude)
    exceptions_set = set(exceptions) if exceptions else set()

    # Build reverse index for logging (index -> activity name)
    idx_to_name = {}
    if technosphere_indices:
        idx_to_name = {v: k[0] for k, v in technosphere_indices.items()}

    # Track zeroed flows for logging
    zeroed_flows = []
    kept_internal = 0
    kept_exceptions = 0
    kept_diagonal = 0

    for act in activities_to_exclude:
        act_name = idx_to_name.get(act, f"idx:{act}")
        # Find all OUTPUTS FROM this activity (where this activity is the supplier/row)
        col_idx = np.where(tm_modified.row == act)[0]
        for idx in col_idx:
            recipient = tm_modified.col[idx]
            recipient_name = idx_to_name.get(recipient, f"idx:{recipient}")

            if recipient == act:
                kept_diagonal += 1
            elif recipient in activities_to_exclude_set:
                kept_internal += 1
            elif recipient in exceptions_set:
                kept_exceptions += 1
            else:
                # Zero this flow
                zeroed_flows.append(
                    {
                        "from": act_name,
                        "from_idx": act,
                        "to": recipient_name,
                        "to_idx": recipient,
                        "value": tm_modified.data[idx],
                    }
                )
                tm_modified.data[idx] = 0

    tm_modified = tm_modified.tocsr()
    tm_modified.eliminate_zeros()

    # Build statistics
    stats = {
        "total_zeroed": len(zeroed_flows),
        "kept_internal": kept_internal,
        "kept_exceptions": kept_exceptions,
        "kept_diagonal": kept_diagonal,
        "zeroed_flows": zeroed_flows,
    }

    # Log summary
    logging.info(
        f"Double accounting removal: {len(zeroed_flows)} flows zeroed, "
        f"{kept_internal} kept (internal energy flows), "
        f"{kept_exceptions} kept (to functional units), "
        f"{kept_diagonal} kept (diagonal)."
    )

    if debug and zeroed_flows:
        # Group by source activity for cleaner logging
        from collections import defaultdict

        by_source = defaultdict(list)
        for flow in zeroed_flows:
            by_source[flow["from"]].append(flow["to"])

        logging.info("Zeroed flows by source activity:")
        for source, recipients in sorted(by_source.items()):
            unique_recipients = list(set(recipients))
            logging.info(
                f"  {source} → {len(recipients)} flows to: "
                f"{unique_recipients[:3]}{'...' if len(unique_recipients) > 3 else ''}"
            )

    # Update the technosphere matrix and recalculate inventory
    lca.technosphere_matrix = tm_modified
    lca.lci()

    return lca, stats


def create_functional_units(
    scenarios,
    region,
    model,
    scenario,
    year,
    variables,
    vars_idx,
    units_map,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """
    Create functional units for the given region, model, scenario, and year.
    The functional units are created based on the demand for each variable in the scenarios dataset.
    The demand is converted to the appropriate units using the units_map.
    The functional units are returned as a dictionary where the keys are the dataset indices
    and the values are the demand for that dataset.
    Additionally, a detailed dictionary is returned containing information about each variable,
    including its dataset index, demand, and unit conversion vector.

    :param scenarios: xarray.Dataset containing the scenarios data.
    :param region: The region for which to create the functional units.
    :param model: The model for which to create the functional units.
    :param scenario: The scenario for which to create the functional units.
    :param year: The year for which to create the functional units.
    :param variables: List of variables to include in the functional units.
    :param vars_idx: Dictionary mapping variables to their dataset indices and units.
    :param units_map: Dictionary mapping units to their conversion factors.
    :return: A tuple containing:
    """
    variables_demand = {}

    total_demand = (
        scenarios.sel(
            region=region,
            model=model,
            pathway=scenario,
            year=year,
        )
        .sum(dim="variables")
        .values
    )

    if total_demand == 0:
        logging.info(
            f"Total demand for {region}, {model}, {scenario}, {year} is zero. "
            f"Skipping."
        )

    for v, variable in enumerate(variables):
        if variable in vars_idx:
            idx, dataset = vars_idx[variable]["idx"], vars_idx[variable]["dataset"]
            # Compute the unit conversion vector for the given activities
            dataset_unit = dataset[2]

            try:
                # check if we need unit conversions
                unit_vector = get_unit_conversion_factors(
                    scenarios.attrs["units"][variable],
                    dataset_unit,
                    units_map,
                ).astype(float)

            except KeyError:

                if "lhv" in vars_idx[variable]:
                    alternative_unit = vars_idx[variable]["lhv"].get("unit")
                    conversion_factor = vars_idx[variable]["lhv"].get("value")

                    if alternative_unit and conversion_factor:
                        unit_vector = (
                            get_unit_conversion_factors(
                                scenarios.attrs["units"][variable],
                                vars_idx[variable]["lhv"]["unit"],
                                units_map,
                            ).astype(float)
                            / vars_idx[variable]["lhv"]["value"]  # MJ/(MJ/kg) = kg
                        )
                    else:
                        logging.warning(
                            f"Alternative unit or conversion factor missing not found for {variable}: {alternative_unit}, conversion_factor: {conversion_factor}."
                        )
                        unit_vector = 1.0
                else:
                    logging.warning(
                        f"Unit conversion factors not found for {variable}."
                    )
                    unit_vector = 1.0

            # Fetch the demand for the given
            # region, model, pathway, and year
            demand = (
                scenarios.sel(
                    variables=variable,
                    region=region,
                    model=model,
                    pathway=scenario,
                    year=year,
                ).values
                * unit_vector
            )

            variables_demand[variable] = {
                "id": idx,
                "demand": demand,
                "fu": {idx: demand},
                "dataset": dataset,
                "unit vector": unit_vector,
            }

    return {
        key: value["fu"] for key, value in variables_demand.items()
    }, variables_demand


def _build_sparse_inventory_results_3d(
    lca,
    characterization_matrix,
    edges_methods: bool,
):
    """
    Return a 3D sparse tensor (pydata/sparse COO):

      - edges_methods=True:
          slices[i] = characterization_matrix.multiply(v_i)       # (n_bio, n_cols)
          -> stacked to (n_inv, n_bio, n_cols)

      - edges_methods=False:
          slices[i] = (characterization_matrix @ v_i)             # (n_methods, n_cols)
          -> stacked to (n_inv, n_methods, n_cols)

    In both cases, each slice is built as a SciPy sparse matrix,
    then converted to pydata/sparse COO and stacked along axis=0.
    """
    if edges_methods:
        # elementwise multiply, requires identical shapes
        slices = []
        for v in lca.inventories.values():
            assert issparse(v), "inventory matrices must be SciPy sparse."
            s = characterization_matrix.multiply(v)  # SciPy sparse (n_bio, n_cols)
            slices.append(spnd.COO.from_scipy_sparse(s))
        return spnd.stack(slices, axis=0)
    else:
        # proper matrix multiply
        slices = []
        for v in lca.inventories.values():
            assert issparse(v), "inventory matrices must be SciPy sparse."
            s = characterization_matrix @ v  # SciPy sparse (n_methods, n_cols)
            slices.append(spnd.COO.from_scipy_sparse(s))
        return spnd.stack(slices, axis=0)


def _build_column_aggregation_matrix(
    acts_category_idx_dict: dict,
    acts_location_idx_dict: dict,
    n_cols: int,
) -> sps.csc_matrix:
    """Map technosphere columns to flattened ``category × location`` groups."""
    n_cat = len(acts_category_idx_dict)
    n_loc = len(acts_location_idx_dict)

    col_to_cat = np.full(n_cols, -1, dtype=np.int32)
    for cat_idx, act_cat_idx in enumerate(acts_category_idx_dict.values()):
        indices = np.asarray(act_cat_idx, dtype=np.int64)
        if indices.size == 0:
            continue
        indices = indices[indices >= 0]
        col_to_cat[indices] = cat_idx

    col_to_loc = np.full(n_cols, -1, dtype=np.int32)
    for loc_idx, act_loc_idx in enumerate(acts_location_idx_dict.values()):
        indices = np.asarray(act_loc_idx, dtype=np.int64)
        if indices.size == 0:
            continue
        indices = indices[indices >= 0]
        col_to_loc[indices] = loc_idx

    valid_cols = np.flatnonzero((col_to_cat >= 0) & (col_to_loc >= 0))
    group_ids = col_to_cat[valid_cols] * n_loc + col_to_loc[valid_cols]

    matrix = sps.csc_matrix(
        (
            np.ones(valid_cols.shape[0], dtype=np.float32),
            (valid_cols, group_ids),
        ),
        shape=(n_cols, n_cat * n_loc),
    )
    matrix.sum_duplicates()
    if matrix.nnz:
        matrix.data[:] = 1.0
    return matrix


def process_region(data: Tuple) -> Dict[str, str | List[str] | List[int]]:
    """Run LCI/LCIA calculations for one region and persist intermediate arrays.

    :param data: Tuple containing model, scenario, year, region, variable metadata, scenario data, unit mapping, demand cutoff, MultiLCA instance, characterization matrix, method names, debug flag, number of Monte Carlo iterations, and uncertainty bookkeeping.
    :type data: tuple
    :returns: Dictionary with saved inventory result file paths and demand vectors; includes uncertainty metadata when Monte Carlo is enabled.
    :rtype: dict[str, Any]
    """
    (
        model,
        scenario,
        year,
        region,
        variables,
        fus_details,
        scenarios,
        units_map,
        demand_cutoff,
        lca,
        characterization_matrix,  # edges: COO (n_methods, n_bio, n_cols); regular: CSR (n_methods, n_bio)
        methods,
        edges_methods,
        debug,
        use_distributions,
        uncertain_parameters,
    ) = data

    id_uncertainty_indices_filepath = None
    id_technosphere_indices_filepath = None
    iter_results_files: List[Path] = []
    iter_param_vals_filepath = None

    n_cat = len(lca.acts_category_idx_dict)
    n_loc = len(lca.acts_location_idx_dict)
    aggregation_matrix = _build_column_aggregation_matrix(
        acts_category_idx_dict=lca.acts_category_idx_dict,
        acts_location_idx_dict=lca.acts_location_idx_dict,
        n_cols=lca.technosphere_matrix.shape[1],
    )

    # Build sparse 4D tensor directly: (n_inv, n_methods, n_cat, n_loc)
    def _inventory_results_4d_edges(lca, char_coo: spnd.COO):
        invs = [mat.tocsr() for mat in lca.inventories.values()]
        rows = []
        for v in invs:
            v_coo = spnd.COO.from_scipy_sparse(v)  # (n_bio, n_cols)
            H = char_coo * v_coo  # (n_methods, n_bio, n_cols)
            S = H.sum(axis=1)  # sum over biosphere -> (n_methods, n_cols)
            grouped = S.to_scipy_sparse().tocsr() @ aggregation_matrix
            rows.append(
                spnd.COO.from_scipy_sparse(grouped).reshape(
                    (grouped.shape[0], n_cat, n_loc)
                )
            )
        return spnd.stack(rows, axis=0)  # (n_inv, n_methods, n_cat, n_loc)

    def _inventory_results_4d_regular(lca, C: sps.csr_matrix):
        invs = [mat.tocsr() for mat in lca.inventories.values()]
        slices = []
        for v in invs:
            grouped = (C @ v) @ aggregation_matrix  # (n_methods, n_cat * n_loc)
            slices.append(
                spnd.COO.from_scipy_sparse(grouped).reshape(
                    (grouped.shape[0], n_cat, n_loc)
                )
            )
        return spnd.stack(slices, axis=0)  # (n_inv, n_methods, n_cat, n_loc)

    if use_distributions == 0:
        # Regular LCA calculations
        # with CustomFilter("(almost) singular matrix"):
        #    lca.lci()

        if debug:
            logging.info(
                f"Edges methods: {edges_methods}. Monte Carlo iters: {use_distributions}."
            )

        # --- Build sparse 4D iter_results: (n_inv, n_methods, n_cat, n_loc)
        if edges_methods:
            # characterization_matrix must be a 3D pydata.sparse COO: (n_methods, n_bio, n_cols)
            if (
                not isinstance(characterization_matrix, spnd.COO)
                or characterization_matrix.ndim != 3
            ):
                raise ValueError(
                    "Edges methods require a 3D pydata.sparse COO characterization tensor (n_methods, n_bio, n_cols)."
                )
            iter_results = _inventory_results_4d_edges(lca, characterization_matrix)
        else:
            # characterization_matrix must be a 2D SciPy sparse (n_methods, n_bio)
            if (
                not issparse(characterization_matrix)
                or characterization_matrix.ndim != 2
            ):
                raise ValueError(
                    "Regular methods require a 2D SciPy sparse characterization matrix (n_methods, n_bio)."
                )
            iter_results = _inventory_results_4d_regular(
                lca, characterization_matrix.tocsr()
            )

        if debug:
            logging.info(f"iter_results shape: {iter_results.shape}")

        if debug:
            logging.info(f"iter_results shape: {iter_results.shape}")

        # Save without densifying
        iter_results_filepath = DIR_CACHED_DB / f"iter_results_{uuid.uuid4()}.npz"
        spnd.save_npz(
            filename=iter_results_filepath, matrix=iter_results, compressed=True
        )
        iter_results_files.append(iter_results_filepath)

    else:
        # Monte Carlo: same sparse flow per iteration
        iter_param_vals = []
        mc_bar = None
        if use_distributions > 1:
            mc_bar = pyprind.ProgBar(
                use_distributions,
                stream=1,
                title=f"[{region}] Monte Carlo iterations",
            )
        with CustomFilter("(almost) singular matrix"):
            for _ in range(use_distributions):
                next(lca)
                lca.lci()

                if edges_methods:
                    if (
                        not isinstance(characterization_matrix, spnd.COO)
                        or characterization_matrix.ndim != 3
                    ):
                        raise ValueError(
                            "Edges methods require a 3D pydata.sparse COO characterization tensor (n_methods, n_bio, n_cols)."
                        )
                    iter_results = _inventory_results_4d_edges(
                        lca, characterization_matrix
                    )
                else:
                    if (
                        not issparse(characterization_matrix)
                        or characterization_matrix.ndim != 2
                    ):
                        raise ValueError(
                            "Regular methods require a 2D SciPy sparse characterization matrix (n_methods, n_bio)."
                        )
                    iter_results = _inventory_results_4d_regular(
                        lca, characterization_matrix.tocsr()
                    )

                # Save per-iteration sparse tensor
                iter_results_filepath = (
                    DIR_CACHED_DB / f"iter_results_{uuid.uuid4()}.npz"
                )
                spnd.save_npz(
                    filename=iter_results_filepath, matrix=iter_results, compressed=True
                )
                iter_results_files.append(iter_results_filepath)

                # Keep your MC bookkeeping
                iter_param_vals.append(
                    [
                        -lca.technosphere_matrix[index]
                        for index in lca.uncertain_parameters
                    ]
                )

                if mc_bar is not None:
                    mc_bar.update()

        # Save MC parameter draws
        iter_param_vals_filepath = DIR_CACHED_DB / f"iter_param_vals_{uuid.uuid4()}.npy"
        np.save(file=iter_param_vals_filepath, arr=np.stack(iter_param_vals, axis=-1))

        # Save indices
        id_uncertainty_indices_filepath = (
            DIR_CACHED_DB / f"mc_indices_{uuid.uuid4()}.npy"
        )
        np.save(file=id_uncertainty_indices_filepath, arr=lca.uncertain_parameters)

        id_technosphere_indices_filepath = (
            DIR_CACHED_DB / f"tech_indices_{uuid.uuid4()}.json"
        )
        payload = [
            {"activity": list(key), "index": int(value)}
            for key, value in lca.technosphere_indices.items()
        ]
        with open(id_technosphere_indices_filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    # Return file paths + FU variables
    d = {
        "iterations_results": iter_results_files,
        "variables": {k: v["demand"] for k, v in fus_details.items()},
    }

    if debug:
        logging.info(f"d: {d}")
        logging.info(f"FUs: {list(lca.inventories.keys())}")

    if use_distributions > 0:
        d["uncertainty_params"] = [str(id_uncertainty_indices_filepath)]
        d["technosphere_indices"] = [str(id_technosphere_indices_filepath)]
        d["iterations_param_vals"] = [str(iter_param_vals_filepath)]

    return d


def _calculate_year(args: tuple):
    """Prepare LCI/LCIA inputs and aggregate per-region results for a scenario year.

    :param args: Tuple comprising scenario configuration, filtering options, data mappings, debug flags, uncertainty settings, and multiprocessing seed.
    :type args: tuple
    :returns: Regional result dictionary keyed by IAM region, or ``None`` when inputs are missing.
    :rtype: dict[str, dict] | None
    """
    (
        model,
        scenario,
        year,
        regions,
        variables,
        methods,
        edges_methods,
        demand_cutoff,
        filepaths,
        mapping,
        units,
        lca_results,
        classifications,
        scenarios,
        reverse_classifications,
        geography_mapping,
        debug,
        use_distributions,
        shares,
        subshares_config,
        subshare_groups,
        uncertain_parameters,
        remove_uncertainty,
        seed,
        double_accounting,
        ei_version,
        solver,
        iterative_rtol,
        iterative_atol,
        iterative_restart,
        iterative_maxiter,
        iterative_use_guess,
        aggregate_by,
        collect_edges_contributors,
        edges_contributors_include_unmatched,
    ) = args

    print(f"------ Calculating LCA results for {year}...")
    if debug:
        logging.info(
            f"############################### "
            f"{model}, {scenario}, {year} "
            f"###############################"
        )

    try:
        geo = Geomap(model=model)
    except FileNotFoundError:
        from constructive_geometries import Geomatcher

        geo = Geomatcher()
        geo.model = model
        geo.geo = geo

    # Try to load LCA matrices for
    # the given model, scenario, and year

    try:
        (
            bw_datapackage,
            technosphere_indices,
            biosphere_indices,
            uncertain_parameters,
            vars_info,
        ) = get_lca_matrices(
            filepaths=filepaths,
            model=model,
            scenario=scenario,
            year=year,
            mapping=mapping,
            regions=regions,
            variables=variables,
            geo=geo,
            remove_uncertainty=remove_uncertainty,
        )

    except FileNotFoundError:
        # If LCA matrices can't be loaded, skip to the next iteration
        if debug:
            logging.warning(
                f"Skipping {model}, {scenario}, {year}, " f"as data not found."
            )
        return

    # Collect all functional unit indices from vars_info to protect them
    # from double accounting removal
    fu_indices = set()
    fu_names = set()  # For logging
    if vars_info is not None:
        for region_data in vars_info.values():
            for var_data in region_data.values():
                if "idx" in var_data and var_data["idx"] is not None:
                    fu_indices.add(var_data["idx"])
                    # Also collect the activity name for logging
                    if "dataset" in var_data and var_data["dataset"]:
                        fu_names.add(var_data["dataset"][0])  # name is first element

    if debug and fu_indices:
        logging.info(
            f"Found {len(fu_indices)} functional unit indices to protect from double accounting."
        )

    # Handle double accounting if specified
    if double_accounting is not None:
        categories = read_categories_from_yaml(DATA_DIR / "smart_categories.yaml")
        combined_filters, exception_filters = get_combined_filters(
            categories, double_accounting
        )
        activities_to_exclude, exceptions, filtered_names, exception_names = (
            apply_filters(
                technosphere_indices,
                combined_filters,
                exception_filters,
                double_accounting,
            )
        )

        # Protect functional unit activities: add them to exceptions and
        # remove them from activities_to_exclude
        if fu_indices:
            # Check if any FU activities were accidentally included in activities_to_exclude
            protected_fus = fu_indices & set(activities_to_exclude)

            # Find the names of protected activities for logging
            protected_names = {
                k[0] for k, v in technosphere_indices.items() if v in protected_fus
            }

            if protected_fus:
                logging.info(
                    f"Double accounting: {len(protected_fus)} of {len(fu_indices)} functional units "
                    f"matched filters and were protected: {list(protected_names)[:5]}..."
                )

                # Update filtered_names: remove protected activity names from each category
                for path_key in filtered_names:
                    filtered_names[path_key] -= protected_names

                # Update exception_names: add protected FU activities under a special category
                fu_category = ("Functional Units (Protected)",)
                if fu_category not in exception_names:
                    exception_names[fu_category] = set()
                exception_names[fu_category] |= protected_names
            else:
                logging.info(
                    f"Double accounting: None of the {len(fu_indices)} functional units "
                    f"matched the double accounting filters (no protection needed)."
                )

            # Remove FU indices from activities to exclude
            activities_to_exclude = [
                idx for idx in activities_to_exclude if idx not in fu_indices
            ]
            # Add ALL FU indices to exceptions (even if they didn't match filters,
            # they should still be protected from having their outputs zeroed)
            exceptions = list(set(exceptions) | fu_indices)

        log_double_accounting(
            filtered_names,
            exception_names,
            STATS_DIR / f"double_accounting_{model}_{scenario}_{year}.xlsx",
            debug=debug,
        )
        if debug:
            logging.info(
                f"Double accounting summary: {len(activities_to_exclude)} activities will have "
                f"outputs zeroed, {len(exceptions)} activities in exceptions "
                f"(including all {len(fu_indices)} functional units)."
            )
    else:
        activities_to_exclude = None
        exceptions = None

    # check unclassified activities
    missing_classifications, classifications, reverse_classifications = (
        check_unclassified_activities(
            technosphere_indices, classifications, reverse_classifications
        )
    )

    if missing_classifications:
        for missing in missing_classifications:
            print(f"Missing classification: {missing}")
        if debug:
            logging.warning(
                f"{len(missing_classifications)} activities are not found "
                f"in the classifications."
                "See missing_classifications.csv for more details."
            )

    results = {}

    aggregate_by = set(aggregate_by or [])

    if "act_category" in aggregate_by:
        acts_category_idx_dict = _group_technosphere_indices(
            technosphere_indices=technosphere_indices,
            group_by=lambda _: "aggregated",
            group_values=lca_results.coords["act_category"].values.tolist(),
        )
    else:
        acts_category_idx_dict = _group_technosphere_indices(
            technosphere_indices=technosphere_indices,
            group_by=lambda x: classifications.get(x[:2], "undefined"),
            group_values=lca_results.coords["act_category"].values.tolist(),
        )

    # reorder keys of acts_category_idx_dict based on lca_results.coords["act_category"].values
    acts_category_idx_dict = {
        k: acts_category_idx_dict[k]
        for k in lca_results.coords["act_category"].values.tolist()
    }

    if "location" in aggregate_by:
        acts_location_idx_dict = _group_technosphere_indices(
            technosphere_indices=technosphere_indices,
            group_by=lambda _: "aggregated",
            group_values=lca_results.coords["location"].values.tolist(),
        )
    else:
        acts_location_idx_dict = _group_technosphere_indices(
            technosphere_indices=technosphere_indices,
            group_by=lambda x: x[-1],
            group_values=list(set([x[-1] for x in technosphere_indices.keys()])),
            mapping=geography_mapping,
        )

    # reorder keys of acts_location_idx_dict based on lca_results.coords["location"].values
    acts_location_idx_dict = {
        k: acts_location_idx_dict[k]
        for k in lca_results.coords["location"].values.tolist()
    }

    shares_indices = None
    if shares:
        shares_indices = find_technology_indices(
            regions,
            technosphere_indices,
            geo,
            subshares=subshares_config,
            groups=subshare_groups,
        )

    effective_use_distributions = _effective_distribution_count(
        requested_iterations=use_distributions,
        uncertain_parameters=uncertain_parameters,
        shares=shares,
        year=year,
    )
    if use_distributions > 0 and effective_use_distributions == 1 and debug:
        logging.info(
            "No active uncertainty sources found for %s/%s/%s. "
            "Running a single deterministic iteration instead of %s Monte Carlo draws.",
            model,
            scenario,
            year,
            use_distributions,
        )
    elif use_distributions > 0 and effective_use_distributions == 1:
        print(
            f"------ No active uncertainty sources for {year}; "
            f"running 1 deterministic iteration instead of {use_distributions}."
        )

    bar = pyprind.ProgBar(len(regions))
    for region in regions:
        fus, fus_details = create_functional_units(
            scenarios=scenarios,
            region=region,
            model=model,
            scenario=scenario,
            year=year,
            variables=variables,
            vars_idx=vars_info[region],
            units_map=units,
        )

        if not fus:
            raise ValueError(
                "No functional units could be created for "
                f"region={region}, model={model}, scenario={scenario}, year={year}. "
                "This usually means mapped activities are missing in the technosphere "
                "for the selected region/location, or selected variables have no "
                "mapped non-zero demand."
            )

        if debug:
            logging.info(
                f"Functional units created. " f"Total number of activities: {len(fus)}"
            )
            for fu in fus:
                logging.info(
                    f"Functional unit: {fu}, demand: {fus[fu]}. Details: {fus_details[fu]}"
                )
            logging.info(f"variables: {variables}")

        lca = _create_multilca(
            demands=fus,
            data_objs=[bw_datapackage],
            use_distributions=True if effective_use_distributions > 0 else False,
            seed=seed,
            solver=solver,
            iterative_rtol=iterative_rtol,
            iterative_atol=iterative_atol,
            iterative_restart=iterative_restart,
            iterative_maxiter=iterative_maxiter,
            iterative_use_guess=iterative_use_guess,
        )

        with CustomFilter("(almost) singular matrix"):
            lca.lci()
            # Apply double accounting removal if activities were identified
            if activities_to_exclude is not None:
                print(f"[{region}] Applying double accounting removal...")
                lca, da_stats = remove_double_accounting(
                    lca=lca,
                    activities_to_exclude=activities_to_exclude,
                    exceptions=exceptions,
                    technosphere_indices=technosphere_indices,
                    debug=debug,
                )
                # Log detailed zeroed flows to Excel
                log_double_accounting_flows(
                    stats=da_stats,
                    region=region,
                    export_path=STATS_DIR
                    / f"double_accounting_{model}_{scenario}_{year}.xlsx",
                    debug=debug,
                )
                if debug:
                    logging.info(
                        f"Double accounting removal applied for {region}: "
                        f"{da_stats['total_zeroed']} flows zeroed."
                    )

        if shares:
            correlated_arrays = adjust_matrix_based_on_shares(
                lca=lca,
                shares_dict=shares_indices,
                subshares=shares,
                year=year,
            )
            bw_correlated = get_subshares_matrix(correlated_arrays)

            if bw_correlated is not None:
                lca = _create_multilca(
                    demands=fus,
                    data_objs=[bw_datapackage, bw_correlated],
                    use_distributions=(
                        True if effective_use_distributions > 0 else False
                    ),
                    seed=seed,
                    solver=solver,
                    iterative_rtol=iterative_rtol,
                    iterative_atol=iterative_atol,
                    iterative_restart=iterative_restart,
                    iterative_maxiter=iterative_maxiter,
                    iterative_use_guess=iterative_use_guess,
                    use_arrays=True,
                )

                with CustomFilter("(almost) singular matrix"):
                    lca.lci()

        lca.uncertain_parameters = uncertain_parameters
        lca.technosphere_indices = technosphere_indices
        lca.acts_category_idx_dict = acts_category_idx_dict
        lca.acts_location_idx_dict = acts_location_idx_dict

        uncertain_parameter_indices = {
            value for tup in lca.uncertain_parameters for value in tup
        }
        lca.technosphere_indices = {
            k: v
            for k, v in lca.technosphere_indices.items()
            if v in uncertain_parameter_indices
        }

        edges_contributor_manifest = []

        if methods:
            # regular LCIA methods
            characterization_matrix = fill_characterization_factors_matrices(
                methods=methods,
                biosphere_matrix_dict=lca.dicts.biosphere,
                biosphere_dict=biosphere_indices,
                debug=debug,
                ei_version=ei_version,
            )
        else:
            print("Using EDGES' LCIA methods...")

            # EDGES' LCIA methods
            formatted_biosphere_index = {
                v: {"name": k[0], "categories": k[1:]}
                for k, v in biosphere_indices.items()
            }
            formatted_technosphere_index = {
                v: {
                    "name": k[0],
                    "reference product": k[1],
                    "unit": k[2],
                    "location": k[3],
                }
                for k, v in technosphere_indices.items()
            }
            characterization_matrix, lca, edges_lcas = (
                create_edges_characterization_matrix(
                    model=model,
                    multilca_obj=lca,
                    methods=edges_methods,
                    indices={
                        "biosphere": formatted_biosphere_index,
                        "technosphere": formatted_technosphere_index,
                    },
                )
            )

            if collect_edges_contributors:
                edges_contributor_manifest = export_edges_contributors(
                    multilca_obj=lca,
                    edges_lcas=edges_lcas,
                    model=model,
                    scenario=scenario,
                    year=year,
                    region=region,
                    functional_units=fus_details,
                    include_unmatched=edges_contributors_include_unmatched,
                )

        if debug:
            logging.info(
                f"Characterization matrix created. "
                f"Shape: {characterization_matrix.shape}"
            )

        bar.update()
        # Iterate over each region
        region_result = process_region(
            (
                model,
                scenario,
                year,
                region,
                variables,
                fus_details,
                scenarios,
                units,
                demand_cutoff,
                lca,
                characterization_matrix,
                methods,
                edges_methods,
                debug,
                effective_use_distributions,
                uncertain_parameters,
            )
        )

        if edges_contributor_manifest:
            region_result["edges_contributors"] = edges_contributor_manifest

        results[region] = region_result

    return results
