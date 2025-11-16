"""
This module contains functions to calculate the Life Cycle Assessment (LCA) results for a given model, scenario, and year.

"""

from __future__ import annotations
import logging
import pickle
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

from .filesystem_constants import DIR_CACHED_DB
from .lcia import fill_characterization_factors_matrices
from .subshares import (
    adjust_matrix_based_on_shares,
    find_technology_indices,
    get_subshares_matrix,
)
from .edges_matrix import create_edges_characterization_matrix
from .utils import (
    CustomFilter,
    _group_technosphere_indices,
    check_unclassified_activities,
    fetch_indices,
    get_unit_conversion_factors,
    read_indices_csv,
)

logger = logging.getLogger(__name__)


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a sparse matrix representation and uncertainties from a CSV export.

    :param file_path: CSV file containing row, column, value, and distribution columns.
    :type file_path: pathlib.Path
    :returns: Tuple of data values, index pairs, sign flags, and distribution metadata.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    # Load the data from the CSV file
    array = np.genfromtxt(file_path, delimiter=";", skip_header=1)

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
                            * vars_idx[variable]["lhv"]["value"]
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

    # Build category × location mapping once
    dict_loc_cat: Dict[tuple, np.ndarray] = {}
    cat_counter = 0
    for _, act_cat_idx in lca.acts_category_idx_dict.items():
        loc_counter = 0
        for _, act_loc_idx in lca.acts_location_idx_dict.items():
            idx = np.intersect1d(act_cat_idx, act_loc_idx)
            filtered_idx = idx[idx != -1]
            if filtered_idx.size > 0:
                dict_loc_cat[(cat_counter, loc_counter)] = filtered_idx
            loc_counter += 1
        cat_counter += 1

    # Helper to build sparse 3D tensor: (n_inv, second_dim, n_cols)
    #   - edges: second_dim = n_methods, using characterization (n_methods, n_bio, n_cols) COO
    #            inventory v (n_bio, n_cols) -> broadcast multiply, then sum over biosphere (axis=1).
    #   - regular: second_dim = n_methods, using C (n_methods, n_bio) CSR @ v (n_bio, n_cols)
    def _inventory_results_3d_edges(lca, char_coo: spnd.COO):
        invs = [mat.tocsr() for mat in lca.inventories.values()]
        rows = []
        for v in invs:
            v_coo = spnd.COO.from_scipy_sparse(v)  # (n_bio, n_cols)
            H = char_coo * v_coo  # (n_methods, n_bio, n_cols)
            S = H.sum(axis=1)  # sum over biosphere -> (n_methods, n_cols)
            rows.append(S)
        return spnd.stack(rows, axis=0)  # (n_inv, n_methods, n_cols)

    def _inventory_results_3d_regular(lca, C: sps.csr_matrix):
        invs = [mat.tocsr() for mat in lca.inventories.values()]
        slices = []
        for v in invs:
            M = C @ v  # (n_methods, n_cols), SciPy sparse
            slices.append(spnd.COO.from_scipy_sparse(M))
        return spnd.stack(slices, axis=0)  # (n_inv, n_methods, n_cols)

    if use_distributions == 0:
        # Regular LCA calculations
        # with CustomFilter("(almost) singular matrix"):
        #    lca.lci()

        if debug:
            logging.info(
                f"Edges methods: {edges_methods}. Monte Carlo iters: {use_distributions}."
            )

        # --- Build sparse 3D inventory_results: (n_inv, n_methods, n_cols)
        if edges_methods:
            # characterization_matrix must be a 3D pydata.sparse COO: (n_methods, n_bio, n_cols)
            if (
                not isinstance(characterization_matrix, spnd.COO)
                or characterization_matrix.ndim != 3
            ):
                raise ValueError(
                    "Edges methods require a 3D pydata.sparse COO characterization tensor (n_methods, n_bio, n_cols)."
                )
            inventory_results = _inventory_results_3d_edges(
                lca, characterization_matrix
            )
        else:
            # characterization_matrix must be a 2D SciPy sparse (n_methods, n_bio)
            if (
                not issparse(characterization_matrix)
                or characterization_matrix.ndim != 2
            ):
                raise ValueError(
                    "Regular methods require a 2D SciPy sparse characterization matrix (n_methods, n_bio)."
                )
            inventory_results = _inventory_results_3d_regular(
                lca, characterization_matrix.tocsr()
            )

        if debug:
            logging.info(f"inventory_results shape: {inventory_results.shape}")

        # Unify downstream aggregation
        # inventory_results: (n_inv, second_dim=n_methods, n_cols)
        n_inv, second_dim, _ = inventory_results.shape
        n_cat = len(lca.acts_category_idx_dict)
        n_loc = len(lca.acts_location_idx_dict)

        zeros_block = spnd.zeros((n_inv, second_dim), dtype=inventory_results.dtype)

        cat_stacks = []
        for cat in range(n_cat):
            loc_blocks = []
            for loc in range(n_loc):
                idx = dict_loc_cat.get((cat, loc))
                if idx is None or idx.size == 0:
                    block = zeros_block  # (n_inv, n_methods)
                else:
                    block = inventory_results[:, :, idx].sum(
                        axis=2
                    )  # (n_inv, n_methods)
                loc_blocks.append(block)
            # Stack blocks across the location axis -> (n_inv, n_methods, n_loc)
            cat_stacks.append(spnd.stack(loc_blocks, axis=2))

        # Stack across categories -> (n_inv, n_methods, n_cat, n_loc)
        iter_results = spnd.stack(cat_stacks, axis=2)

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
                    inventory_results = _inventory_results_3d_edges(
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
                    inventory_results = _inventory_results_3d_regular(
                        lca, characterization_matrix.tocsr()
                    )

                # Aggregate to (n_inv, n_methods, n_cat, n_loc)
                n_inv, second_dim, _ = inventory_results.shape
                n_cat = len(lca.acts_category_idx_dict)
                n_loc = len(lca.acts_location_idx_dict)

                zeros_block = spnd.zeros(
                    (n_inv, second_dim), dtype=inventory_results.dtype
                )

                cat_stacks = []
                for cat in range(n_cat):
                    loc_blocks = []
                    for loc in range(n_loc):
                        idx = dict_loc_cat.get((cat, loc))
                        if idx is None or idx.size == 0:
                            block = zeros_block
                        else:
                            block = inventory_results[:, :, idx].sum(axis=2)
                        loc_blocks.append(block)
                    cat_stacks.append(spnd.stack(loc_blocks, axis=2))

                iter_results = spnd.stack(cat_stacks, axis=2)

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

        # Save MC parameter draws
        iter_param_vals_filepath = DIR_CACHED_DB / f"iter_param_vals_{uuid.uuid4()}.npy"
        np.save(file=iter_param_vals_filepath, arr=np.stack(iter_param_vals, axis=-1))

        # Save indices
        id_uncertainty_indices_filepath = (
            DIR_CACHED_DB / f"mc_indices_{uuid.uuid4()}.npy"
        )
        np.save(file=id_uncertainty_indices_filepath, arr=lca.uncertain_parameters)

        id_technosphere_indices_filepath = (
            DIR_CACHED_DB / f"tech_indices_{uuid.uuid4()}.pkl"
        )
        pickle.dump(
            lca.technosphere_indices, open(id_technosphere_indices_filepath, "wb")
        )

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
        uncertain_parameters,
        remove_uncertainty,
        seed,
        double_accounting,
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

        if debug:
            logging.info(
                f"Functional units created. " f"Total number of activities: {len(fus)}"
            )
            for fu in fus:
                logging.info(
                    f"Functional unit: {fu}, demand: {fus[fu]}. Details: {fus_details[fu]}"
                )
            logging.info(f"variables: {variables}")

        lca = bc.MultiLCA(
            demands=fus,
            method_config={"impact_categories": []},
            data_objs=[
                bw_datapackage,
            ],
            use_distributions=True if use_distributions > 0 else False,
            seed_override=seed,
        )

        with CustomFilter("(almost) singular matrix"):
            lca.lci()

        if shares:
            shares_indices = find_technology_indices(regions, technosphere_indices, geo)
            correlated_arrays = adjust_matrix_based_on_shares(
                lca=lca,
                shares_dict=shares_indices,
                subshares=shares,
                year=year,
            )
            bw_correlated = get_subshares_matrix(correlated_arrays)

            lca = bc.MultiLCA(
                demands=fus,
                method_config={"impact_categories": []},
                data_objs=[bw_datapackage, bw_correlated],
                use_distributions=True if use_distributions > 0 else False,
                use_arrays=True,
            )

            with CustomFilter("(almost) singular matrix"):
                lca.lci()

        lca.uncertain_parameters = uncertain_parameters
        lca.technosphere_indices = technosphere_indices
        lca.acts_category_idx_dict = acts_category_idx_dict
        lca.acts_location_idx_dict = acts_location_idx_dict

        lca.technosphere_indices = {
            k: v
            for k, v in lca.technosphere_indices.items()
            if v in {value for tup in lca.uncertain_parameters for value in tup}
        }

        if methods:
            # regular LCIA methods
            characterization_matrix = fill_characterization_factors_matrices(
                methods=methods,
                biosphere_matrix_dict=lca.dicts.biosphere,
                biosphere_dict=biosphere_indices,
                debug=debug,
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
            characterization_matrix, lca = create_edges_characterization_matrix(
                model=model,
                multilca_obj=lca,
                methods=edges_methods,
                indices={
                    "biosphere": formatted_biosphere_index,
                    "technosphere": formatted_technosphere_index,
                },
            )

        if debug:
            logging.info(
                f"Characterization matrix created. "
                f"Shape: {characterization_matrix.shape}"
            )

        bar.update()
        # Iterate over each region
        results[region] = process_region(
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
                use_distributions,
                uncertain_parameters,
            )
        )

    return results
