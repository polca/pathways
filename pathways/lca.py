"""
This module contains functions to calculate the Life Cycle Assessment (LCA) results for a given model, scenario, and year.

"""

import logging
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import bw2calc as bc
import bw_processing as bwp
import numpy as np
import pyprind
import sparse as sp
from bw_processing import Datapackage
from premise.geomap import Geomap
from scipy import sparse

from .filesystem_constants import DIR_CACHED_DB, USER_LOGS_DIR
from .lcia import fill_characterization_factors_matrices
from .subshares import (
    adjust_matrix_based_on_shares,
    find_technology_indices,
    get_subshares_matrix,
)
from .utils import (
    CustomFilter,
    _group_technosphere_indices,
    check_unclassified_activities,
    fetch_indices,
    get_unit_conversion_factors,
    read_indices_csv,
    apply_filters,
    get_combined_filters,
    read_categories_from_yaml,
)

logging.basicConfig(
    level=logging.DEBUG,
    filename=USER_LOGS_DIR / "pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a CSV file and returns its contents as a CSR sparse matrix.

    :param file_path: The path to the CSV file.
    :type file_path: Path
    :return: A tuple containing the data, indices, and sign of the data as well as the exchanges with distributions.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, list]
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
    list[tuple[int, int]],
    [dict, None],
]:
    """
    Retrieve Life Cycle Assessment (LCA) matrices from disk.

    :param filepaths: A list of filepaths containing the LCA matrices.
    :type filepaths: List[str]
    :param model: The name of the model.
    :type model: str
    :param scenario: The name of the scenario.
    :type scenario: str
    :param year: The year of the scenario.
    :type year: int
    :rtype: Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict, Dict, List]
    """

    # find the correct filepaths in filepaths
    # the correct filepath are the strings that contains
    # the model, scenario and year
    def filter_filepaths(suffix: str, contains: List[str]):
        return [
            Path(fp)
            for fp in filepaths
            if all(kw in fp for kw in contains)
            and Path(fp).suffix == suffix
            and Path(fp).exists()
        ]

    def select_filepath(keyword: str, fps):
        matches = [fp for fp in fps if keyword in fp.name]
        if not matches:
            raise FileNotFoundError(f"Expected file containing '{keyword}' not found.")
        return matches[0]

    fps = filter_filepaths(".csv", [model, scenario, str(year)])
    if len(fps) != 4:
        raise ValueError(f"Expected 4 filepaths, got {len(fps)}")

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
    """
    Find the uncertain parameters in the distributions array.
    They will be used for the stats report
    :param distributions_array:
    :param indices_array:
    :return:
    """
    uncertain_indices = np.where(distributions_array["uncertainty_type"] != 0)[0]
    uncertain_parameters = [tuple(indices_array[idx]) for idx in uncertain_indices]

    return uncertain_parameters


def remove_double_counting(
    technosphere_matrix: np.array, activities_to_zero: List[int], infra: List[int]
):
    """
    Remove double counting from a technosphere matrix by zeroing out the demanded row values
    in all columns, except for those on the diagonal.
    :param technosphere_matrix: bw2calc.LCA object
    :return: Technosphere matrix with double counting removed
    """

    # Copy and convert the technosphere matrix
    # to COO format for easy manipulation
    technosphere_matrix = technosphere_matrix.tocoo()

    # Create a mask for elements to zero out
    mask = np.isin(technosphere_matrix.row, activities_to_zero) & (
        technosphere_matrix.row != technosphere_matrix.col
    )

    # Apply the mask to set the relevant elements to zero
    technosphere_matrix.data[mask] = 0
    technosphere_matrix.eliminate_zeros()

    return technosphere_matrix.tocsr()


def create_functional_units(
    scenarios,
    region,
    model,
    scenario,
    year,
    variables,
    vars_idx,
    units_map,
) -> [dict, dict]:
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
        idx, dataset = vars_idx[variable]["idx"], vars_idx[variable]["dataset"]
        # Compute the unit conversion vector for the given activities
        dataset_unit = dataset[2]

        # check if we need units conversion
        unit_vector = get_unit_conversion_factors(
            scenarios.attrs["units"][variable],
            dataset_unit,
            units_map,
        ).astype(float)

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


def process_region(data: Tuple) -> Dict[str, str | List[str] | List[int]]:
    """
    Process the region data.
    :param data: Tuple containing the model, scenario, year, region, variables, vars_idx, scenarios, units_map,
                    demand_cutoff, lca, characterization_matrix, debug, use_distributions, uncertain_parameters.
    :return: Dictionary containing the region data.
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
        characterization_matrix,
        methods,
        debug,
        use_distributions,
        uncertain_parameters,
    ) = data

    id_uncertainty_indices_filepath = None
    id_technosphere_indices_filepath = None
    iter_results_files = []
    iter_param_vals_filepath = None

    dict_loc_cat = {}

    cat_counter = 0
    for cat, act_cat_idx in lca.acts_category_idx_dict.items():
        loc_counter = 0
        for loc, act_loc_idx in lca.acts_location_idx_dict.items():
            # Find the intersection of indices
            idx = np.intersect1d(act_cat_idx, act_loc_idx)
            # Filter out any -1 indices
            filtered_idx = idx[idx != -1]

            if filtered_idx.size > 0:
                # Assign the filtered index array
                # to the dict_loc_cat with (cat, loc) as key
                dict_loc_cat[(cat_counter, loc_counter)] = filtered_idx

            loc_counter += 1
        cat_counter += 1

    if use_distributions == 0:
        # Regular LCA calculations
        with CustomFilter("(almost) singular matrix"):
            lca.lci()

        if debug:
            logging.info(f"Iterations no.: {use_distributions}.")

        # Create a numpy array with the results
        inventory_results = np.array(
            [
                (characterization_matrix @ value).toarray()
                for value in lca.inventories.values()
            ]
        )

        if debug:
            logging.info(f"Shape of inventory_results: {inventory_results.shape}")

        if debug:
            for fu, inventory in lca.inventories.items():
                logging.info(
                    f"Functional unit: {fu}. Impact: {(characterization_matrix @ inventory).sum()}"
                )

        iter_results = np.zeros(
            (
                inventory_results.shape[0],
                inventory_results.shape[1],
                len(lca.acts_category_idx_dict),
                len(lca.acts_location_idx_dict),
            )
        )

        if debug:
            logging.info(f"Shape of iter_results: {iter_results.shape}")

        for (cat, loc), idx in dict_loc_cat.items():
            iter_results[:, :, cat, loc] = inventory_results[:, :, idx].sum(axis=2)

        if debug:
            for f, fu in enumerate(lca.inventories.keys()):
                logging.info(f"Functional unit: {fu}. Impact: {iter_results[f].sum()}")

        # Save iteration results to disk
        iter_results_filepath = DIR_CACHED_DB / f"iter_results_{uuid.uuid4()}.npz"
        sp.save_npz(
            filename=iter_results_filepath,
            matrix=sp.COO(iter_results),
            compressed=True,
        )
        iter_results_files.append(iter_results_filepath)

    else:
        # Use distributions for LCA calculations
        iter_param_vals = []
        with CustomFilter("(almost) singular matrix"):
            for iteration in range(use_distributions):
                next(lca)
                lca.lci()

                # Create a numpy array with the results
                inventory_results = np.array(
                    [
                        (characterization_matrix @ value).toarray()
                        for value in lca.inventories.values()
                    ]
                )
                iter_param_vals.append(
                    [
                        -lca.technosphere_matrix[index]
                        for index in lca.uncertain_parameters
                    ]
                )

                iter_results = np.zeros(
                    (
                        inventory_results.shape[0],
                        inventory_results.shape[1],
                        len(lca.acts_category_idx_dict),
                        len(lca.acts_location_idx_dict),
                    )
                )

                for (cat, loc), idx in dict_loc_cat.items():
                    iter_results[:, :, cat, loc] = inventory_results[:, :, idx].sum(
                        axis=2
                    )

                # Save iteration results to disk
                iter_results_filepath = (
                    DIR_CACHED_DB / f"iter_results_{uuid.uuid4()}.npz"
                )
                sp.save_npz(
                    filename=iter_results_filepath,
                    matrix=sp.COO(iter_results),
                    compressed=True,
                )
                iter_results_files.append(iter_results_filepath)

        # Save iteration parameter values to disk
        iter_param_vals_filepath = DIR_CACHED_DB / f"iter_param_vals_{uuid.uuid4()}.npy"
        np.save(file=iter_param_vals_filepath, arr=np.stack(iter_param_vals, axis=-1))

        # Save the uncertainty indices to disk
        id_uncertainty_indices_filepath = (
            DIR_CACHED_DB / f"mc_indices_{uuid.uuid4()}.npy"
        )
        np.save(
            file=id_uncertainty_indices_filepath,
            arr=lca.uncertain_parameters,
        )

        # Save the technosphere indices to disk
        id_technosphere_indices_filepath = (
            DIR_CACHED_DB / f"tech_indices_{uuid.uuid4()}.pkl"
        )
        pickle.dump(
            lca.technosphere_indices,
            open(id_technosphere_indices_filepath, "wb"),
        )

    # Returning a dictionary containing the id_array and the variables
    # to be able to fetch them back later
    d = {
        "iterations_results": iter_results_files,
        "variables": {k: v["demand"] for k, v in fus_details.items()},
    }

    if debug:
        logging.info(f"d: {d}")
        logging.info(f"FUs: {list(lca.inventories.keys())}")

    if use_distributions > 0:
        d["uncertainty_params"] = [
            str(id_uncertainty_indices_filepath),
        ]
        d["technosphere_indices"] = [
            str(id_technosphere_indices_filepath),
        ]
        d["iterations_param_vals"] = [
            str(iter_param_vals_filepath),
        ]

    return d


def _calculate_year(args: tuple):
    """
    Prepares the data for the calculation of LCA results for a given year
    and calls the process_region function to calculate the results for each region.
    """
    (
        model,
        scenario,
        year,
        regions,
        variables,
        methods,
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
    missing_classifications = check_unclassified_activities(
        technosphere_indices, classifications
    )

    if missing_classifications:
        if debug:
            logging.warning(
                f"{len(missing_classifications)} activities are not found "
                f"in the classifications."
                "See missing_classifications.csv for more details."
            )

    results = {}

    acts_category_idx_dict = _group_technosphere_indices(
        technosphere_indices=technosphere_indices,
        group_by=lambda x: classifications.get(x[:3], "unclassified"),
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
            region=regions[0],
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

        characterization_matrix = fill_characterization_factors_matrices(
            methods=methods,
            biosphere_matrix_dict=lca.dicts.biosphere,
            biosphere_dict=biosphere_indices,
            debug=debug,
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
                debug,
                use_distributions,
                uncertain_parameters,
            )
        )

    return results
