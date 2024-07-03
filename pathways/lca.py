"""
This module contains functions to calculate the Life Cycle Assessment (LCA) results for a given model, scenario, and year.

"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bw2calc as bc
import bw_processing as bwp
import numpy as np
import pyprind
from bw2calc.monte_carlo import MonteCarloLCA
from bw2calc.utils import get_datapackage
from bw_processing import Datapackage
from numpy import dtype, ndarray
from premise.geomap import Geomap
from scipy import sparse

from .filesystem_constants import DATA_DIR, DIR_CACHED_DB, STATS_DIR, USER_LOGS_DIR
from .lcia import fill_characterization_factors_matrices
from .stats import (
    create_mapping_sheet,
    log_double_accounting,
    log_intensities_to_excel,
    log_results_to_excel,
    log_subshares_to_excel,
    run_stats_analysis,
)
from .subshares import (
    adjust_matrix_based_on_shares,
    find_technology_indices,
    get_subshares_matrix,
)
from .utils import (
    _group_technosphere_indices,
    apply_filters,
    check_unclassified_activities,
    fetch_indices,
    get_all_indices,
    get_combined_filters,
    get_unit_conversion_factors,
    read_categories_from_yaml,
    read_indices_csv,
    CustomFilter
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
    remove_infrastructure: bool = False,
    double_counting: dict = None,
    remove_uncertainty: bool = False,
) -> tuple[
    Datapackage,
    dict[tuple[str, str, str, str], int],
    dict[tuple, int],
    list[tuple[int, int]],
    dict | None,
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
    :param remove_infrastructure: Whether to remove infrastructure exchanges from the technosphere matrix.
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
    for matrix_name, fp in [("technosphere_matrix", fp_A), ("biosphere_matrix", fp_B)]:
        data, indices, sign, distributions = load_matrix_and_index(fp)

        if remove_uncertainty is True:
            distributions = np.array(
                [
                    (0, None, None, None, None, None, False)
                    for _ in range(len(distributions))
                ],
                dtype=bwp.UNCERTAINTY_DTYPE,
            )

        if remove_infrastructure is True and matrix_name == "technosphere_matrix":
            print("--------- Removing infrastructure exchanges")
            infrastructure_indices = np.array(
                [
                    j
                    for i, j in technosphere_inds.items()
                    if i[2] == "unit" and "uranium" not in i[0].lower()
                ]
            )
            idx = np.where(
                np.isin(indices["row"], infrastructure_indices)
                & (indices["row"] != indices["col"])
            )
            data[idx] = 0

        if double_counting is not None and matrix_name == "technosphere_matrix":
            print("--------- Correcting for double-counting")
            # remove double counting
            for k, v in double_counting.items():
                for region, values in vars_info.items():
                    if k in values:
                        idx_k = values[k]["idx"]
                        for activity, y in v.items():
                            if activity in values:
                                idx_v = values[activity]["idx"]
                                amount = y[year]
                                # add indices to `indices`
                                indices = np.append(
                                    indices,
                                    np.array([(idx_k, idx_v)], dtype=bwp.INDICES_DTYPE),
                                    axis=0,
                                )
                                data = np.append(data, amount)
                                distributions = np.append(
                                    distributions,
                                    np.array(
                                        (0, None, None, None, None, None, False),
                                        dtype=bwp.UNCERTAINTY_DTYPE,
                                    ),
                                )
                                sign = np.append(sign, np.array([True]))

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
) -> list[tuple[int, int]]:
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
    :param demand: dict with demand values
    :param activities_to_exclude: list of row indices to zero out
    :return: Technosphere matrix with double counting removed
    """

    # Copy and convert the technosphere matrix to COO format for easy manipulation
    # technosphere_matrix_original = technosphere_matrix.technosphere_matrix.copy()
    technosphere_matrix = technosphere_matrix.tocoo()

    # Create a mask for elements to zero out
    mask = np.isin(technosphere_matrix.row, activities_to_zero) & (
        technosphere_matrix.row != technosphere_matrix.col
    )

    # Apply the mask to set the relevant elements to zero
    print(technosphere_matrix.data[mask].sum() * -1)
    technosphere_matrix.data[mask] = 0

    technosphere_matrix.eliminate_zeros()

    return technosphere_matrix.tocsr()


# def remove_double_accounting(
#     lca: bc.LCA,
#     demand: Dict,
#     activities_to_exclude: List[int],
#     exceptions: List[int],
# ):
#     """
#     Remove double counting from a technosphere matrix by zeroing out specified row values
#     in all columns, except for those on the diagonal and those in the exceptions list.
#
#     :param lca: bw2calc.LCA object
#     :param demand: dict with demand values
#     :param activities_to_exclude: list of row indices to zero out
#     :param exceptions: list of column indices that should not be zeroed out
#     :return: Technosphere matrix with double counting removed
#     """
#
#     print("Activities to exclude (rows):", activities_to_exclude)
#     print("Exceptions (columns):", exceptions)
#
#     # Copy and convert the technosphere matrix to COO format for easy manipulation
#     tm_original = lca.technosphere_matrix.copy()
#     tm_modified = tm_original.tocoo()
#
#     # Create a mask for elements to zero out
#     mask = np.isin(tm_modified.row, activities_to_exclude) & (tm_modified.row != tm_modified.col) & ~np.isin(tm_modified.col, exceptions)
#
#     # Apply the mask to set the relevant elements to zero
#     tm_modified.data[mask] = 0
#
#     # Convert the modified matrix back to CSR format and eliminate explicit zeros
#     tm_modified = tm_modified.tocsr()
#     tm_modified.eliminate_zeros()
#
#     # Update the LCA object's technosphere matrix and recalculate the inventory
#     lca.technosphere_matrix = tm_modified
#     lca.lci(demand=demand)
#     return lca

# def check_non_zero_values(matrix: sparse.csr_matrix, activities_to_exclude: List[int]):
#     """
#     Check that all values in the specified rows are zero (except for the diagonal).
#     Print the position (row, column) of any non-zero values that should be zero.
#
#     :param matrix: 2D scipy.sparse.csr_matrix representing the matrix
#     :param activities_to_exclude: List of integers representing the row indices to check
#     """
#     matrix = matrix.tocoo()  # Ensure the matrix is in COO format for easy iteration
#
#     for row in activities_to_exclude:
#         row_data = matrix.getrow(row).tocoo()  # Get the row in COO format for easy iteration
#         for col, value in zip(row_data.col, row_data.data):
#             if value != 0 and row != col:
#                 print(f"WARNING: Non-zero value found at position ({row}, {col}). Value: {value}. It may still correspond to one of "
#                       f"the activities that should be excluded. If not, something went wrong adjusting for double-counting.")
#
#     print("Double-counting check complete.")


def process_region(data: Tuple) -> dict[str, ndarray[Any, dtype[Any]] | list[int]]:
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
        vars_idx,
        scenarios,
        units_map,
        demand_cutoff,
        lca,
        characterization_matrix,
        methods,
        debug,
        use_distributions,
        uncertain_parameters,
        # activities_to_exclude,
    ) = data

    variables_demand = {}
    d = []
    impacts_by_method = {method: [] for method in methods}
    param_keys = set()
    params_container = []

    for v, variable in enumerate(variables):
        idx, dataset = vars_idx[variable]["idx"], vars_idx[variable]["dataset"]
        # Compute the unit conversion vector for the given activities
        dataset_unit = dataset[2]

        # check if we need units conversion
        unit_vector = get_unit_conversion_factors(
            scenarios.attrs["units"][variable],
            dataset_unit,
            units_map,
        )

        # Fetch the demand for the given region, model, pathway, and year
        demand = scenarios.sel(
            variables=variable,
            region=region,
            model=model,
            pathway=scenario,
            year=year,
        )

        total_demand = scenarios.sel(
            region=region,
            model=model,
            pathway=scenario,
            year=year,
        ).sum(dim="variables")

        if total_demand == 0:
            print(
                f"Total demand for {region}, {model}, {scenario}, {year} is zero. Skipping."
            )
            continue

        # If the demand is below the cut-off criteria, skip to the next iteration
        share = demand / total_demand

        # If the total demand is zero, return None
        if share < demand_cutoff:
            continue

        variables_demand[variable] = {
            "id": idx,
            "demand": demand.values * float(unit_vector),
        }

        functional_unit = {idx: demand.values * float(unit_vector)}

        with CustomFilter("(almost) singular matrix"):
            lca.lci(demand=functional_unit)

        # if activities_to_exclude is not None:
        #     check_non_zero_values(lca.technosphere_matrix, activities_to_exclude)

        if use_distributions == 0:
            # Regular LCA
            characterized_inventory = (
                characterization_matrix @ lca.inventory
            ).toarray()

        else:
            # Use distributions for LCA calculations
            # next(lca) is a generator that yields the inventory matrix
            temp_results = []
            params = {}

            for _ in zip(range(use_distributions), lca):
                matrix_result = (characterization_matrix @ lca.inventory).toarray()
                temp_results.append(matrix_result)
                for i in range(len(uncertain_parameters)):
                    param_key = (
                        f"{uncertain_parameters[i][0]}_to_{uncertain_parameters[i][1]}"
                    )
                    param_keys.add(param_key)
                    if param_key not in params:
                        params[param_key] = []
                    value = -lca.technosphere_matrix[
                        uncertain_parameters[i][0], uncertain_parameters[i][1]
                    ]
                    params[param_key].append(value)

            params_container.append(params)

            results = np.array(temp_results)
            for idx, method in enumerate(methods):
                # Sum over items for each stochastic result, not across all results
                method_impacts = results[:, idx, :].sum(axis=1)
                impacts_by_method[method] = method_impacts.tolist()

            # calculate quantiles along the first dimension
            characterized_inventory = np.quantile(results, [0.05, 0.5, 0.95], axis=0)

        d.append(characterized_inventory)

        if debug:
            logging.info(
                f"var.: {variable}, name: {dataset[0][:50]}, "
                f"ref.: {dataset[1]}, unit: {dataset[2][:50]}, idx: {idx},"
                f"loc.: {dataset[3]}, demand: {variables_demand[variable]['demand']}, "
                f"unit conv.: {unit_vector}, "
                f"impact: {np.round(characterized_inventory.sum(axis=-1) / variables_demand[variable]['demand'], 3)}. "
            )

    if len(params_container) > 0:
        log_intensities_to_excel(
            model=model,
            scenario=scenario,
            year=year,
            params=params_container,
            export_path=STATS_DIR / f"{model}_{scenario}_{year}.xlsx",
        )

    # Save the characterization vectors to disk
    id_array = uuid.uuid4()
    np.save(file=DIR_CACHED_DB / f"{id_array}.npy", arr=np.stack(d))

    # just making sure that the memory is freed. Maybe not needed-check later
    del d

    # returning a dictionary containing the id_array and the variables
    # to be able to fetch them back later
    return {
        "id_array": id_array,
        "variables": {k: v["demand"] for k, v in variables_demand.items()},
        "impact_by_method": impacts_by_method,
        "param_keys": param_keys,
    }


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
        debug,
        use_distributions,
        shares,
        uncertain_parameters,
        remove_infrastructure,
        double_counting,
        remove_uncertainty,
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

    # Try to load LCA matrices for the given model, scenario, and year
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
            remove_infrastructure=remove_infrastructure,
            double_counting=double_counting,
            remove_uncertainty=remove_uncertainty,
        )

    except FileNotFoundError:
        # If LCA matrices can't be loaded, skip to the next iteration
        if debug:
            logging.warning(f"Skipping {model}, {scenario}, {year}, as data not found.")
        return

    # if double_accounting is not None:
    #     categories = read_categories_from_yaml(DATA_DIR / "smart_categories.yaml")
    #     combined_filters, exception_filters = get_combined_filters(
    #         categories, double_accounting
    #     )
    #     activities_to_exclude, exceptions, filtered_names, exception_names = (
    #         apply_filters(
    #             technosphere_indices,
    #             combined_filters,
    #             exception_filters,
    #             double_accounting,
    #         )
    #     )
    #     log_double_accounting(
    #         filtered_names=filtered_names,
    #         exception_names=exception_names,
    #         export_path=STATS_DIR / f"{model}_{scenario}_{year}.xlsx",
    #     )
    # else:
    #     activities_to_exclude = None

    # List of indices in vars_info
    # vars_info_idx = get_all_indices(vars_info)

    # check unclassified activities
    missing_classifications = check_unclassified_activities(
        technosphere_indices, classifications
    )

    if missing_classifications:
        if debug:
            logging.warning(
                f"{len(missing_classifications)} activities are not found in the classifications."
                "See missing_classifications.csv for more details."
            )

    results = {}

    locations = lca_results.coords["location"].values.tolist()

    acts_category_idx_dict = _group_technosphere_indices(
        technosphere_indices=technosphere_indices,
        group_by=lambda x: classifications.get(x[:3], "unclassified"),
        group_values=list(set(lca_results.coords["act_category"].values)),
    )

    acts_location_idx_dict = _group_technosphere_indices(
        technosphere_indices=technosphere_indices,
        group_by=lambda x: x[-1],
        group_values=locations,
    )

    results["other"] = {
        "acts_category_idx_dict": acts_category_idx_dict,
        "acts_location_idx_dict": acts_location_idx_dict,
    }

    if use_distributions == 0:
        logging.info("Calculating LCA results without distributions.")

        lca = bc.LCA(
            demand={0: 1},
            data_objs=[
                bw_datapackage,
            ],
        )

        with CustomFilter("(almost) singular matrix"):
            lca.lci(factorize=True)

        # infra_idx = [v for k, v in technosphere_indices.items() if k[2] == "unit"]

        # lca.technosphere_matrix = remove_double_counting(
        #    technosphere_matrix=lca.technosphere_matrix,
        #    activities_to_zero=vars_info_idx,
        #    infra=infra_idx,
        # )

        # if activities_to_exclude is not None:
        #     lca = remove_double_accounting(
        #         lca=lca,
        #         demand={0: 1},
        #         activities_to_exclude=activities_to_exclude,
        #         exceptions=exceptions,
        #     )

    else:
        logging.info("Calculating LCA results with distributions.")
        lca = MonteCarloLCA(
            demand={0: 1},
            data_objs=[
                bw_datapackage,
            ],
            use_distributions=True,
        )

        with CustomFilter("(almost) singular matrix"):
            lca.lci()
        # lca = remove_double_counting(
        #     technosphere_matrix=lca,
        #     demand={0: 1},
        #     activities_to_exclude=vars_info_idx,
        # )
        # if activities_to_exclude is not None:
        #     lca = remove_double_accounting(
        #         lca=lca,
        #         demand={0: 1},
        #         activities_to_exclude=activities_to_exclude,
        #         exceptions=exceptions,
        #     )

        if shares is True:
            logging.info("Calculating LCA results with subshares.")
            shares_indices = find_technology_indices(regions, technosphere_indices, geo)
            correlated_arrays = adjust_matrix_based_on_shares(
                filepaths,
                lca,
                shares_indices,
                shares,
                use_distributions,
                model,
                scenario,
                year,
            )
            bw_correlated = get_subshares_matrix(correlated_arrays)

            lca.packages.append(get_datapackage(bw_correlated))
            lca.use_arrays = True

    characterization_matrix = fill_characterization_factors_matrices(
        methods=methods,
        biosphere_matrix_dict=lca.dicts.biosphere,
        biosphere_dict=biosphere_indices,
        debug=debug,
    )

    if debug:
        logging.info(
            f"Characterization matrix created. Shape: {characterization_matrix.shape}"
        )

    total_impacts_by_method = {method: [] for method in methods}
    all_param_keys = set()
    bar = pyprind.ProgBar(len(regions))
    for region in regions:
        bar.update()
        # Iterate over each region
        results[region] = process_region(
            (
                model,
                scenario,
                year,
                region,
                variables,
                vars_info[region],
                scenarios,
                units,
                demand_cutoff,
                lca,
                characterization_matrix,
                methods,
                debug,
                use_distributions,
                uncertain_parameters,
                # activities_to_exclude,
            )
        )

        if use_distributions != 0:
            for method, impacts in results[region]["impact_by_method"].items():
                total_impacts_by_method[method].extend(impacts)
            all_param_keys.update(results[region]["param_keys"])

    if use_distributions != 0:
        if shares is True:
            log_subshares_to_excel(
                year=year,
                shares=shares,
                export_path=STATS_DIR / f"{model}_{scenario}_{year}.xlsx",
            )
        log_results_to_excel(
            total_impacts_by_method=total_impacts_by_method,
            methods=methods,
            filepath=STATS_DIR / f"{model}_{scenario}_{year}.xlsx",
        )
        create_mapping_sheet(
            filepaths=filepaths,
            model=model,
            scenario=scenario,
            year=year,
            parameter_keys=all_param_keys,
            export_path=STATS_DIR / f"{model}_{scenario}_{year}.xlsx",
        )
        run_stats_analysis(
            model=model,
            scenario=scenario,
            year=year,
            methods=methods,
            export_path=STATS_DIR / f"{model}_{scenario}_{year}.xlsx",
        )
    return results
