"""
This module contains functions to calculate the Life Cycle Assessment (LCA) results for a given model, scenario, and year.

"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import bw2calc as bc
import bw_processing as bwp
import numpy as np
import pyprind
from bw2calc.utils import get_datapackage
from bw_processing import Datapackage
from numpy import dtype, ndarray
from premise.geomap import Geomap
from scipy import sparse

from .filesystem_constants import DIR_CACHED_DB, STATS_DIR, USER_LOGS_DIR
from .lcia import fill_characterization_factors_matrices
from .stats import (
    create_mapping_sheet,
    log_intensities_to_excel,
    log_results_to_excel,
    log_subshares_to_excel,
    run_GSA_delta,
    run_GSA_OLS,
)
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
)

# from .montecarlo import MonteCarloLCA
# from .sensitivity_analysis import GlobalSensitivityAnalysis


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
    uncertain_parameters = None
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
        # mc_lca,
    ) = data

    variables_demand = {}
    d = []


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
        return d

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

        if np.sum(demand) == 0:
            logging.info(
                f"Total demand for {variable} in {region}, {model}, {scenario}, {year} is zero. "
                f"Skipping."
            )
            continue

        # If the demand is below the cut-off criteria, skip to the next iteration
        share = demand / total_demand

        # If the total demand is zero, return None
        if share < demand_cutoff:
            continue

        variables_demand[variable] = {
            "id": idx,
            "demand": demand,
            "fu": {idx: demand},
            "dataset": dataset,
            "unit vector": unit_vector,
        }

    param_keys = set()


    if use_distributions == 0:
        # Regular LCA calculations
        with CustomFilter("(almost) singular matrix"):
            for key, value in variables_demand.items():
                lca.lci(demand=value["fu"])

                characterized_inventory = (
                    characterization_matrix @ lca.inventory
                ).toarray()
                d.append(characterized_inventory)

                if debug:
                    logging.info(
                        f"var.: {key}, name: {value['dataset'][0][:50]}, "
                        f"ref.: {value['dataset'][1]}, unit: {value['dataset'][2][:50]}, idx: {value['idx']},"
                        f"loc.: {value['dataset'][3]}, demand: {value['demand']}, "
                        f"unit conv.: {value['unit vector']}, "
                        f"impact: {np.round(characterized_inventory.sum(axis=-1) / value['demand'], 3)}. "
                    )

    else:
        # Use distributions for LCA calculations
        # next(lca) is a generator that yields the inventory matrix
        results = np.zeros(
            (use_distributions, len(variables_demand), len(methods))
        )
        params = defaultdict(list)

        params_container = defaultdict(list)
        with CustomFilter("(almost) singular matrix"):
            for iteration in zip(range(use_distributions), lca):
                for key, value in enumerate(variables_demand.values()):
                    lca.lci(demand=value["fu"])

                    matrix_result = (characterization_matrix @ lca.inventory).toarray()
                    results[iteration[0], key] = matrix_result.sum(axis=-1)

                    for i in range(len(uncertain_parameters)):
                        param_key = (
                            f"{uncertain_parameters[i][0]}_to_{uncertain_parameters[i][1]}"
                        )
                        param_keys.add(param_key)
                        value = -lca.technosphere_matrix[
                            uncertain_parameters[i][0], uncertain_parameters[i][1]
                        ]
                        params[param_key].append(value)

                params_container[key].append(params)

        # calculate quantiles along the first dimension
        characterized_inventory = np.quantile(results, [0.05, 0.5, 0.95], axis=0)

        d.append(characterized_inventory)

        if len(params_container) > 0:
            log_intensities_to_excel(
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

    lca = bc.LCA(
        demand={0: 1},
        data_objs=[
            bw_datapackage,
        ],
        use_distributions=True if use_distributions > 0 else False,
    )

    with CustomFilter("(almost) singular matrix"):
        lca.lci(factorize=True)

    if shares:
        logging.info("Calculating LCA results with subshares.")
        shares_indices = find_technology_indices(regions, technosphere_indices, geo)
        correlated_arrays = adjust_matrix_based_on_shares(
            lca=lca,
            shares_dict=shares_indices,
            subshares=shares,
            year=year,
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
            f"Characterization matrix created. "
            f"Shape: {characterization_matrix.shape}"
        )

    total_impacts_by_method = {method: [] for method in methods}
    all_param_keys = set()
    bar = pyprind.ProgBar(len(regions))
    for region in regions:
        bar.update()
        # Iterate over each region
        if use_distributions != 0:
            # mc_lca = MonteCarloLCA([{0: 1}], methods)
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
                    # mc_lca,
                )
            )
            #for method, impacts in results[region]["impact_by_method"].items():
            #    total_impacts_by_method[method].extend(impacts)
            all_param_keys.update(results[region]["param_keys"])
        else:
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
                    # None, # No Monte Carlo LCA object needed
                )
            )

    if use_distributions != 0:
        export_path = STATS_DIR / f"{model}_{scenario}_{year}.xlsx"
        if shares:
            log_subshares_to_excel(
                year=year,
                shares=shares,
                export_path=export_path,
            )
        #log_results_to_excel(
        #    total_impacts_by_method=total_impacts_by_method,
        #    methods=methods,
        #    filepath=export_path,
        #)
        create_mapping_sheet(
            filepaths=filepaths,
            model=model,
            scenario=scenario,
            year=year,
            parameter_keys=all_param_keys,
            export_path=export_path,
        )
        #run_GSA_OLS(
        #    methods=methods,
        #    export_path=export_path,
        #)
        #print(
        #    f"OLS summaries have been saved to the 'OLS' sheets in {export_path.resolve()}"
        #)

        run_GSA_delta(
            methods=methods,
            export_path=export_path,
        )
        print(
            f"Delta analysis has been saved to the 'Delta' sheets in {export_path.resolve()}"
        )

    return results
