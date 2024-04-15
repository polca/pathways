"""
This module contains functions to calculate the Life Cycle Assessment (LCA) results for a given model, scenario, and year.

"""

import csv
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings

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
from scipy.sparse import csr_matrix

from .filesystem_constants import DIR_CACHED_DB
from .lcia import fill_characterization_factors_matrices
from .subshares import (
    adjust_matrix_based_on_shares,
    get_subshares_matrix,
    find_technology_indices,
)
from .utils import (
    _group_technosphere_indices,
    check_unclassified_activities,
    fetch_indices,
    get_unit_conversion_factors,
)

# disable warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.DEBUG,
    filename="pathways.log",  # Log file to save the entries
    filemode="a",  # Append to the log file if it exists, 'w' to overwrite
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_indices_csv(file_path: Path) -> dict[tuple[str, str, str, str], int]:
    """
    Reads a CSV file and returns its contents as a dictionary.

    Each row of the CSV file is expected to contain four string values followed by an index.
    These are stored in the dictionary as a tuple of the four strings mapped to the index.

    :param file_path: The path to the CSV file.
    :type file_path: Path

    :return: A dictionary mapping tuples of four strings to indices.
    For technosphere indices, the four strings are the activity name, product name, location, and unit.
    For biosphere indices, the four strings are the flow name, category, subcategory, and unit.
    :rtype: Dict[Tuple[str, str, str, str], str]
    """
    indices = dict()
    with open(file_path) as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=";")
        for row in csv_reader:
            try:
                indices[(row[0], row[1], row[2], row[3])] = int(row[4])
            except IndexError as err:
                logging.error(
                    f"Error reading row {row} from {file_path}: {err}. "
                    f"Could it be that the file uses commas instead of semicolons?"
                )
    return indices


def load_matrix_and_index(
    file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a CSV file and returns its contents as a CSR sparse matrix.

    :param file_path: The path to the CSV file.
    :type file_path: Path
    :return: A tuple containing the data, indices, and sign of the data.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
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
) -> Tuple[Datapackage, Dict, Dict]:
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
    :rtype: Tuple[sparse.csr_matrix, sparse.csr_matrix, Dict, Dict]
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

    dp = bwp.create_datapackage()

    fp_A = select_filepath("A_matrix", [fp for fp in fps if "index" not in fp.name])
    fp_B = select_filepath("B_matrix", [fp for fp in fps if "index" not in fp.name])

    # Load matrices and add them to the datapackage
    for matrix_name, fp in [("technosphere_matrix", fp_A), ("biosphere_matrix", fp_B)]:
        data, indices, sign, distributions = load_matrix_and_index(fp)
        dp.add_persistent_vector(
            matrix=matrix_name,
            indices_array=indices,
            data_array=data,
            flip_array=sign if matrix_name == "technosphere_matrix" else None,
            distributions_array=distributions,
        )

    return dp, technosphere_inds, biosphere_inds


def remove_double_counting(A: csr_matrix, vars_info: dict) -> csr_matrix:
    """
    Remove double counting from a technosphere matrix.
    :param A: Technosphere matrix
    :param vars_info: Dictionary with information about which indices to zero out.
    :return: Technosphere matrix with double counting removed.
    """

    A_coo = A.tocoo()

    list_of_idx = []

    for region in vars_info:
        for variable in vars_info[region]:
            idx = vars_info[region][variable]["idx"]
            if idx not in list_of_idx:
                list_of_idx.append(idx)
                row_mask = np.isin(A_coo.row, idx)
                col_mask = np.isin(A_coo.col, idx)
                A_coo.data[row_mask & ~col_mask] = 0

    A_coo.eliminate_zeros()
    return A_coo.tocsr()


def process_region(data: Tuple) -> dict[str, ndarray[Any, dtype[Any]] | list[int]]:
    """
    Process the region data.
    :param data: Tuple containing the model, scenario, year, region, variables, vars_idx, scenarios, units_map,
                    demand_cutoff, lca, characterization_matrix, debug, use_distributions.
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
        debug,
        use_distributions,
    ) = data

    variables_demand = {}
    d = []

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

        # If the demand is below the cut-off criteria, skip to the next iteration
        share = demand / scenarios.sel(
            region=region,
            model=model,
            pathway=scenario,
            year=year,
        ).sum(dim="variables")

        # If the total demand is zero, return None
        if share < demand_cutoff:
            continue

        variables_demand[variable] = {
            "id": idx,
            "demand": demand.values * float(unit_vector),
        }

        lca.lci(demand={idx: demand.values * float(unit_vector)})

        if use_distributions == 0:
            # Regular LCA
            characterized_inventory = (
                characterization_matrix @ lca.inventory
            ).toarray()

        else:
            # Use distributions for LCA calculations
            # next(lca) is a generator that yields the inventory matrix
            results = np.array(
                [
                    (characterization_matrix @ lca.inventory).toarray()
                    for _ in zip(range(use_distributions), lca)
                ]
            )

            # calculate quantiles along the first dimension
            characterized_inventory = np.quantile(results, [0.05, 0.5, 0.95], axis=0)

        d.append(characterized_inventory)

        if debug:
            logging.info(
                f"var.: {variable}, name: {dataset[0][:50]}, "
                f"ref.: {dataset[1]}, unit: {dataset[2][:50]}, idx: {idx},"
                f"loc.: {dataset[3]}, demand: {round(float(demand.values * float(unit_vector)), 2)}, "
                f"unit conv.: {unit_vector}, "
                f"impact: {np.round(characterized_inventory.sum(axis=-1) / (demand.values * float(unit_vector)), 3)}. "
            )

    # Save the characterization vectors to disk
    id_array = uuid.uuid4()
    np.save(file=DIR_CACHED_DB / f"{id_array}.npy", arr=np.stack(d))

    # just making sure that the memory is freed. Maybe not needed- check later
    del d

    # returning a dictionary containing the id_array and the variables
    # to be able to fetch them back later
    return {
        "id_array": id_array,
        "variables": {k: v["demand"] for k, v in variables_demand.items()},
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
        subshares,
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
        bw_datapackage, technosphere_indices, biosphere_indices = get_lca_matrices(
            filepaths, model, scenario, year
        )

    except FileNotFoundError:
        # If LCA matrices can't be loaded, skip to the next iteration
        if debug:
            logging.warning(f"Skipping {model}, {scenario}, {year}, as data not found.")
        return

    # Fetch indices
    vars_info = fetch_indices(mapping, regions, variables, technosphere_indices, geo)

    # Remove contribution from activities in other activities
    # A = remove_double_counting(A, vars_info)

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
        lca.lci(factorize=True)

    else:
        logging.info("Calculating LCA results with distributions.")
        lca = MonteCarloLCA(
            demand={0: 1},
            data_objs=[
                bw_datapackage,
            ],
            use_distributions=True,
        )
        lca.lci()

        if subshares:
            logging.info("Calculating LCA results with subshares.")
            shares_indices = find_technology_indices(regions, technosphere_indices, geo)
            correlated_arrays = adjust_matrix_based_on_shares(
                lca, shares_indices, subshares, use_distributions, year
            )
            bw_correlated = get_subshares_matrix(correlated_arrays)

            lca.packages.append(
                get_datapackage(bw_correlated)
            )
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
                debug,
                use_distributions,
            )
        )

    return results
