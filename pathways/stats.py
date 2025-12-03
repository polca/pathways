from pathlib import Path
from typing import Dict, Set, Tuple
from zipfile import BadZipFile
import logging

import numpy as np
import pandas as pd
from SALib.analyze import delta

from pathways.filesystem_constants import STATS_DIR

logger = logging.getLogger(__name__)


def log_double_accounting(
    filtered_names: Dict[Tuple[str, ...], Set[str]],
    exception_names: Dict[Tuple[str, ...], Set[str]],
    export_path: Path,
):
    """Write filtered and exceptional activity names by category to Excel.

    :param filtered_names: Mapping from category paths to zeroed activity names.
    :type filtered_names: dict[tuple[str, ...], set[str]]
    :param exception_names: Mapping from category paths to exception names.
    :type exception_names: dict[tuple[str, ...], set[str]]
    :param export_path: Destination Excel workbook path.
    :type export_path: pathlib.Path
    :returns: ``None``
    :rtype: None
    """

    data_filtered = {
        "/".join(category): list(names)
        for category, names in filtered_names.items()
        if names
    }
    data_exceptions = {
        "/".join(category): list(names)
        for category, names in exception_names.items()
        if names
    }

    filtered_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in data_filtered.items()])
    )
    exception_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in data_exceptions.items()])
    )

    export_path.parent.mkdir(parents=True, exist_ok=True)

    if export_path.exists():
        try:
            # Load the existing workbook
            with pd.ExcelWriter(
                export_path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
            ) as writer:

                # Remove the existing sheets if they exist
                if "Double accounting - Zeroed" in writer.book.sheetnames:
                    idx = writer.book.sheetnames.index("Double accounting - Zeroed")
                    std = writer.book.worksheets[idx]
                    writer.book.remove(std)
                    writer.book.create_sheet("Double accounting - Zeroed", idx)
                if "Double accounting - Exceptions" in writer.book.sheetnames:
                    idx = writer.book.sheetnames.index("Double accounting - Exceptions")
                    std = writer.book.worksheets[idx]
                    writer.book.remove(std)
                    writer.book.create_sheet("Double accounting - Exceptions", idx)

                # Write DataFrames to the appropriate sheets
                filtered_df.to_excel(
                    writer,
                    sheet_name="Double accounting - Zeroed",
                    index=False,
                )
                exception_df.to_excel(
                    writer,
                    sheet_name="Double accounting - Exceptions",
                    index=False,
                )

        except BadZipFile:
            print(
                f"Warning: '{export_path}' is not a valid Excel file. Creating a new file."
            )
            with pd.ExcelWriter(export_path, engine="openpyxl", mode="w") as writer:
                filtered_df.to_excel(
                    writer, sheet_name="Double accounting - Zeroed", index=False
                )
                exception_df.to_excel(
                    writer, sheet_name="Double accounting - Exceptions", index=False
                )
    else:
        with pd.ExcelWriter(export_path, engine="openpyxl", mode="w") as writer:
            filtered_df.to_excel(
                writer, sheet_name="Double accounting - Zeroed", index=False
            )
            exception_df.to_excel(
                writer, sheet_name="Double accounting - Exceptions", index=False
            )


def log_double_accounting_flows(
    stats: Dict,
    region: str,
    export_path: Path,
):
    """Write detailed zeroed flows to an Excel sheet.

    Creates a detailed log of which flows were zeroed during double accounting
    removal, including source activity, recipient activity, and original value.

    :param stats: Statistics dict returned by remove_double_accounting containing
        'zeroed_flows', 'kept_internal', 'kept_exceptions', 'kept_diagonal'.
    :type stats: dict
    :param region: Region identifier for this calculation.
    :type region: str
    :param export_path: Destination Excel workbook path.
    :type export_path: pathlib.Path
    :returns: ``None``
    :rtype: None
    """
    # Always log the summary, even if no flows were zeroed
    total_zeroed = stats.get("total_zeroed", 0)
    kept_internal = stats.get("kept_internal", 0)
    kept_fus = stats.get("kept_exceptions", 0)
    kept_diagonal = stats.get("kept_diagonal", 0)

    logger.info(
        f"[Double Accounting - {region}] "
        f"Zeroed: {total_zeroed}, "
        f"Kept internal: {kept_internal}, "
        f"Kept FUs: {kept_fus}, "
        f"Kept diagonal: {kept_diagonal}"
    )

    zeroed_flows = stats.get("zeroed_flows", [])

    # Create summary DataFrame (always created)
    summary_data = {
        "Metric": [
            "Total flows zeroed",
            "Flows kept (internal energy system)",
            "Flows kept (to functional units)",
            "Flows kept (diagonal)",
        ],
        "Count": [
            stats.get("total_zeroed", 0),
            stats.get("kept_internal", 0),
            stats.get("kept_exceptions", 0),
            stats.get("kept_diagonal", 0),
        ],
    }
    summary_df = pd.DataFrame(summary_data)

    # Create DataFrame from zeroed flows (may be empty)
    if zeroed_flows:
        flows_df = pd.DataFrame(zeroed_flows)
        flows_df["region"] = region

        # Reorder columns for clarity
        flows_df = flows_df[["region", "from", "to", "value", "from_idx", "to_idx"]]
        flows_df.columns = [
            "Region",
            "Source Activity",
            "Recipient Activity",
            "Original Value",
            "Source Index",
            "Recipient Index",
        ]

        # Aggregate flows by source
        by_source = (
            flows_df.groupby("Source Activity")
            .agg(
                {
                    "Recipient Activity": "count",
                    "Original Value": "sum",
                }
            )
            .reset_index()
        )
        by_source.columns = [
            "Source Activity",
            "Number of Flows Zeroed",
            "Total Value Zeroed",
        ]
        by_source = by_source.sort_values("Number of Flows Zeroed", ascending=False)
    else:
        flows_df = pd.DataFrame(
            columns=[
                "Region",
                "Source Activity",
                "Recipient Activity",
                "Original Value",
                "Source Index",
                "Recipient Index",
            ]
        )
        by_source = pd.DataFrame(
            columns=["Source Activity", "Number of Flows Zeroed", "Total Value Zeroed"]
        )

    export_path.parent.mkdir(parents=True, exist_ok=True)

    sheet_name = f"Zeroed Flows - {region}"

    try:
        # Determine if file exists and mode
        file_exists = export_path.exists()
        mode = "a" if file_exists else "w"

        with pd.ExcelWriter(
            export_path, engine="openpyxl", mode=mode, if_sheet_exists="overlay"
        ) as writer:
            # If sheet exists, remove it first (like log_double_accounting does)
            if sheet_name in writer.book.sheetnames:
                idx = writer.book.sheetnames.index(sheet_name)
                old_sheet = writer.book.worksheets[idx]
                writer.book.remove(old_sheet)
                writer.book.create_sheet(sheet_name, idx)
                print(f"  → Replaced existing sheet '{sheet_name}'")
            else:
                print(f"  → Creating new sheet '{sheet_name}'")

            # Write summary (always first)
            summary_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

            # Calculate next row position
            next_row = len(summary_df) + 3

            # Write aggregated by source if available
            if not by_source.empty:
                by_source.to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=next_row
                )
                next_row = next_row + len(by_source) + 3
            else:
                # Add note if no flows were zeroed
                note_df = pd.DataFrame(
                    {
                        "Note": [
                            "No flows were zeroed - all energy flows were kept (internal or to functional units)."
                        ]
                    }
                )
                note_df.to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=next_row
                )
                next_row += len(note_df) + 2

            # Write detailed flows ONLY if not empty
            if not flows_df.empty:
                flows_df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    startrow=next_row,
                )

        logger.info(
            f"Double accounting flows logged to {export_path} (sheet: {sheet_name})"
        )

    except Exception as e:
        logger.error(f"Error writing double accounting flows to Excel: {e}")
        raise


def log_subshares(
    shares: dict,
    region: str,
) -> pd.DataFrame:
    """Convert sampled subshare values for a region into a tabular report.

    :param shares: Mapping of technology names to sampled share arrays.
    :type shares: dict[str, numpy.ndarray]
    :param region: IAM region name associated with the samples.
    :type region: str
    :returns: DataFrame indexed by iteration with one column per technology.
    :rtype: pandas.DataFrame
    """

    df = pd.DataFrame(shares)
    df["region"] = region
    df["iteration"] = range(1, len(df) + 1)

    return df[["iteration", "region"] + list(shares.keys())]


def log_uncertainty_values(
    region: str,
    uncertainty_indices: np.array,
    uncertainty_values: np.array,
) -> pd.DataFrame:
    """Tabulate Monte Carlo parameter draws for a region.

    :param region: IAM region identifier.
    :type region: str
    :param uncertainty_indices: Structured identifiers for uncertain exchanges.
    :type uncertainty_indices: numpy.ndarray
    :param uncertainty_values: Sampled values per iteration.
    :type uncertainty_values: numpy.ndarray
    :returns: DataFrame keyed by ``iteration`` and ``region`` with parameter columns.
    :rtype: pandas.DataFrame
    """

    # convert 2D numpy array to list of tuples
    uncertainty_indices = uncertainty_indices.tolist()
    uncertainty_indices = [[str(x) for x in index] for index in uncertainty_indices]
    uncertainty_indices = ["::".join(index) for index in uncertainty_indices]

    df = pd.DataFrame(uncertainty_values.T, columns=uncertainty_indices)
    df["region"] = region
    df["iteration"] = range(1, len(df) + 1)

    return df[["iteration", "region"] + uncertainty_indices]


def log_results(
    total_impacts: np.array,
    methods: list,
    region: str,
):
    """Format total LCIA impacts per iteration for export.

    :param total_impacts: Aggregated impact scores per method.
    :type total_impacts: numpy.ndarray
    :param methods: Ordered method names matching the impact array.
    :type methods: list[str]
    :param region: Region associated with the impacts.
    :type region: str
    :returns: DataFrame with ``iteration`` and ``region`` plus impact columns.
    :rtype: pandas.DataFrame
    """

    df = pd.DataFrame(total_impacts.T, columns=methods)
    df["region"] = region
    df["iteration"] = range(1, len(df) + 1)

    return df[["iteration", "region"] + methods]


def create_mapping_sheet(indices: dict) -> pd.DataFrame:
    """Create a readable mapping table for uncertain activity indices.

    :param indices: Mapping from activity tuples to technosphere indices.
    :type indices: dict[tuple[str, str, str, str], int]
    :returns: DataFrame with split columns for name, product, unit, and region.
    :rtype: pandas.DataFrame
    """

    # Converting the dictionary into a pandas DataFrame
    df = pd.DataFrame(indices.items(), columns=["Index", "Value"])

    # Split the 'Index' column into four separate columns
    df[["Name", "Product", "Unit", "Region"]] = pd.DataFrame(
        df["Index"].tolist(), index=df.index
    )

    # Drop the now unnecessary 'Index' column
    df.drop(columns=["Index"], inplace=True)

    return df


def escape_formula(text: str):
    """Prevent Excel from interpreting strings as formulas by prefixing an apostrophe.

    :param text: Cell text to sanitize.
    :type text: str
    :returns: Safe string for Excel export.
    :rtype: str
    """
    return "'" + text if text.startswith(("=", "-", "+")) else text


def run_GSA_delta(
    total_impacts: pd.DataFrame,
    uncertainty_values: pd.DataFrame,
    technology_shares: pd.DataFrame,
) -> pd.DataFrame:
    """Run SALib Delta moment-independent sensitivity analysis.

    :param total_impacts: DataFrame of impact totals with ``iteration`` and ``region`` columns.
    :type total_impacts: pandas.DataFrame
    :param uncertainty_values: Monte Carlo parameter samples keyed by iteration.
    :type uncertainty_values: pandas.DataFrame
    :param technology_shares: Optional sampled technology share DataFrame.
    :type technology_shares: pandas.DataFrame | None
    :returns: Delta sensitivity indices per LCIA method and parameter.
    :rtype: pandas.DataFrame
    """

    # merge uncertainty_values and technology_shares
    # based on "iteration" and "region" columns

    if len(technology_shares) > 0:
        df_parameters = uncertainty_values.merge(
            technology_shares, on=["iteration", "region"]
        )
    else:
        df_parameters = uncertainty_values

    parameters = [
        param for param in df_parameters.columns if param not in ["iteration", "region"]
    ]

    problem = {
        "num_vars": len(parameters),
        "names": parameters,
        "bounds": [
            [df_parameters[param].min(), df_parameters[param].max()]
            for param in parameters
        ],
    }

    methods = [m for m in total_impacts.columns if m not in ["iteration", "region"]]

    results = []

    for method in methods:
        param_values = df_parameters[parameters].values

        # total impacts for the method
        Y = total_impacts[method].values

        delta_results = delta.analyze(problem=problem, X=param_values, Y=Y)

        for i, param in enumerate(parameters):
            results.append(
                [
                    method,
                    param,
                    delta_results["delta"][i],
                    delta_results["delta_conf"][i],
                    delta_results["S1"][i],
                    delta_results["S1_conf"][i],
                ]
            )

    return pd.DataFrame(
        results,
        columns=["LCIA method", "Parameter", "Delta", "Delta Conf", "S1", "S1 Conf"],
    )


def log_mc_parameters_to_excel(
    model: str,
    scenario: str,
    year: int,
    methods: list,
    result: dict,
    uncertainty_parameters: dict,
    uncertainty_values: dict,
    technosphere_indices: dict,
    iteration_results: dict,
    shares: dict = None,
):
    """Export Monte Carlo parameter samples and impacts to an Excel workbook.

    :param model: IAM model name.
    :type model: str
    :param scenario: Scenario identifier.
    :type scenario: str
    :param year: Scenario year analyzed.
    :type year: int
    :param methods: LCIA method names.
    :type methods: list[str]
    :param result: Regional iteration metadata returned from :func:`process_region`.
    :type result: dict[str, dict]
    :param uncertainty_parameters: Mapping of regions to uncertain exchange indices.
    :type uncertainty_parameters: dict[str, numpy.ndarray]
    :param uncertainty_values: Monte Carlo samples for each region.
    :type uncertainty_values: dict[str, numpy.ndarray]
    :param technosphere_indices: Saved technosphere indices per region.
    :type technosphere_indices: dict[str, dict]
    :param iteration_results: Inventory results per region and iteration.
    :type iteration_results: dict[str, numpy.ndarray]
    :param shares: Optional sampled technology shares keyed by region.
    :type shares: dict | None
    :returns: ``None``
    :rtype: None
    """
    export_path = STATS_DIR / f"{model}_{scenario}_{year}.xlsx"

    # create Excel workbook using openpyxl
    with pd.ExcelWriter(export_path, engine="openpyxl") as writer:

        df_sum_impacts = pd.DataFrame()
        df_uncertainty_values = pd.DataFrame()
        df_technology_shares = pd.DataFrame()
        writer.book.create_sheet("Indices mapping")
        writer.book.create_sheet("Monte Carlo values")
        writer.book.create_sheet("Technology shares")
        writer.book.create_sheet("Total impacts")

        for region, data in result.items():

            total_impacts = np.sum(iteration_results[region], axis=(0, 2, 3))

            df_sum_impacts = pd.concat(
                [
                    df_sum_impacts,
                    log_results(
                        total_impacts=total_impacts,
                        methods=methods,
                        region=region,
                    ),
                ]
            )

            uncertainty_indices = uncertainty_parameters[region]
            uncertainty_vals = uncertainty_values[region]

            df_uncertainty_values = pd.concat(
                [
                    df_uncertainty_values,
                    log_uncertainty_values(
                        region=region,
                        uncertainty_indices=uncertainty_indices,
                        uncertainty_values=uncertainty_vals,
                    ),
                ],
            )

            if shares:
                sub_shares = {}
                for k, v in shares.items():
                    for x, y in v.items():
                        if x == year:
                            for z, w in y.items():
                                sub_shares[f"{k} - {z}"] = w

                df_technology_shares = pd.concat(
                    [
                        df_technology_shares,
                        log_subshares(
                            shares=sub_shares,
                            region=region,
                        ),
                    ],
                )

        indices = technosphere_indices[region]

        if indices:
            df_technosphere_indices = create_mapping_sheet(indices=indices)
            df_technosphere_indices.to_excel(
                writer, sheet_name="Indices mapping", index=False
            )

        df_sum_impacts.to_excel(writer, sheet_name="Total impacts", index=False)
        df_uncertainty_values.to_excel(
            writer, sheet_name="Monte Carlo values", index=False
        )
        df_technology_shares.to_excel(
            writer, sheet_name="Technology shares", index=False
        )

        print(f"Monte Carlo parameters added to: {export_path.resolve()}")


def run_gsa(directory: [str, None] = STATS_DIR, method: str = "delta") -> None:
    """Iterate over exported Monte Carlo workbooks and append GSA results.

    :param directory: Directory containing per-scenario Excel files.
    :type directory: str | pathlib.Path
    :param method: Sensitivity analysis method name (currently only ``"delta"`` supported).
    :type method: str
    :raises ValueError: If an unsupported method is requested.
    :returns: ``None``
    :rtype: None
    """
    if method != "delta":
        raise ValueError(f"Method {method} is not supported.")

    # iterate through the Excel files in the directory

    for file in Path(directory).rglob("*.xlsx"):
        # load content of "Monte Carlo values" sheet into a pandas DataFrame
        df_mc_vals = pd.read_excel(file, sheet_name="Monte Carlo values")

        # load content of "Technology shares" sheet into a pandas DataFrame
        # if it exists

        try:
            df_technology_shares = pd.read_excel(
                file,
                sheet_name="Technology shares",
            )
        except:
            df_technology_shares = None

        # load content of "Total impacts" sheet into a pandas DataFrame
        df_sum_impacts = pd.read_excel(file, sheet_name="Total impacts")

        # open Excel workbook
        with pd.ExcelWriter(file, engine="openpyxl", mode="a") as writer:

            df_GSA_results = run_GSA_delta(
                total_impacts=df_sum_impacts,
                uncertainty_values=df_mc_vals,
                technology_shares=df_technology_shares,
            )

            df_GSA_results.to_excel(
                writer, sheet_name=f"GSA {method.capitalize()}", index=False
            )

        print(f"GSA results added to: {file.resolve()}")
