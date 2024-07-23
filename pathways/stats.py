import os
import re
from pathlib import Path
from typing import Dict, Set, Tuple
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl import Workbook, load_workbook
from SALib.analyze import delta


def log_double_accounting(
    filtered_names: Dict[Tuple[str, ...], Set[str]],
    exception_names: Dict[Tuple[str, ...], Set[str]],
    export_path: Path,
):
    """
    Log the unique names of the filtered activities and exceptions to an Excel file,
    distinguished by categories.

    :param filtered_names: Dictionary of category paths to sets of filtered activity names.
    :param exception_names: Dictionary of category paths to sets of exception names.
    :param export_path: Path to the export Excel file.
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


def log_subshares_to_excel(
    year: int, shares: dict, total_impacts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Logs results to an Excel file named according to model, scenario, and year, specifically for the given year.
    This method assumes that each entry in the shares defaultdict is structured to be directly usable in a DataFrame.

    Parameters:
    :param year: The specific year for which data is being logged.
    :param shares: A nested defaultdict containing shares data for multiple years and types.
    :param total_impacts_df: DataFrame containing total impacts for each method.
    """

    data = []

    first_tech = next(iter(shares), None)
    if not first_tech or year not in shares[first_tech]:
        print(f"No data found for year {year} in any technology.")
        return

    num_iterations = len(shares[first_tech][year][next(iter(shares[first_tech][year]))])
    for i in range(num_iterations):
        iteration_data = {"Iteration": i + 1, "Year": year}
        for tech, years_data in shares.items():
            if year in years_data:
                for subtype, values in years_data[year].items():
                    iteration_data[f"{tech}_{subtype}"] = (
                        values[i] if i < len(values) else None
                    )
        data.append(iteration_data)

    new_df = pd.DataFrame(data)

    # Merge new data into existing data, selectively updating share columns
    combined_df = (
        total_impacts_df.set_index(["Iteration", "Year"])
        .combine_first(new_df.set_index(["Iteration", "Year"]))
        .reset_index()
    )
    # Optionally, ensure the columns are in a meaningful order
    new_columns = [col for col in new_df.columns if col not in ["Iteration", "Year"]]
    existing_columns = [
        col for col in total_impacts_df.columns if col not in new_df.columns
    ]
    combined_df = combined_df[["Iteration", "Year"] + new_columns + existing_columns]

    return combined_df


def log_intensities_to_excel(year: int, params: list, export_path: Path):
    """
    Update or create an Excel file with new columns of data, based on model, scenario, and year.

    :param year: The year for which the data is logged.
    :param params: Dictionary where keys are the new column names and values are lists of data for each column.
    :param export_path: The path to the Excel file where the data will be logged.
    """

    if not params:
        print("Warning: No new data provided to log.")
        return

    try:
        # merge list of dictionaries into a single dictionary
        params = {k: v for d in params for k, v in d.items()}

        max_length = max(len(v) for v in params.values())

        df_new = pd.DataFrame(params)
        df_new["Iteration"] = range(1, max_length + 1)
        df_new["Year"] = [year] * max_length

        if os.path.exists(export_path):
            df_existing = pd.read_excel(export_path)

            combined_df = pd.merge(
                df_existing,
                df_new,
                on=["Iteration", "Year"],
                how="outer",
                suffixes=("", "_new"),
            )

            for col in df_new.columns:
                if col + "_new" in combined_df:
                    combined_df[col] = combined_df[col].combine_first(
                        combined_df.pop(col + "_new")
                    )

            combined_df = combined_df.loc[:, ~combined_df.columns.str.endswith("_new")]
            df = combined_df
        else:
            df = df_new
        df.to_excel(export_path, index=False)
    except Exception as e:
        print(f"Failed to update the Excel file: {e}")


def log_results_to_excel(
    total_impacts_by_method: dict,
    methods: list,
):
    """
    Log the characterized inventory results for each LCIA method into separate columns in an Excel file.

    :param total_impacts_by_method: Dictionary where keys are method names and values are lists of impacts
    from all regions and distributions.
    :param methods: List of method names.
    :param filepath: Optional. File path for the Excel file to save the results.
    """

    df = pd.DataFrame()

    for method, impacts in total_impacts_by_method.items():
        df[method] = pd.Series(impacts)

    base_cols = ["Iteration", "Year"] if "Iteration" in df.columns else []
    other_cols = [col for col in df.columns if col not in base_cols + methods]
    df = df[base_cols + methods + other_cols]

    return df


def create_mapping_sheet(
    filepaths: list,
    model: str,
    scenario: str,
    year: int,
    parameter_keys: set,
) -> pd.DataFrame:
    """
    Create a mapping sheet for the activities with uncertainties.
    :param filepaths: List of paths to data files.
    :param model: Model name as a string.
    :param scenario: Scenario name as a string.
    :param year: Year as an integer.
    :param parameter_keys: List of parameter keys used in intensity iterations.
    :param export_path: Path to the directory where the Excel file will be saved.
    """

    def filter_filepaths(suffix: str, contains: list):
        return [
            Path(fp)
            for fp in filepaths
            if all(kw in fp for kw in contains)
            and Path(fp).suffix == suffix
            and Path(fp).exists()
        ]

    unique_indices = {int(idx) for key in parameter_keys for idx in key.split("_to_")}

    fps = filter_filepaths(".csv", [model, scenario, str(year)])
    if len(fps) < 1:
        raise ValueError(f"No relevant files found for {model}, {scenario}, {year}")

    technosphere_indices_path = None
    for fp in fps:
        if "A_matrix_index" in fp.name:
            technosphere_indices_path = fp
            break

    if not technosphere_indices_path:
        raise FileNotFoundError("Technosphere indices file not found.")

    technosphere_inds = pd.read_csv(technosphere_indices_path, sep=";", header=None)
    technosphere_inds.columns = ["Activity", "Product", "Unit", "Location", "Index"]

    mapping_df = technosphere_inds[technosphere_inds["Index"].isin(unique_indices)]
    mapping_df = mapping_df[
        ["Activity", "Product", "Location", "Unit", "Index"]
    ]  # Restrict columns if necessary

    return mapping_df


def escape_formula(text: str):
    """
    Prevent a string from being interpreted as a formula in Excel.
    Strings starting with '=', '-', or '+' are prefixed with an apostrophe.

    :param text: The string to be adjusted.
    :return: The adjusted string.
    """
    return "'" + text if text.startswith(("=", "-", "+")) else text


def run_GSA_OLS(methods: list, export_path: Path):
    """
    Runs OLS regression for specified methods and writes summaries to an Excel file.

    :param methods: Methods corresponding to dataset columns.
    :param export_path: Path to the directory where the Excel file will be saved.
    """
    try:
        book = load_workbook(export_path)
    except FileNotFoundError:
        book = Workbook()
        book.save(export_path)
        book = load_workbook(export_path)

    data = pd.read_excel(export_path, sheet_name="Sheet1")

    if "OLS" in book.sheetnames:
        ws = book["OLS"]
        book.remove(ws)

    ws = book.create_sheet("OLS")

    X_base = data.drop(columns=["Iteration", "Year"] + methods)
    X_base = sm.add_constant(X_base)
    corr_matrix = X_base.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_correlation = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > 0.95)
    ]

    if high_correlation:
        print(f"OLS: High multicollinearity detected in columns: {high_correlation}")
        X_base = X_base.drop(columns=high_correlation)

    results = []
    for method in methods:
        if method not in data.columns:
            print(f"Data for {method} not found in the file.")
            continue

        Y = data[method]
        X = X_base.copy()

        try:
            model_results = sm.OLS(Y, X).fit()
            summary = model_results.summary().as_text()
            summary_lines = summary.split("\n")

            results.append([f"OLS Summary for {method}"])
            for line in summary_lines:
                line = escape_formula(line)
                columns = re.split(r"\s{2,}", line)
                results.append(columns)
            results.append([])
        except Exception as e:
            print(f"Error running OLS for method {method}: {e}")

    for result in results:
        ws.append(result)

    book.save(export_path)


def run_GSA_delta(methods: list, df_tot_impacts: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Delta Moment-Independent Measure analysis for specified methods and writes summaries to an Excel file.

    :param methods: List of method names corresponding to dataset columns.
    :param export_path: Path to the directory where the Excel file will be saved.
    """

    standard_columns = {"Iteration", "Year"}
    params = [
        col
        for col in df_tot_impacts.columns
        if col not in standard_columns and col not in methods
    ]

    problem = {
        "num_vars": len(params),
        "names": params,
        "bounds": [
            [df_tot_impacts[param].min(), df_tot_impacts[param].max()]
            for param in params
        ],
    }

    results = []
    for method in methods:
        if method not in df_tot_impacts.columns:
            print(f"Data for {method} not found in the file.")
            continue

        param_values = df_tot_impacts[params].values
        Y = df_tot_impacts[method].values

        delta_results = delta.analyze(problem, param_values, Y, print_to_console=False)

        results.append([f"Delta Moment-Independent Measure for {method}"])
        results.append(["Parameter", "Delta", "Delta Conf", "S1", "S1 Conf"])
        for i, param in enumerate(params):
            results.append(
                [
                    param,
                    delta_results["delta"][i],
                    delta_results["delta_conf"][i],
                    delta_results["S1"][i],
                    delta_results["S1_conf"][i],
                ]
            )
        results.append([])

    return pd.DataFrame(results)
