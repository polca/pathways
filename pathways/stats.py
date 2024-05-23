import os
import re
from pathlib import Path
from typing import Dict, Set, Tuple
from zipfile import BadZipFile

import pandas as pd
import statsmodels.api as sm
from openpyxl import load_workbook


def log_double_accounting(
    model: str,
    scenario: str,
    year: int,
    filtered_names: Dict[Tuple[str, ...], Set[str]],
    exception_names: Dict[Tuple[str, ...], Set[str]],
):
    """
    Log the unique names of the filtered activities and exceptions to an Excel file,
    distinguished by categories.

    :param model: The model name.
    :param scenario: The scenario name.
    :param year: The year.
    :param filtered_names: Dictionary of category paths to sets of filtered activity names.
    :param exception_names: Dictionary of category paths to sets of exception names.
    """
    filename = f"stats_report_{model}_{scenario}_{year}.xlsx"

    # Prepare data for DataFrame
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

    # Convert dictionaries to DataFrames
    filtered_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in data_filtered.items()])
    )
    exception_df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in data_exceptions.items()])
    )

    if os.path.exists(filename):
        try:
            # Load the existing workbook
            book = load_workbook(filename)
            with pd.ExcelWriter(filename, engine="openpyxl", mode="a") as writer:
                writer.book = book

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
                    writer, sheet_name="Double accounting - Zeroed", index=False
                )
                exception_df.to_excel(
                    writer, sheet_name="Double accounting - Exceptions", index=False
                )

                writer.save()
        except BadZipFile:
            print(
                f"Warning: '{filename}' is not a valid Excel file. Creating a new file."
            )
            with pd.ExcelWriter(filename) as writer:
                filtered_df.to_excel(
                    writer, sheet_name="Double accounting - Zeroed", index=False
                )
                exception_df.to_excel(
                    writer, sheet_name="Double accounting - Exceptions", index=False
                )
    else:
        with pd.ExcelWriter(filename) as writer:
            filtered_df.to_excel(
                writer, sheet_name="Double accounting - Zeroed", index=False
            )
            exception_df.to_excel(
                writer, sheet_name="Double accounting - Exceptions", index=False
            )


def log_subshares_to_excel(model: str, scenario: str, year: int, shares: dict):
    """
    Logs results to an Excel file named according to model, scenario, and year, specifically for the given year.
    This method assumes that each entry in the shares defaultdict is structured to be directly usable in a DataFrame.

    Parameters:
    :param model: The model name.
    :param scenario: The scenario name.
    :param year: The specific year for which data is being logged.
    :param shares: A nested defaultdict containing shares data for multiple years and types.
    """
    filename = f"stats_report_{model}_{scenario}_{year}.xlsx"
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

    try:
        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)
            # Merge new data into existing data, selectively updating share columns
            combined_df = (
                df_existing.set_index(["Iteration", "Year"])
                .combine_first(new_df.set_index(["Iteration", "Year"]))
                .reset_index()
            )
            # Optionally, ensure the columns are in a meaningful order
            new_columns = [
                col for col in new_df.columns if col not in ["Iteration", "Year"]
            ]
            existing_columns = [
                col for col in df_existing.columns if col not in new_df.columns
            ]
            combined_df = combined_df[
                ["Iteration", "Year"] + new_columns + existing_columns
            ]

            combined_df.to_excel(filename, index=False)
        else:
            new_df.to_excel(filename, index=False)
    except Exception as e:
        print(f"Error updating Excel file: {str(e)}")


def log_intensities_to_excel(model: str, scenario: str, year: int, new_data: dict):
    """
    Update or create an Excel file with new columns of data, based on model, scenario, and year.

    :param model: The model name.
    :param scenario: The scenario name.
    :param year: The year for which the data is logged.
    :param new_data: Dictionary where keys are the new column names and values are lists of data for each column.
    """
    filename = f"stats_report_{model}_{scenario}_{year}.xlsx"

    if not new_data:
        print("Warning: No new data provided to log.")
        return

    try:
        max_length = max(len(v) for v in new_data.values())

        df_new = pd.DataFrame(new_data)
        df_new["Iteration"] = range(1, max_length + 1)
        df_new["Year"] = [year] * max_length

        if os.path.exists(filename):
            df_existing = pd.read_excel(filename)

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

        df.to_excel(filename, index=False)
    except Exception as e:
        print(f"Failed to update the Excel file: {e}")


def log_results_to_excel(
    model: str,
    scenario: str,
    year: int,
    total_impacts_by_method: dict,
    methods: list,
    filepath=None,
):
    """
    Log the characterized inventory results for each LCIA method into separate columns in an Excel file.

    :param model: The model name.
    :param scenario: The scenario name.
    :param year: The year for which the data is being logged.
    :param total_impacts_by_method: Dictionary where keys are method names and values are lists of impacts
    from all regions and distributions.
    :param methods: List of method names.
    :param filepath: Optional. File path for the Excel file to save the results.
    """

    if filepath is None:
        filepath = f"stats_report_{model}_{scenario}_{year}.xlsx"

    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        df = pd.DataFrame()

    for method, impacts in total_impacts_by_method.items():
        df[method] = pd.Series(impacts)

    base_cols = ["Iteration", "Year"] if "Iteration" in df.columns else []
    other_cols = [col for col in df.columns if col not in base_cols + methods]
    df = df[base_cols + methods + other_cols]

    df.to_excel(filepath, index=False)


def create_mapping_sheet(
    filepaths: list, model: str, scenario: str, year: int, parameter_keys: list
):
    """
    Create a mapping sheet for the activities with uncertainties.
    :param filepaths: List of paths to data files.
    :param model: Model name as a string.
    :param scenario: Scenario name as a string.
    :param year: Year as an integer.
    :param parameter_keys: List of parameter keys used in intensity iterations.
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

    excel_path = f"stats_report_{model}_{scenario}_{year}.xlsx"

    try:
        with pd.ExcelWriter(
            excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
        ) as writer:
            mapping_df.to_excel(writer, index=False, sheet_name="Mapping")
    except Exception as e:
        print(f"Error writing mapping sheet to {excel_path}: {str(e)}")


def escape_formula(text: str):
    """
    Prevent a string from being interpreted as a formula in Excel.
    Strings starting with '=', '-', or '+' are prefixed with an apostrophe.

    :param text: The string to be adjusted.
    :return: The adjusted string.
    """
    return "'" + text if text.startswith(("=", "-", "+")) else text


def run_stats_analysis(model: str, scenario: str, year: int, methods: list):
    """
    Runs OLS regression for specified methods and writes summaries to an Excel file.

    Each method's OLS summary is placed on a new sheet in the file named
    'stats_report_{model}_{scenario}_{year}.xlsx'.

    :param model: Model identifier.
    :param scenario: Scenario name.
    :param year: Year of interest.
    :param methods: Methods corresponding to dataset columns.
    """

    filename = f"stats_report_{model}_{scenario}_{year}.xlsx"

    try:
        book = load_workbook(filename)
    except FileNotFoundError:
        book = pd.ExcelWriter(
            filename, engine="openpyxl"
        )  # Create a new workbook if not found
        book.close()
        book = load_workbook(filename)

    data = pd.read_excel(filename, sheet_name="Sheet1")

    for idx, method in enumerate(methods):
        if method not in data.columns:
            print(f"Data for {method} not found in the file.")
            continue

        Y = data[method]
        X = data.drop(columns=["Iteration", "Year"] + methods)
        X = sm.add_constant(X)

        model_results = sm.OLS(Y, X).fit()
        summary = model_results.summary().as_text()

        sheet_name_base = f"{method[:20]} OLS"
        sheet_name = f"{sheet_name_base} {idx + 1}"
        while sheet_name in book.sheetnames:
            idx += 1
            sheet_name = f"{sheet_name_base} {idx + 1}"

        if sheet_name in book.sheetnames:
            std = book[sheet_name]
            book.remove(std)
        ws = book.create_sheet(sheet_name)

        summary_lines = summary.split("\n")

        for line in summary_lines:
            line = escape_formula(line)
            columns = re.split(r"\s{2,}", line)
            ws.append(columns)

    book.save(filename)
    print("Analysis complete and results saved.")
