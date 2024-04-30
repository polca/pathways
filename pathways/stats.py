import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import re
from openpyxl import load_workbook


def log_subshares_to_excel(model, scenario, year, shares):
    """
    Logs results to an Excel file named according to model, scenario, and year, specifically for the given year.
    This method assumes that each entry in the shares defaultdict is structured to be directly usable in a DataFrame.

    Parameters:
    - model (str): The model name.
    - scenario (str): The scenario name.
    - year (int): The specific year for which data is being logged.
    - shares (defaultdict): A nested defaultdict containing shares data for multiple years and types.

    Creates:
    - An Excel file with data logged for specified parameters.
    """
    filename = f"stats_report_{model}_{scenario}_{year}.xlsx"
    data = []

    # Check if the year data exists for any technology
    sample_tech = next(iter(shares), None)
    if sample_tech and year in shares[sample_tech]:
        # Create data for each iteration
        num_iterations = len(shares[sample_tech][year][next(iter(shares[sample_tech][year]))])
        for i in range(num_iterations):
            iteration_data = {'Iteration': i + 1, 'Year': year}
            for tech, years_data in shares.items():
                if year in years_data:
                    for subtype, values in years_data[year].items():
                        iteration_data[f'{tech}_{subtype}'] = values[i]
            data.append(iteration_data)

        new_df = pd.DataFrame(data)
        try:
            # Try to load the existing Excel file
            with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                existing_df = pd.read_excel(filename)
                # Combine the old data with the new data, aligning on 'Iteration' and 'Year'
                final_df = pd.merge(existing_df, new_df, on=['Iteration', 'Year'], how='outer')
                # Reorder columns to ensure 'Iteration' and 'Year' are first, followed by any new columns
                column_order = ['Iteration', 'Year'] + [c for c in new_df.columns if c not in ['Iteration', 'Year']]
                final_df = final_df[column_order]
                final_df.to_excel(writer, index=False)
        except FileNotFoundError:
            # If the file doesn't exist, create it
            new_df.to_excel(filename, index=False)
    else:
        print(f"Year {year} not found in shares for any technology.")


def log_intensities_to_excel(model: str, scenario: str, year: int, new_data: dict):
    """
    Update or create an Excel file with new columns of data, based on model, scenario, and year.

    :param model: The model name.
    :param scenario: The scenario name.
    :param year: The year for which the data is logged.
    :param new_data: Dictionary where keys are the new column names and values are lists of data for each column.
    """
    filename = f'stats_report_{model}_{scenario}_{year}.xlsx'

    try:
        df = pd.read_excel(filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    if 'Iteration' not in df.columns or df.empty:
        max_length = max(len(data) for data in new_data.values())
        df['Iteration'] = range(1, max_length + 1)
        df['Year'] = [year] * max_length

    if not df.empty and len(df) != len(new_data[next(iter(new_data))]):
        df = df.iloc[:len(new_data[next(iter(new_data))])]

    for column_name, data in new_data.items():
        if len(data) != len(df):
            raise ValueError(f"Length of data for '{column_name}' ({len(data)}) does not match DataFrame length ({len(df)}).")
        df[column_name] = data

    df.to_excel(filename, index=False)



def log_results_to_excel(
        model: str,
        scenario: str,
        year: int,
        total_impacts_by_method: dict,
        methods: list,
        filepath=None):
    """
    Log the characterized inventory results for each LCIA method into separate columns in an Excel file.

    :param model: The model name.
    :param scenario: The scenario name.
    :param year: The year for which the data is being logged.
    :param total_impacts_by_method: Dictionary where keys are method names and values are lists of impacts
    from all regions and distributions.
    :param methods: List of method names.
    :param filepath: Optional. File path for the Excel file to save the results.
    Defaults to 'stats_report_{model}_{scenario}_{year}.xlsx' if not provided.
    """

    if filepath is None:
        filepath = f"stats_report_{model}_{scenario}_{year}.xlsx"

    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        df = pd.DataFrame()

    for method, impacts in total_impacts_by_method.items():
        df[method] = pd.Series(impacts)

    base_cols = ['Iteration', 'Year'] if 'Iteration' in df.columns else []
    other_cols = [col for col in df.columns if col not in base_cols + methods]
    df = df[base_cols + methods + other_cols]

    df.to_excel(filepath, index=False)


def create_mapping_sheet(filepaths: list, model: str, scenario: str, year: int, parameter_keys: list):
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

    # Convert parameter keys into a set of unique indices
    unique_indices = {int(idx) for key in parameter_keys for idx in key.split('_to_')}

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

    # Filter the DataFrame using unique indices
    mapping_df = technosphere_inds[technosphere_inds['Index'].isin(unique_indices)]
    mapping_df = mapping_df[["Activity", "Product", "Location", "Unit", "Index"]]  # Restrict columns if necessary

    excel_path = f"stats_report_{model}_{scenario}_{year}.xlsx"

    try:
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
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
    return "'" + text if text.startswith(('=', '-', '+')) else text


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

    filename = f'stats_report_{model}_{scenario}_{year}.xlsx'

    # Attempt to load the existing workbook
    try:
        book = load_workbook(filename)
    except FileNotFoundError:
        book = pd.ExcelWriter(filename, engine='openpyxl')  # Create a new workbook if not found
        book.close()
        book = load_workbook(filename)

    data = pd.read_excel(filename, sheet_name='Sheet1')

    for idx, method in enumerate(methods):
        if method not in data.columns:
            print(f"Data for {method} not found in the file.")
            continue

        Y = data[method]
        X = data.drop(columns=['Iteration', 'Year'] + methods)
        X = sm.add_constant(X)

        model_results = sm.OLS(Y, X).fit()
        summary = model_results.summary().as_text()

        # Create a unique sheet name
        sheet_name_base = f"{method[:20]} OLS"
        sheet_name = f"{sheet_name_base} {idx + 1}"
        while sheet_name in book.sheetnames:
            idx += 1
            sheet_name = f"{sheet_name_base} {idx + 1}"

        if sheet_name in book.sheetnames:
            std = book[sheet_name]
            book.remove(std)
        ws = book.create_sheet(sheet_name)

        # Split summary into lines and write upper part to the sheet
        summary_lines = summary.split('\n')
        upper_part = summary_lines[:10]
        lower_part = summary_lines[10:]

        # for line in upper_part:
        #     line = escape_formula(line)
        #     ws.append([line])

        # Process and write lower part to the sheet
        # for line in lower_part:
        for line in summary_lines:
            line = escape_formula(line)
            # Split line based on consecutive spaces for proper column separation
            columns = re.split(r'\s{2,}', line)
            ws.append(columns)

    book.save(filename)
    print("Analysis complete and results saved.")



