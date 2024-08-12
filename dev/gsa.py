import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from SALib.analyze import delta

directory = "/data/user/sacchi_r/stats_/"


def run_GSA_delta(
    total_impacts: pd.DataFrame,
    uncertainty_values: pd.DataFrame,
    technology_shares: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs Delta Moment-Independent Measure analysis for specified methods and writes summaries to an Excel file.

    :param total_impacts: DataFrame with total impacts for each method.
    :param uncertainty_values: DataFrame with uncertainty values.
    :param technology_shares: DataFrame with technology shares.
    :return: DataFrame with Delta Moment-Independent Measure analysis results.
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


def gsa(file):
    print(file)
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

        df_GSA_results.to_excel(writer, sheet_name=f"GSA", index=False)


start = time.time()
print(start)

print(Path(directory).glob("*.xlsx"))

with Pool(cpu_count()) as pool:
    pool.map(gsa, Path(directory).glob("*.xlsx"))

end = time.time()
print(end - start)
