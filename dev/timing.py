from pathways import Pathways

p = Pathways(datapackage="./image-SSP2-RCP19/datapackage.json")

for scenario in [
    # "SSP2-Base",
    "SSP2-RCP19"
]:
    p.calculate(
        methods=[
            "RELICS - metals extraction - Lithium",
            # "RELICS - metals extraction - Molybdenum",
        ],
        regions=[
            "WEU",
            #"USA",
        ],
        scenarios=[scenario],
        years=[
            2010,
            2020,
            2030,
        ],
        variables=[
            v
            for v in p.scenarios.coords["variables"].values
            if any(i in v for i in ["Industry", "Transport", "Heating"])
        ],
        demand_cutoff=0.01,
        multiprocessing=True,
    )
    arr = p.display_results()
    arr.to_netcdf(f"results_image_{scenario}.nc")
