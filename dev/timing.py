from pathways import Pathways

p = Pathways(datapackage="./image-SSP2-RCP19/datapackage.json")

for scenario in [
    # "SSP2-Base",
    "SSP2-RCP19"
]:
    p.calculate(
        methods=[
            "RELICS - metals extraction - Lithium",
            "RELICS - metals extraction - Molybdenum",
        ],
        regions=[
            "WEU",
        ],
        scenarios=[scenario],
        years=[
            2020,
        ],
        variables=[
            v
            for v in p.scenarios.coords["variables"].values
            if any(i in v for i in ["Industry", "Transport", "Heating"])
        ],
        demand_cutoff=0.01,
    )
    arr = p.display_results()
    arr.to_netcdf(f"results_image_{scenario}.nc")
