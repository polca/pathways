from pathways import Pathways

p = Pathways(datapackage="/Users/romain/GitHub/premise/dev/image-SSP2/datapackage.json")

for scenario in [
    # "SSP2-Base",
    "SSP2-RCP19"
]:
    p.calculate(
        methods=[
            "EF v3.1 - acidification - accumulated exceedance (AE)",
            "EF v3.1 - climate change - global warming potential (GWP100)",
            "EF v3.1 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
            "EF v3.1 - energy resources: non-renewable - abiotic depletion potential (ADP): fossil fuels",
            "EF v3.1 - eutrophication: freshwater - fraction of nutrients reaching freshwater end compartment (P)",
            "EF v3.1 - human toxicity: carcinogenic - comparative toxic unit for human (CTUh)",
            "EF v3.1 - material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves)",
            "EF v3.1 - particulate matter formation - impact on human health",
            "EF v3.1 - water use - user deprivation potential (deprivation-weighted water consumption)",
            "RELICS - metals extraction - Aluminium",
            "RELICS - metals extraction - Cobalt",
            "RELICS - metals extraction - Copper",
            "RELICS - metals extraction - Graphite",
            "RELICS - metals extraction - Lithium",
            "RELICS - metals extraction - Molybdenum",
            "RELICS - metals extraction - Neodymium",
            "RELICS - metals extraction - Nickel",
            "RELICS - metals extraction - Platinum",
            "RELICS - metals extraction - Vanadium",
            "RELICS - metals extraction - Zinc",
        ],
        regions=[r for r in p.scenarios.coords["region"].values if r != "World"],
        scenarios=[scenario],
        years=[2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100],
        variables=[
            v
            for v in p.scenarios.coords["variables"].values
            if any(i in v for i in ["Industry", "Transport", "Heating"])
        ],
        demand_cutoff=0.01,
    )
    arr = p.display_results(cutoff=0.0001)

    arr.to_netcdf(f"results_image_{scenario}.nc")
