import time

from pathways import Pathways

for scenario in [
    "/data/user/sacchi_r/remind-SSP2-PkBudg1150-stem-SPS1.zip",
]:
    start = time.time()
    p = Pathways(
        datapackage=scenario,
        debug=False,
        geography_mapping="/data/user/sacchi_r/geo_mapping_remind.yaml",
        activities_mapping="/data/user/sacchi_r/act_categories_agg.yaml",
    )

    p.calculate(
        methods=[
            "EF v3.1 EN15804 - climate change - global warming potential (GWP100)",
            "EF v3.1 EN15804 - acidification - accumulated exceedance (AE)",
            "EF v3.1 EN15804 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
            "EF v3.1 EN15804 - material resources: metals/minerals - abiotic depletion potential (ADP): elements (ultimate reserves)",
            "EF v3.1 EN15804 - eutrophication: freshwater - fraction of nutrients reaching freshwater end compartment (P)",
            "EF v3.1 EN15804 - photochemical oxidant formation: human health - tropospheric ozone concentration increase",
            "Inventory results and indicators - resources - total freshwater extraction",
            "Crustal Scarcity Indicator 2020 - material resources: metals/minerals - crustal scarcity potential (CSP)",
            "Inventory results and indicators - resources - total surface occupation",
            "Inventory results and indicators - resources - land occupation",
            "ReCiPe 2016 v1.03, midpoint (H) - particulate matter formation - particulate matter formation potential (PMFP)",
        ]
        + [m for m in p.lcia_methods if "RELICS" in m],
        regions=["CH"],
        scenarios=p.scenarios.pathway.values.tolist(),
        years=[2020, 2030, 2040, 2050],
        variables=[
            v for v in p.scenarios.coords["variables"].values if v.startswith("FE")
        ],
        use_distributions=400,
        subshares=True,
        remove_uncertainty=True,
        multiprocessing=True,
    )
    p.export_results()
    end = time.time()
    print(end - start)
