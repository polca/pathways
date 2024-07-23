from pathways import Pathways

p = Pathways(datapackage="remind-SSP2-PkBudg1150-stem-SPS1.zip")


p.calculate(
    methods=[
        "EF v3.1 EN15804 - climate change - global warming potential (GWP100)",
        "EF v3.1 EN15804 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
    ],
    regions=[
        "CH",
    ],
    scenarios=p.scenarios.pathway.values.tolist(),
    years=[
        2050,
    ],
    variables=[v for v in p.scenarios.coords["variables"].values if v.startswith("FE")],
    use_distributions=20,
    subshares=True,
    multiprocessing=False,
)
