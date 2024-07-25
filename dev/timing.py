from pathways import Pathways

p = Pathways(datapackage="remind-SSP2-PkBudg1150-stem-SPS1.zip")

vars = [v for v in p.scenarios.coords["variables"].values if v.startswith("FE")]

p.calculate(
    methods=[
        "EF v3.1 EN15804 - climate change - global warming potential (GWP100)",
        "EF v3.1 EN15804 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
    ]
    + [m for m in p.lcia_methods if "relics" in m.lower()][-3:],
    regions=[
        "CH",
    ],
    scenarios=p.scenarios.pathway.values.tolist(),
    years=[2020, 2030, 2040, 2050],
    variables=vars,
    use_distributions=100,
    subshares=True,
    multiprocessing=False,
)

p.export_results()
