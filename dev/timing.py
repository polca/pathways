from pathways import Pathways

p = Pathways(
    datapackage="remind-SSP2-PkBudg1150-stem-SPS1.zip",
    geography_mapping="geo_mapping_remind.yaml"
)

vars = [v for v in p.scenarios.coords["variables"].values if v.startswith("FE")]

p.calculate(
    methods=[
        "EF v3.1 EN15804 - climate change - global warming potential (GWP100)",
        "EF v3.1 EN15804 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
    ]
    + [m for m in p.lcia_methods if "relics" in m.lower()][-5:],
    regions=[
        "CH",
    ],
    scenarios=p.scenarios.pathway.values.tolist(),
    years=[
        2020,
        2030,
        2040,
        2050
    ],
    variables=vars,
    use_distributions=1000,
    subshares=True,
    multiprocessing=True,
    statistical_analysis=True,
)

p.export_results()
