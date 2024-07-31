from pathways import Pathways, run_gsa

p = Pathways(
    datapackage="remind-SSP2-PkBudg1150-stem-SPS1.zip",
    geography_mapping="geo_mapping_remind.yaml",
    #activities_mapping="act_categories_agg.yaml",
)

vars = [v for v in p.scenarios.coords["variables"].values if v.startswith("FE")]

p.calculate(
    methods=[
        "EF v3.1 EN15804 - climate change - global warming potential (GWP100)",
        "EF v3.1 EN15804 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
    ],
    regions=["CH"],
    scenarios=p.scenarios.pathway.values.tolist(),
    years=[2020, 2030, 2040, 2050],
    variables=vars,
    use_distributions=10,
    subshares=True,
)

p.export_results()

print(p.lca_results.coords)
print(p.lca_results.shape)
print(p.lca_results.sum())

run_gsa()
