from pathways import Pathways, configure_logging

configure_logging(mode="per-run", console=False, run_tag="SSP2-2050")

p = Pathways(
    datapackage="pathways_2025-10-22.zip",
    # geography_mapping="geo_mapping_remind.yaml",
    # activities_mapping="act_categories_agg.yaml",
)


vars = [v for v in p.scenarios.coords["variables"].values if v.startswith("FE")]
print(f"Calculating {len(vars)} variables")

p.calculate(
    methods=[
        "EF v3.1 EN15804 - climate change - global warming potential (GWP100)",
        "EF v3.1 EN15804 - ecotoxicity: freshwater - comparative toxic unit for ecosystems (CTUe)",
    ],
    regions=[
        "World",
    ],
    scenarios=p.scenarios.pathway.values.tolist(),
    years=[
        2020,
        2030,
        2040,
        2050,
    ],
    variables=vars,
    # use_distributions=10,
    # subshares=True,
)

p.export_results()
