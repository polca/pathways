Quickstart
==========

Install
-------

.. code-block:: bash

   pip install pathways

Minimal example
---------------

Compute scenario-driven LCA results, aggregate them by activity category with a cut-off,
and export the non-zero entries to Parquet:

.. code-block:: python

   from pathways import Pathways

   # 1) Load the datapackage (zip or folder)
   pw = Pathways("remind-SSP2-NPi.zip", ecoinvent_version="3.12")

   # 2) Inspect available LCIA methods
   print("Available LCIA methods:", pw.lcia_methods)

   # 3) Calculate results (pick what you need; None means “all available”)
   pw.calculate(
       methods=["AWARE"],                # impact categories to compute
       models=["REMIND"],                # model(s) present in the datapackage
       scenarios=["SSP2-NPi"],           # scenario(s)
       regions=["World", "Europe"],      # IAM regions or national codes
       years=[2020, 2030, 2050],         # time points
       variables=["Electricity|Generation"],  # scenario variables to map
       demand_cutoff=1e-3,               # drop tiny demands before solving
       use_distributions=0,              # 0 deterministic, >0 enables sampling
       subshares=False,                  # use sub-share allocation if provided
       remove_uncertainty=False,         # strip CF uncertainty if True
       seed=0,                           # RNG seed
       multiprocessing=True              # parallelize across model/scenario/year
   )

   # Results are an xarray DataArray with dims:
   # (act_category, variable, year, region, location, model, scenario, impact_category)
   pw.lca_results

   # 4) Aggregate for display: cut small contributions by act_category, optional interpolation
   pw.aggregate_results(cutoff=0.01, interpolate=False)

   # 5) Export non-zero cells to compressed Parquet
   out = pw.export_results("results_baseline")
   print("Wrote:", out)  # e.g., results_baseline.gzip

Large Monte Carlo runs
----------------------

When ``use_distributions`` is greater than zero, you can switch from the
default direct solver to the experimental iterative backend and reduce cache
size by collapsing dimensions you do not need later:

.. code-block:: python

   pw.calculate(
       methods=["AWARE"],
       models=["REMIND"],
       scenarios=["SSP2-NPi"],
       regions=["World"],
       years=[2050],
       variables=["Electricity|Generation"],
       use_distributions=300,
       solver="jacobi-gmres",
       iterative_rtol=1e-8,
       aggregate_by=["act_category", "location"],
       multiprocessing=True,
       postprocess_multiprocessing=True,
   )

``solver="direct"`` remains the default. ``aggregate_by`` currently supports
``"act_category"`` and ``"location"`` and replaces those coordinates with a
single ``"aggregated"`` label in the resulting array, so use it only when you
do not need detailed attribution along those dimensions.
