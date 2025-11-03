Examples
========

The repository ships with a miniature datapackage under
``example/datapackage_sample``.  The snippets below demonstrate how to work with
it using :class:`pathways.Pathways`.

Load the datapackage
--------------------

.. code-block:: python

   from pathlib import Path

   from pathways import Pathways

   sample_pkg = Path("example/datapackage_sample")
   pw = Pathways(sample_pkg, ecoinvent_version="3.11", debug=True)

   print(pw.scenarios["models"])   # -> ['ModelX']
   print(pw.lcia_methods[:3])       # show first available LCIA methods

Slice the scenario catalog
--------------------------

You can limit the calculations to a subset of the available dimensions by
passing iterables.  The helper below focuses on **ModelX**, the
``baseline`` scenario, two years, and a single region.

.. code-block:: python

   pw.calculate(
       methods=["IPCC 2021 climate change GWP 100a"],
       models=["ModelX"],
       scenarios=["baseline"],
       regions=["World"],
       years=[2020, 2030],
       variables=["Electricity|Generation"],
       demand_cutoff=1e-3,
   )

   print(pw.lca_results.sel(impact_category="IPCC 2021 climate change GWP 100a"))

Monte Carlo sampling and export
-------------------------------

Set ``use_distributions`` to a non-zero integer to trigger Monte Carlo draws
for the technosphere uncertainty parameters stored in the datapackage.  The
example below runs five iterations and writes both the aggregated results and
per-iteration outputs to disk.

.. code-block:: python

   pw.calculate(
       methods=["IPCC 2021 climate change GWP 100a"],
       models=["ModelX"],
       scenarios=["baseline"],
       regions=["World"],
       years=[2030],
       variables=["Electricity|Generation"],
       use_distributions=5,
       remove_uncertainty=False,
   )

   # Collapse small contributions (<0.1%) into an "other" bucket
   pw.aggregate_results(cutoff=0.001)

   # Export the dense tensor to a compressed Parquet file
   output = pw.export_results("sample_results")
   print(f"Results written to {output}")

   # Optional: inspect Monte Carlo parameter samples saved during calculate()
   from pathways.filesystem_constants import STATS_DIR

   mc_book = STATS_DIR / "ModelX_baseline_2030.xlsx"
   print(mc_book.exists())
