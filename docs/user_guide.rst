User Guide
==========

Overview
--------

Pathways consumes a **datapackage** with:

- **Scenario data:** model, scenario, year, region, variables, values
- **Mappings:** links from scenario variables → LCA activities (and optional subshares)
- **Matrices:** prebuilt technosphere/biosphere/characterization elements

The main workflow is driven by :class:`pathways.Pathways`.

Pathways workflow
-----------------

1. **Initialize**

   .. code-block:: python

      from pathways import Pathways
      pw = Pathways("remind-SSP2-NPi.zip", ecoinvent_version="3.12")

   The initializer parses the datapackage, builds internal indices, loads
   classification info and LCIA method names, and prepares the scenario catalog:

   - ``pw.scenarios``: available ``models``, ``scenarios``, ``regions``, ``years``
   - ``pw.lcia_methods``: available impact categories (method names)
   - ``pw.units``: unit conversion helpers

   **Example**

   .. code-block:: python

      print("LCIA methods:", pw.lcia_methods)
      # e.g. ["AWARE", "IPCC 2021 climate change GWP 100a", ...]

2. **Calculate**

   .. code-block:: python

      pw.calculate(
          methods=["AWARE"], models=["REMIND"], scenarios=["SSP2-NPi"],
          regions=["World"], years=[2020, 2050],
          demand_cutoff=1e-3, use_distributions=0
      )

   ``methods`` must be chosen from the list returned by ``pw.lcia_methods``.

   **Output shape**

   ``pw.lca_results`` is an :mod:`xarray` **DataArray** with dimensions:

   ``(act_category, variable, year, region, location, model, scenario, impact_category)``

3. **Aggregate for display**

   .. code-block:: python

      pw.aggregate_results(cutoff=0.01, interpolate=False)

   Aggregates contributions below the cutoff into an ``"other"`` category.

4. **Export to Parquet**

   .. code-block:: python

      path = pw.export_results("my_results")  # → my_results.gzip

Notes
-----

- Supported ``ecoinvent_version`` values are ``"3.10"``, ``"3.11"``, and ``"3.12"``.
- Matrix files ``A_matrix.csv`` and ``B_matrix.csv`` are semicolon-delimited and may include a header row.
- Index files ``A_matrix_index.csv`` and ``B_matrix_index.csv`` may include a header row.
