API Reference
=============

Pathways
--------

.. automodule:: pathways
   :members:
   :undoc-members:
   :show-inheritance:

   **Key methods**

   - :py:meth:`pathways.Pathways.calculate` —
     Compute LCA results for selected methods, models, scenarios, regions, years, and variables.
     Parameters include ``demand_cutoff``, ``use_distributions``, ``subshares``,
     ``remove_uncertainty``, ``seed``, ``solver``, the ``iterative_*`` tuning
     arguments, ``aggregate_by``, ``multiprocessing``,
     ``postprocess_multiprocessing``, and ``double_accounting``.
     ``solver="direct"`` uses ``bw2calc.MultiLCA`` and
     ``solver="jacobi-gmres"`` selects the experimental iterative backend.

   - :py:meth:`pathways.Pathways.aggregate_results` —
     Aggregate low-contribution activity categories under ``"other"``; optional interpolation.

   - :py:meth:`pathways.Pathways.export_results` —
     Export **non-zero** results to compressed Parquet (``.gzip``).
