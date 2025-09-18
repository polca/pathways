Welcome to Pathways’ documentation!
===================================

Pathways provides tools for **prospective life cycle assessment (LCA)** driven by
**scenario pathways** and **impact assessment**. It reads a *datapackage* containing:

- Scenario data (e.g., model/scenario/year/region → variables/values)
- Mapping between scenario variables and LCA activities
- Prebuilt LCA matrices

You can then compute multi-year, multi-region impact results with a single call to
:py:meth:`pathways.Pathways.calculate`, aggregate the results with
:py:meth:`pathways.Pathways.aggregate_results`, and export them to a compact
Parquet file with :py:meth:`pathways.Pathways.export_results`.

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   user_guide
   api_references

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
