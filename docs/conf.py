import os
import sys

# Ensure the installed package (and local sources during RTD builds) are importable
sys.path.insert(0, os.path.abspath(".."))

project = "Pathways"
copyright = "2025"
author = "Paul Scherrer Institute"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "bw2calc",
    "bw2data",
    "sparse",
    "xarray",
    "prettytable",
    "tqdm",
    "constructive_geometries",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "../assets/pathways-high-resolution-logo-transparent.png"
