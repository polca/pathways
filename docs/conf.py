import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "pathways"
copyright = "2025"
author = "Paul Scherrer Institute"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = []

# docs/conf.py
autodoc_mock_imports = [
    "numpy", "pandas", "bw2calc", "bw2data", "sparse", "xarray",
    "prettytable", "tqdm", "constructive_geometries"
]



html_theme = "alabaster"
html_static_path = ["_static"]
html_logo = "https://raw.githubusercontent.com/polca/pathways/refs/heads/main/assets/pathways-high-resolution-logo-transparent.png"

import os
import sys

sys.path.insert(0, os.path.abspath("../"))  # or '../src' if your code is in src/
