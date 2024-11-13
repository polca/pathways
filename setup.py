import os
from pathlib import Path

from setuptools import setup

packages = []
root_dir = os.path.dirname(__file__)
if root_dir:
    os.chdir(root_dir)

# read the contents of your README file
this_directory = Path(__file__).parent
README = (this_directory / "README.md").read_text()

# Probably should be changed, __init__.py is no longer required for Python 3
for dirpath, dirnames, filenames in os.walk("pathways"):
    # Ignore dirnames that start with '.'
    if "__init__.py" in filenames:
        pkg = dirpath.replace(os.path.sep, ".")
        if os.path.altsep:
            pkg = pkg.replace(os.path.altsep, ".")
        packages.append(pkg)


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


setup(
    name="pathways",
    version="1.0.0",
    python_requires=">=3.9,<3.12",
    packages=packages,
    author="Romain Sacchi <romain.sacchi@psi.ch>",
    license=open("LICENSE").read(),
    # Only if you have non-python data (CSV, etc.).
    # Might need to change the directory name as well.
    include_package_data=True,
    install_requires=[
        "numpy==1.24.4",
        "pathlib",
        "pandas",
        "xarray<=2024.2.0",
        "bw2calc >= 2.0.dev18",
        "bw2data >= 4.0.dev59",
        "scipy",
        "premise",
        "pyyaml",
        "bw_processing",
        "bw2calc>=2.0.dev18",
        "datapackage",
        "pyprind",
        "platformdirs",
        "fs",
        "statsmodels",
        "SALib",
        "pyarrow",
        "fastparquet",
    ],
    url="https://github.com/polca/pathways",
    description="Scenario-level LCA of energy systems and transition pathways",
    long_description_content_type="text/markdown",
    long_description=README,
    classifiers=[
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
