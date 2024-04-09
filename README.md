<p align="center">
<img src="https://github.com/polca/pathways/blob/main/assets/pathways-high-resolution-logo-transparent.png" height="300"/>
</p>


# pathways

``pathways`` is a Python package that characterizes the
environmental impacts of products, secctors or energy systems 
and transition scenarios over time using Life Cycle Assessment (LCA).
Compared to traditional scenario results from energy models, 
``pathways`` provides a more detailed and transparent view of the
environmental impacts of a scenario by resolving supply chains
between producers and consumers (as an LCA does). Hence, direct
and indirect emissions are accounted for, and double-counting
issues are partially resolved. 

``pathways`` is initially designed to work with data packages produced
by ``premise``, but can be used with any IAM scenarios and LCA databases.

## Installation

``pathways`` is in an early development stage, and
can be installed from the Github repo  with ``pip``:

```bash

  pip install pathways

```


## Usage

``pathways`` is a Python package, and can be used in Python scripts
or in a Python interpreter.

### Python

```python

from pathways import Pathways
p = Pathways(datapackage="some datapackage.zip")
p.calculate(
    methods=[
            "EF v3.1 - acidification - accumulated exceedance (AE)"
        ],
    years=[2080, 2090, 2100],
    regions=["World"],
    scenarios=["SSP2-Base", "SSP2-RCP26",]
)

```

The argument `datapackage` is the path to the datapackage.zip file
that describes the scenario and the LCA databases -- see dev/sample.
The argument `methods` is a list of methods to be used for the LCA
calculations. The argument `years` is a list of years for which the
LCA calculations are performed. The argument `regions` is a list of
regions for which the LCA calculations are performed. The argument
`scenarios` is a list of scenarios for which the LCA calculations are
performed.

If not specified, all the methods, years, regions and scenarios
defined in the datapackage.json file are used, which can be very
time-consuming.

Once calculated, the results of the LCA calculations are stored in the `.lcia_results`
attribute of the `Pathways` object as an ``xarray.DataArray``. 

```python

p.lcia_results

```


It can be further formatted
to a pandas' DataFrame or export to a CSV/Excel file using the built-in
methods of ``xarray``.


which can then be visualized using your favorite plotting library.
![Screenshot](assets/screenshot.png)

## Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

### Types of Contributions

#### Report Bugs

Report bugs by filing issues on GitHub.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.
* For visual bugs, a screenshot or animated GIF of the bug in action.

#### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to
implement it.

#### Write Documentation

``pathways`` could always use more documentation, whether as part of
the official ``pathways`` docs, in docstrings, or even on the web in
blog posts, articles, and such.

#### Submit Feedback

The best way to send feedback is to file an issue on the GitHub repository.

## Credits

### Contributors

* [Romain Sacchi](https://github.com/romainsacchi)
* Alvaro Hahn Menacho (https://github.com/alvarojhahn)


### Financial Support

* [PRISMA project](https://www.net0prisma.eu/)


## License

``pathways`` is licensed under the terms of the BSD 3-Clause License.

