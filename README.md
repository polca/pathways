# pathways

``pathways`` is a Python package that characterizes
environmental impacts of energy systems and transition scenarios using
Life Cycle Assessment (LCA).

``pathways`` is a work in progress. It reads in
scenarios and corresponding premise-generated LCA databases,
and calculates the environmental impacts over a defined period.

## Installation

For now ``pathways`` can be installed form the Github repo  with ``pip``:

```bash

  pip install git+https://github.com/romainsacchi/pathways.git

```


## Usage

``pathways`` is a Python package, and can be used in Python scripts
or in a Python interpreter. It is also a command-line tool, and can
be used in a terminal.

### Python

```python

from pathways import Pathways
p = Pathways(datapackage="sample/datapackage.json")
p.calculate(
    methods=[
            "EF v3.1 - acidification - accumulated exceedance (AE)"
        ],
    years=[2080, 2090, 2100],
    regions=["World"],
    scenarios=["SSP2-Base", "SSP2-RCP26",]
)

```

### Command-line

```bash

  pathways --help

```

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

#### Provide IAM scenarios

The IAM scenarios are the core of the ``pathways`` package. If you
have access to IAM scenarios, please consider sharing them with us.

#### Write Documentation

``pathways`` could always use more documentation, whether as part of
the official ``pathways`` docs, in docstrings, or even on the web in
blog posts, articles, and such.

#### Submit Feedback

The best way to send feedback is to file an issue on GitHub.



## Credits

### Contributors

* [Romain Sacchi](https://github.com/romainsacchi)


### Financial Support

* [PRISMA project](https://www.net0prisma.eu/)


## License

``pathways`` is licensed under the terms of the BSD 3-Clause License.

