---
title: '`pathways`: life cycle assessment of energy transition scenarios'
tags:
  - Python
  - life cycle assessment
  - prospective
  - database
  - premise
  - scenario
  - integrated assessment models
  - energy system models

authors:
  - name: Romain Sacchi
    orcid: 0000-0003-1440-0905
    affiliation: 1
  - name: Alvaro J. Hahn-Menacho
    orcid: 0000-0003-2592-2186
    affiliation: "1,2"

affiliations:
 - name: Laboratory for Energy Systems Analysis, Centers for Energy and Environmental Sciences and Nuclear Engineering and Sciences, Paul Scherrer Institute, 5232 Villigen, Switzerland
   index: 1
 - name: Chair of Energy Systems Analysis, Institute of Energy and Process Engineering, ETH Zürich, 8092 Zürich, Switzerland
   index: 2

date: 24 May 2024
bibliography: paper.bib

---

# Summary


`pathways` is a Python package that conducts Life Cycle Assessment (LCA) to evaluate 
the environmental impacts of products, sectors, or transition scenarios over time. 
Unlike most energy (ESM) or integrated assessment models (IAM), `pathways` offers 
a clearer view on impacts caused by a scenario by considering supply chain relations 
between producers and consumers, thereby addressing direct and indirect emissions. 
Unlike the reported emissions in ESM and IAM scenarios, which focus primarily on operation,
`pathways` allows reporting the environmental impacts of infrastructure build-up 
and decommissioning. Finally, scenarios can be characterized across a wide range of 
indicators which are usually not included in ESM or IAM: land use, water consumption,
toxicity impacts, etc.

# Statement of need

Most IAMs and ESMs project cost- or utility-optimized future scenarios within 
specified greenhouse gas emissions trajectories, outlining changes needed 
in regional energy mixes and means of transport for global warming mitigation [@Riahi:2017]. 
Prospective Life Cycle Assessment (pLCA) is crucial for evaluating the 
environmental performance of existing and emerging production systems, with 
a growing body of literature in scenario-based pLCA for emerging technologies 
[@Bisinella:2021].

Extending present-day life-cycle inventories into the future using IAM outputs, 
initially explored by [@MendozaBeltran:2018] and formalized by the Python library 
`premise` [@Sacchi:2022], forms the methodological basis for pLCA. Efforts in pLCA 
focus on improving forecasting accuracy. Performing system-wide LCAs with 
adjusted life cycle inventories at each time step has potential to enhance 
sustainability assessments, broadening focus beyond greenhouse gas emissions 
to include broader environmental impacts like land use, water consumption, 
and toxicity, addressing both direct and indirect emissions. However, system-wide 
LCA remains challenging due to computational costs and methodological 
complexities, such as defining functional units based on IAM outputs and 
resolving double-counting issues [@Vandepaer:2020],[@Volkart:2018].

Several studies characterize energy scenarios with LCA, including 
[@Gibon:2015], [@Rauner:2017] and [@Pehl:2017], who quantified ESM or 
IAM scenario outputs using a hybrid-LCA framework. There is also the work of
[@Xu:2020], who developed the ambitious EAFESA framework aiming for 
bidirectional coupling between ESM and LCA. Yet, these studies focused 
on specific sectors or technologies and have not yet generalized to broader 
scenarios and indicators, nor have they made their implementations widely available.

Beyond conventional pLCA approaches, several tools and frameworks have been developed
that leverage LCA data to support further analysis, often through automation and integration
with broader modeling frameworks. For example, the `ODYM-RECC` framework integrates LCA data
to assess resource efficiency within climate mitigation scenarios, providing insights on 
material demand and supply chain impacts [@RECC:2021]. Similarly, the `Mat-dp` tool, when 
supplied with suitable input data, can be used to calculate materials needed and estimate 
environmental impacts of transition scenarios [@Mat-dp:2022], [@Mat-dp:2024]. However, because 
these tools depend on exogeneous input data, they are not designed to 
systematically consider the time-dependent technology mixes influencing the production system. 
This limits their ability to endogenously and dynamically assess evolving environmental impacts 
and material demand, restricting consistency with the scenario assessed.

To address these challenges, the open-source library `pathways` utilizes the 
LCA framework `brightway` [@Mutel:2017] to systematically evaluate 
environmental impacts of energy transition scenarios. `pathways` works with 
data packages containing LCA matrices adjusted to each time step of the 
ESM/IAM scenario, providing detailed and transparent insights into 
scenario environmental impacts. `pathways` works particularly well with
data packages produced by `premise`, but can be used with any ESM/IAM scenarios
and LCA databases. Using LCA matrices which have been modified to reflect
the scenario's time-dependent technology mixes ensures a consistent and coherent
characterization of said scenario.


# Description

`pathways` reads a data package containing scenario data, mapping information,
and LCA matrices. The data package should be a zip file containing the following
files:

- `datapackage.json`: a JSON file describing the contents of the data package
- a `mapping` folder containing a `mapping.yaml` file that describes the mapping
  between the IAM scenario variables and the LCA datasets
- an `inventories` folder containing the LCA matrices as CSV files for each time step
- a `scenario_data` folder containing the scenario data as CSV files

`pathways` reads the scenario data files (1 in Figure 1), and iterates, 
for each time step and region, through technologies with a non-null
production volume. For each technology, `pathways` retrieves the corresponding
LCI dataset by looking it up in the mapping file (2 in Figure 1). The lookup
indicates `pathways` which LCA matrices to fetch from the data package (3 in Figure 1).
The LCA matrices are loaded in `bw2calc` (the LCA calculation module of `brightway`)
and multiplied by the production volume (see 4 in Figure 1). The results are aggregated 
and saved in a dataframe, where impacts are broken down per technology, region, 
time step, geographical origin of impact, life-cycle stage and impact assessment 
method (6 in Figure 1).

Some post-processing is done on the inventory matrices, including managing double counting. 
Double counting occurs when resource demands are counted multiple times across 
interconnected system components, inflating environmental impacts. This issue is 
particularly relevant when the reference scenario (e.g., from an IAM) already accounts
for total regional demand, such as electricity or transport. For example, if electricity and 
steel production are interdependent, evaluating total electricity demand as defined by 
the scenario may lead to overlap: electricity requires steel, and steel production, in turn, 
requires additional electricity beyond the initial total. This overlap results in duplicative 
demand estimates.

To address this, the original LCI database is adjusted by 
zeroing out all regional energy inputs that the energy system 
model accounts for and might demand during the system's life cycle,
following the same workflow presented in [@Volkart:2018] (see 5 in Figure 1). 
Practitioners are required to selectively cancel out overlapping activities already
accounted for by the scenario. We use a modular approach in this adjustment process, 
where practitioners, based on their understanding of the model generating the scenario, 
can select specific activity categories (e.g., electricity, heat, or specific product inputs) 
to exclude. For instance, if the IAM models regional electricity generation, the 
corresponding electricity inputs in the LCA system for upstream processes are 
removed to prevent double counting. Returning to the electricity-steel example, 
this means the practitioner would exclude electricity inputs for steel production 
within the LCA, as the scenario’s total electricity demand already covers this requirement.

This process is implemented in the `remove_double_accounting` function, which modifies the 
technosphere matrix to remove redundant entries. The function identifies flagged 
products for removal, locates the associated rows, and zeroes out the corresponding positions
taking any specified exceptions.For instance, in the electricity-steel example, the function
would find the row corresponding to regional electricity and cancel out the input in the column
associated with steel production, effectively preventing double counting of electricity demand.
This modular approach enhances transparency and traceability, making it easier to document and 
track which system components are modified, ensuring consistency between the scenario outputs and the LCA.

Finally, Global Sensitivity Analysis (GSA) can be performed on the results.
Currently, `pathways` supports the use of the `SALib` library for GSA [@Herman2017], 
[@Iwanaga2022], notably the Delta Moment-Independent Measure (DMIM) method [@BORGONOVO2007771], to rank
the influence of the database exchanges on the results.

![`pathways` workflow: from data package to impact assessment.\label{fig:workflow}](assets/workflow_diagram.png)

A detailed [example notebook](https://github.com/polca/pathways/blob/main/example/example.ipynb) 
is available for using `pathways` with a sample data package.

# Impact

By systematically updating and integrating LCA matrices over time, `pathways` improves the accuracy and relevance of 
environmental impact assessments for transition scenarios. This tool fosters greater alignment between LCAs and ESM/IAM 
outputs, enhancing the consistency and reliability of environmental assessments across different modelling platforms.

Additionally, `pathways` offers a detailed and structured workflow that enables IAM 
modellers to incorporate LCA into their analyses. This opens new avenues for these modellers to enhance the 
environmental dimension of their work.

Designed to be both reproducible and transparent, `pathways` facilitates collaboration and verification within the 
scientific community. This approach ensures that improvements in environmental impact assessments are accessible and 
beneficial to a broader range of stakeholders.


# Conclusion

`pathways` is a tool that evaluates the environmental impacts of transition 
scenarios over time using time-adjusted and scenario-based LCA matrices. This
approach allows for characterizing the environmental impacts of a scenario
across a wide range of indicators, including land use, water consumption,
toxicity impacts, etc. It also allows to attribute supply chain emissions
to the final energy carriers, thus providing a more detailed and transparent
view of the environmental impacts of a scenario.

# Acknowledgements

The authors gratefully acknowledge the financial support from the Swiss State 
Secretariat for Education, Research and Innovation (SERI), under the Horizon 
Europe project PRISMA (grant agreement no. 101081604). The authors also thank the
Swiss Federal Office of Energy (SFOE) for the support in the development of the 
`premise` and `pathways` tools through the SWEET-SURE program.

# References
