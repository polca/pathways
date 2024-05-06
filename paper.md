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
    affiliation: 1

affiliations:
 - name: Laboratory for Energy Systems Analysis, Paul Scherrer Institute, 5232 Villigen, Switzerland
   index: 1

date: 03 May 2024
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

Most IAMs and ESMs project cost-optimized future energy supplies within 
specified greenhouse gas emissions trajectories, outlining changes needed 
in regional energy mixes for global warming mitigation [@Riahi:2017]. 
Prospective Life Cycle Assessment (pLCA) is crucial for evaluating the 
environmental performance of existing and emerging production systems, with 
a growing body of literature in scenario-based pLCA for emerging technologies 
[@Bisinella:2021].

Extending present-day life-cycle inventories into the future using IAM outputs, 
initially explored by [@MendozaBeltran:2018] and formalized by the Python library 
`premise` [@Sacchi:2022], forms the methodological basis for pLCA. Efforts in pLCA 
focus on improving forecasting accuracy. Performing scenario-wide LCAs with 
adjusted life cycle inventories at each time step has potential to enhance 
sustainability assessments, broadening focus beyond greenhouse gas emissions 
to include broader environmental impacts like land use, water consumption, 
and toxicity, addressing both direct and indirect emissions. However, system-wide 
LCA remains challenging due to computational costs and methodological 
complexities, such as defining functional units based on IAM outputs and 
resolving double-counting issues.

Several studies characterize energy scenarios with LCA, including 
[@Gibon:2015], [@Rauner:2017] and [@Pehl:2017], who quantified ESM or 
IAM scenario outputs using a hybrid-LCA framework. There is also the work of
[@Xu:2020], who developed the ambitious EAFESA framework aiming for 
bidirectional coupling between ESM and LCA. Yet, these studies focused 
on specific sectors or technologies and haven't yet generalized to broader 
scenarios and indicators, nor made their implementations widely available.

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
and multiplied by the production volume (see 4 in Figure 1). Some post-processing
is done on the inventory matrices (e.g., Monte Carlo iterations, dealing with
double accounting, etc., see 5 in Figure 1) before the results are aggregated and saved in a
dataframe (6 in Figure 1). Impacts are broken down per technology, region, time step,
geographical origin of impact, life-cycle stage and impact assessment method.

![`pathways` workflow: from data package to impact assessment.\label{fig:workflow}](assets/workflow_diagram.png)

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
