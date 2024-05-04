---
title: '`pathways`: enhancing environmental impact assessments of transition scenarios through Life Cycle Assessment (LCA)'
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
 - name: Paul Scherrer Institute, Laboratory for Energy Systems Analysis, 5232 Villigen, Switzerland
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

Most IAMs and ESMs project future energy supply optimized for cost under a given 
greenhouse gas emissions trajectory. These scenarios outline changes 
required in regional energy mixes to achieve global warming mitigation goals 
[@Riahi:2017]. By analyzing these scenarios, we can assess how future system 
changes will affect the environmental performance of various technologies across 
supply chains. 

Prospective LCA (pLCA) emerges as a valuable tool for evaluating 
the environmental performance of both existing and emerging production systems.
The body of literature applying scenario-based pLCA to emerging technologies has 
flourished in the past decade -- see literature review of [@Bisinella:2021].

Extending present-day process-based life-cycle inventories into the future using 
IAM outputs lays the methodological groundwork for pLCA. Such approach was 
initially started with the work of [@MendozaBeltran:2018], and more recently 
formalized with the Python library `premise` [@Sacchi:2022]. 

However, efforts in pLCA have primarily focused on improving the accuracy of 
forecasting future life cycle inventories. Performing scenario-wide LCAs
with life cycle inventories adjusted to each time step of the scenario has 
significant potential to enhance sustainability assessments. This approach broadens 
the focus beyond greenhouse gas emissions to encompass broader environmental 
impacts like land use, water consumption, and toxicity, accounting for both 
direct and indirect emissions. Nonetheless, conducting system-wide LCA remains 
challenging due to computational costs and methodological complexities, such as 
defining the functional unit based on IAM outputs and addressing issues like 
double-counting. 

Several studies have attempted to address the challenges of coupling
ESM/IAM with LCA, with notable contributions from [@Gibon:2015], [@Rauner:2017] and
[@Pehl:2017], who quantified the outputs of an ESM or IAM scenario, 
with a hybrid-LCA framework. The comprehensive and ambitious framework EAFESA 
developed by Xu and colleagues [@Xu:2020], which aimed at a bidirectional coupling
between ESM and LCA is also worth mentioning. However, these studies have
focused on specific sectors or technologies, and have not yet been generalized
to a broader range of scenarios and indicators. Also, to the authors' knowledge,
their implementation has not been made available to the broader scientific community.

To tackle these challenges, the open-source library `pathways` leverages the 
LCA framework `brightway2` [@Mutel:2017] and offers a systematic tool for 
evaluating the environmental impacts of energy transition scenarios. `pathways` is
designed to work with data packages containing LCA matrices which have been
adjusted to each time step of the ESM/IAM scenario. The library calculates the 
environmental impacts of the scenario (or a subset of it) over time, 
providing a more detailed and transparent view of the environmental impacts implied
by the scenario.

# Description

1. What pathways does

![Workflow for characterizing the environmental impacts of transition scenarios using `pathways`.\label{fig:workflow}](assets/diagram_1.png)


2. Figure of the workflow

[@Sacchi:2022]

# Usage

# Impact



# Conclusion

1. pathways offers a tool to systematically evaluate the environmental impacts of transition scenarios, considering the 
   full supply chain of products and services in a dynamic way where the results of the scenario are integrated within the LCA database at each timestep

# Acknowledgements

The authors gratefully acknowledge the financial support from the Swiss State Secretariat for Education, Research and 
Innovation (SERI), under the Horizon Europe project PRISMA (grant agreement no. 101081604). The authors also thank the
Swiss Federal Office of Energy (SFOE) for the support in the development of the `premise` and `pathways` tools through 
the SWEET-SURE program.

# References
