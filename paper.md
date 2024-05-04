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

IAMs, frequently based on Shared Socioeconomic Pathways (SSPs), offer cost-optimized projections of future scenarios. 
These scenarios highlight, for example, the necessary changes in regional electricity mixes and different means of 
transport to meet global warming mitigation objectives [@Riahi:2017]. This scenario analysis exercise enables us to 
consider future system changes and their effects on the environmental performance of different technologies 
along the different supply chains. In this context, prospective Life Cycle Assessment (pLCA) emerges as a unique tool 
to provide a robust evaluation of the environmental performance of both existing and anticipated production systems. 
At the methodological level, [@Sacchi:2022] has recently laid the foundations for extending present-day process-based 
life-cycle inventory into the future using the output from IAMs. Meanwhile, most efforts in pLCA have been centred 
around improving the ability to forecast future life cycle inventories accurately.

At this juncture, performing an LCA of the transition scenarios using the updated life cycle inventories at each time step
uncaps excellent potential to improve the sustainability assessment of these scenarios. LCA would expand the 
conventional focus on GHG emissions to broader environmental impacts, such as land use, water consumption, and toxicity 
while considering direct and indirect emissions. However, running LCAs of the transition scenarios provided by IAMs -
or energy system models - at the system level remains challenging. Mainly because of the computational expense of running 
LCAs for each time step and region of each scenario and the methodological complexity of consistently defining the 
functional unit of the LCA based on the IAMs outputs while dealing with issues such as double-counting. 
`pathways`, using the LCA framework `brightway2` [@Mutel:2017] and building on `premise`, offers a solution to these 
challenges by providing a tool to evaluate the environmental impacts of transition scenarios systematically.

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
