---
title: 'MetaCast: A package for broad**CAST**ing epidemiological and ecological models over **META**-populations.'
tags:
  - Python
  - Epimediology
  - Ecology
  - Virology
  - Compartmental
  - Model
  - Metapopulations
  - Structured Populations
  - Discrete Event
  - Sensitivity Analyses
  - Latin Hypercube Sampling
  - Scenario Analysis
authors:
  - name: Martin Grunnill
    corresponding: true 
    orcid: 0000-0002-4835-8007
    affiliation: 1
  - name: Julien Arino
    orcid: 0000-0001-6409-5027
    affiliation: 2
  - name: Abbas Ghasemi
    orcid: 0000-0003-2442-5962
    affiliation: 3
  - name: Edward W. Thommes
    orcid: 0000-0001-6800-2000
    affiliation: "4, 5"
  - name:  Jianhong Wu
    affiliation: "1, 6"
affiliations:
 - name: Laboratory of Industrial and Applied Mathematics (LIAM), York University, Toronto, Ontario, Canada
   index: 1
 - name: Department of Mathematics, University of Manitoba, Winnipeg, Manitoba, Canada
   index: 2
 - name: Mechanical Industrial and Mechatronics Engineering Department, Toronto Metropolitan University (TMU), Toronto, Ontario, Canada
   index: 3
 - name: New Products and Innovation (NPI), Sanofi, Toronto, Ontario, Canada
   index: 4
 - name: Department of Mathematics and Statistics, University of Guelph, Guelph, Ontario, Canada
   index: 5
 - name: York Emergency Mitigation, Response, Engagement and Governance Institute, York University, Toronto, Ontario, Canada
   index: 6

date: 1 April 2024
bibliography: paper.bib

---

# Summary

`MetaCast` is a Python package for broadcasting epidemiological and ecological Ordinary Differential Equation (ODE)
based models over metapopulations (structured populations). Users define a function describing the
subpopulation model. `MetaCast`'s `MetaCaster` then broadcasts the subpopulation model function over dimensions
of metapopulations. These dimensions can be defined and redefined flexibly, allowing for comparisons
of multidimensional metapopulation models that can have migration (flows) of populations between
subpopulations. In addition to the metapopulation suite `MetaCast` has several features. A
multinomial seeder allows users to randomly select infected stages to place an infected
population in, based on the occupancy time of infected states. `MetaCast`'s event queue suite
can handle discrete events within simulations, such as movement of populations between compartments and changes in 
parameter values. Sensitivity analysis can be done in `MetaCast` using parallelisable Latin Hypercube Sampling and 
Partial Rank Correlation Coefficient functions. All of this makes MetaCast an ideal package not only for modelling 
metapopulations but for wider scenario analysis.

# Statement of Need

`MetaCast` was developed from the code base used in a project modelling the spread of 
COVID-19 at Mass Gathering Events (MGEs), such as the FIFA 2022 World Cup [@Grunnill2024]. 
During this project there were a number of MGEs that we considered as potential case studies
before settling on the FIFA 2022 World Cup. As such, even though our epidemiological model 
was remaining much the same, the resulting change in metapopulation structure between potential case study models
meant we had to extensively recode the model. In order to expedite this recoding due to changes in metapopulation
structure we developed the code in @Grunnill2024. This code allowed us to broadcast our COVID-19 subpopulation model
over different two-dimensional metapopulations (based on clusters of people and their vaccination status [@Grunnill2024]), 
whilst calculating the force of infections for all subpopulations [@Keeling2008c]. 
`MetaCast` builds upon and improves the code used in @Grunnill2024. These improvements mean that metapopulations are no 
longer limited to two dimensions. Furthermore, metapopulation dimensions don't have to be based on clusters of people and their 
vaccination status. `MetaCast` also includes more user-friendly versions of the discrete event, sensitivity analyses and
infectious population seeding features from @Grunnill2024. Making MetaCast an ideal package for scenario analyses
based around metapopulation models within epidemiology or ecology.

# State of Field

There are a number of packages that can be used for epidemiological or ecological modelling accross a number of 
platforms including Python. However, to our knowledge, none bring together all the features for scenario analyses based 
around ODE metapopulation models as described above.

#### Ordinary Differential Equation (ODE) Modelling Packages for Epidemiology and Ecology
R's `EpiMode` [@Jenness2018] has some pre-coded epidemiological ODE models (such as SIR and SIS), as does
 the Python package `Eir` [@Jacob2021]. `EpiMode` [@Jenness2018] can also perform sensitivity analyses on these pre-coded 
models. `PyGOM` [@Tye2018] and `Epipack` [@Maier2021] are Python packages that can produce ODE models from 
a list of transitions defining the flow between epidemiological compartments. Both `PyGOM` [@Tye2018] and `Epipack` [@Maier2021] can 
then simulate the ODE models deterministically or stochastically, with `PyGOM` having some extra stochastic methods. 
`PyGOM` [@Tye2018] also has a suite of maximum likelihood based and Approximate Bayesian Computation fitting procedures.

#### Individual Based Modelling (IBM) Packages for Epidemiology and Ecology
Python's `Epipack` [@Maier2021] has modules for defining transitions between states for nodes in network 
modelling. The Python open-source package `Covasim` (COVID-19 Agent-based Simulator) [@Kerr2021] provides detailed 
demographic data tailored to specific countries, encompassing age distributions and population sizes, offering 
sophisticated transmission networks for various social settings (households, schools, workplaces, long-term care 
facilities, etc). It also incorporates age-specific disease outcomes, viral load dynamics, and a wide array of 
intervention strategies. R's `EpiMode` [@Jenness2018] implements agent based modelling based around contacts as 
discrete events or as a static network model. There are a number of other R epidemiological IBM packages that take 
spatial or network contact based approaches: `individual` [@Charles2021], `hybridModels` [@Marques2020] and `EpiILMCT` 
[@Almutiry2021]. The Python package `Eir` [@Jacob2021] provides epidemiological models that incorporate the movements of people. 
`Pathogen.jl` [@Angevaare2022] is a Julia package for continuous time simulation and inference of transmission network 
individual level models (TN-ILMs). 

# Acknowledgements and Funding

The authors of this manuscript and of the package `MetaCast` would like to thank the funders who made this possible:

* Martin Grunnill's position was funded through the Fields Institute’s Mathematics for Public Health Next Generation program 
[http://www.fields.utoronto.ca/activities/public-health](http://www.fields.utoronto.ca/activities/public-health), grant 
number 72062654. 

* Julian Arino is funded through the Discovery Grant program from the Natural Science and Engineering Research Council 
of Canada (NSERC, [https://www.nserc-crsng.gc.ca/index_eng.asp](https://www.nserc-crsng.gc.ca/index_eng.asp)), grant 
number RGPIN-2017-05466. 

* Jianhong Wu's work is supported by the ADERSIM (Ontario Research Fund 33270), along with the Canada Research Chairs 
program ([https://www.chairs-chaires.gc.ca/home-accueil-eng.aspx](https://www.chairs-chaires.gc.ca/home-accueil-eng.aspx)
, 230720), and the Discovery Grant program from NSERC (105588).

* This work is supported by the NSERC- Sanofi Alliance program in Vaccine Mathematics, Modelling, and Manufacturing (517504). 
 
The funders had no role in the design, decision to publish, or preparation of the manuscript or the package `MetaCast`.

# References