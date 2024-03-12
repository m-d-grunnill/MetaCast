# MetaCast


A package for broad**CAST**ing epidemiological and ecological models over **META**-populations.

## Summary

`MetaCast` is a python package for broadcasting epidemiological and ecological ODE based models
over metapopulations (structured populations). Users first define a function describing the
subpopulation model. `MetaCast`'s users then define the dimensions of metapopulations that this
subpopulation is broadcast over. These dimensions can be flexibly defined allowing for multiple
dimensions and migration (flows) of populations between subpopulations. In addition to the 
metapopulation suite `MetaCast` has several features. A multinomial seeder allows users to randomly
 select infected stages to place an infected population in based on the occupancy time of infected
states. `MetaCast`'s event queue suite can handle discrete events within simulations, such as 
movement of populations between compartments and changes in parameter values. Sensitivity 
analysis can be done in `MetaCast` using parallelisable Latin Hypercube Sampling and Partial Rank Correlation Coefficient
functions. All of this makes MetaCast an ideal package not only for modelling metapopulations but
for wider scenario analysis.

## Installation
### Requirements

Python 3.10 and pip.
Package requirements:
* numpy >= 1.26.3
* pandas >= 2.1.4
* scipy >= 1.11.4
* pingouin >= 0.5.4
* tqdm >= 4.66.1
* dask >= 2024.2.1
* distributed >= 2024.2.1

For running [demonstration jupyter notebooks](https://github.com/m-d-grunnill/MetaCast/tree/main/demonstrations)
* bokeh >= 3.3.4
* seaborn >= 0.13.2
* jupyter >= 1.0.0

### Installing via pip
**Note** this should also install required packages.
```
pip install metacast
```

## Usage
See jupyter notebooks in demonstration directory of homepage:
[https://github.com/m-d-grunnill/MetaCast/tree/main/demonstrations](https://github.com/m-d-grunnill/MetaCast/tree/main/demonstrations)

## Documentation
[https://metacast.readthedocs.io/en/latest/index.html](https://metacast.readthedocs.io/en/latest/index.html)
