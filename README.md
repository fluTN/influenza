# A general method for estimating the prevalence ofInfluenza-Like-Symptoms with Wikipedia data

This repository contains the supporting code for the paper **"A general method for estimating the prevalence ofInfluenza-Like-Symptoms with Wikipedia data"**.

## Repository structure

The repo was organized following this directory structure:
* `data`: it contains the actual data used to train and test the models. You can
find Wikipedia parsed page view logs, the list of Wikipedia's pages used and the
real influenza incidence data taken from official italian, german, belgian and dutch
surveillance systems;
* `data_analysis`: it contains scripts used to parse and analyze Wikipedia page view
data. These script were used to generate the datasets inside the `data` directory. Moreover,
it contains several scripts to parse the model results to obtain the graph used in the
paper;
* `models`: it contains the actual machine learning models;
* `pagecounts`: it contains the code used to download and store the complete page
view logs.
