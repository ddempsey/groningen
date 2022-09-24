# Groningen
Induced seismicity forecast model for Groningen gas field (the Netherlands). This model implements a poroelastic loading model of induced seismicity triggering, with spatiotemporal history matching and probabilistic forecasting.

## Installation

Ensure you have Anaconda Python 3.8 installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/groningen
```

2. CD into the repo and create a conda environment

```bash
cd groningen

conda env create -f environment.yml

conda activate groningen
```

The installation has been tested on Windows operating system. Total install with Anaconda Python should be less than 10 minutes.

## Running models
All code is contained in the script ```groningen_v2.py```. There are three main functions that can be activated by uncommenting at the bottom of the file.

The first, ```test()```, performs a pseudo-prospective evaluation of the updated forecast model. It constructs the rate model, performs L- and N-tests, and displays these as a plot to the screen.

The second, ```global_optimization()```, implements the genetic algorithm that searches for minimum misfit models.

The third, ```prospective_forecast()```, constructs the revised forecast model for Groningen using a tapered and unbounded GR distribution.

To run the models, open ```groningen_v2.py```, comment/uncomment the functions you want to run, then in a terminal type
```bash
cd scripts

python groningen_v2.py
```

## Disclaimers
1. This forecast model is predicated on certain assumptions and operation of the field in the future. Variance from these may materially affect the forecast outputs. In our paper, we discuss the conditions under which the forecast model is likely to perform poorly.

2. This software is not guaranteed to be free of bugs or errors. Most codes have a few errors and, provided these are minor, they will have only a marginal effect on accuracy and performance. This is especially true when forecast models are subject to history matching, as this one is. That being said, if you discover a bug or error, please report this at [https://github.com/ddempsey/groningen/issues](https://github.com/ddempsey/groningen/issues).

## Acknowledgments
Support and data that enabled this work were provided by Nederlandse Aardolie Maatschappij (NAM).
