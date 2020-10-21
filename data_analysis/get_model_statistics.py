# -*- coding: utf-8 -*-

"""Script which can be used to compare the features obtained of two different influenza models

Usage:
  get_model_statistics.py <model> [--country=<country_name>] [--no-future] [--basedir=<directory>] [--start-year=<start_year>] [--end-year=<end_year>] [--save] [--no-graph]

  <baseline>      Data file of the first model
  <other_method>     Data file of the second model
  -h, --help        Print this help message
"""

import pandas as pd
import numpy as np
from scipy import stats
from docopt import docopt
import os
import glob

from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def get_results_filename(basepath):
    files = [f for f in glob.glob(basepath + "/*-prediction.csv", recursive=True)]
    y = os.path.basename(files[0]).split("-")[0]
    y2 = os.path.basename(files[0]).split("-")[1]
    return "{}-{}".format(y, y2)

if __name__ == "__main__":

    args = docopt(__doc__)

    model = args["<model>"]
    base_dir = args["--basedir"] if args["--basedir"] else "../complete_results"
    country = args["--country"] if args["--country"] else "italy"
    future = "no-future" if args["--no-future"] else "future"

    # Read the baseline results and merge them
    model_path = os.path.join(base_dir, args["<model>"], future, country)
    season_years = get_results_filename(model_path)
    model_file = os.path.join(model_path, "{}-prediction.csv".format(season_years))

    # Load the data
    data = pd.read_csv(model_file)

    # Get only the weeks we care for
    start_year = "2007-42" if not args["--start-year"] else args["--start-year"]
    end_year = "2019-15" if not args["--end-year"] else args["--end-year"]

    start_season = data["week"] >= start_year
    end_season = data["week"] <= str(int(end_year.split("-")[0]) + 1) + "-" + end_year.split("-")[1]
    total = start_season & end_season

    data = data[total]

    # Describe the data
    print("")
    print("[*] Describe the given dataset {}".format(model_file))
    print(data.describe())

    # Generate residuals
    print("")
    print("[*] Describe the residuals")
    residuals = data["incidence"]-data["prediction"]
    print(residuals.describe())

    # Get some statistics
    print("")
    total_pearson = 0
    for i in np.arange(0, len(data["prediction"]), 26):
        total_pearson += stats.pearsonr(data["prediction"][i:i+26], data["incidence"][i:i+26])[0]
    print("Pearson Correlation (value/p): ", total_pearson/(len(data["prediction"])/26))
    print("")

    print("Mean Squared Error: ", mean_squared_error(data["prediction"], data["incidence"]))
    print("")

    if not args["--no-graph"]:
        ax = sns.distplot(residuals, label="Residual")
        plt.figure()
        ax = sns.distplot(data["incidence"], label="Incidence")
        ax = sns.distplot(data["prediction"], label="Prediction")
        plt.legend()
        plt.show()
