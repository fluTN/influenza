#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine learning model which uses Wikipedia data to predicts ILI levels.

Usage:
  model.py <year_start> <year_end> <dataset_path> <incidence_path> <keywords_file> <country_name> [--p] [--ne] [--f] [--v] [--e] [--d=<directory>] [--no-future] [--no-images] [--no-month-year] [--standardize-week]

  <year_start>      The first influenza season we want to predict.
  <year_end>        The last influenza season we want to predict.
  <dataset_path>    The path to dataset files (the ones which contain Wikipedia pageviews).
  <incidence_path>  The path to the files which specify the ILI levels for each week.
  <keywords_file>   The path to the file which contains the Wikipedia pages used.
  <country_name>    The name of the country we are analyzing
  -p, --poisson     Use the Poisson model + LASSO instead of the linear one.
  -ne, --negative-bin   Use the Negative Binomial model + LASSO.
  -f, --file        Write informations to file
  -v, --verbose     Output more informations
  -e, --elastic     Use ElasticNet
  -d, --directory   Select a directory in which save your files
  -n, --no-future   Use a different method to train the model (avoid using seasonal influenza which are
                    later than the one we want to predict)
  --no-images       Do not generate any graphs. Print only to screen.
  --no-month-year   Do not add a month column.
  -h, --help        Print this help message
"""

import csv

from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import r2_score
from scipy import stats
from tabulate import tabulate
from docopt import docopt

import pandas as pd

from models_utils import *

try :
 from pyglmnet import GLMCV

except ImportError as e:
    print("I was not able to import GLM model. You will run into errors if used.")

# Try to import matplotlib (if we are running on a system without Tkinter
# then we will just skip it)
can_print = True
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    can_print = False
    print(e.message)
    print("The ability of generating graph is currently disabled.")


#########################
#### INITIALIZATION #####
#########################

# Parse the command line
arguments = docopt(__doc__)

# Directory where to save files
save_directory="./"
if arguments["--d"]:
    save_directory = arguments["--d"]

# Year selected
year_start = int(arguments["<year_start>"])
year_end = int(arguments["<year_end>"])+1

# Model type
if arguments["--p"] and arguments["--ne"]:
    print("[*] Cannot use both Poisson and Negative Binomial, please select just one of them.")

model_type = "Linear Model"
using_poisson = False
using_negbinomial = False
if arguments["--p"]:
    using_poisson = True
    model_type="Poisson Model"
elif arguments["--ne"]:
    using_negbinomial = True
    model_type="Negative Binomial Model"

using_elastic = False
if arguments["--e"]:
    using_elastic = True
    model_type = "ElasticNet Model"

# Feature path and labels
path_features = arguments["<dataset_path>"]
path_labels = arguments["<incidence_path>"]
keywords = arguments["<keywords_file>"]

# Selected columns that will be extracted from the dataframes
selected_columns = generate_keywords(keywords)+["week"]
if not arguments["--no-month-year"]:
    selected_columns = selected_columns + ["year", "month"]

all_features_values = pd.DataFrame()
all_predicted_values = []
all_true_labels = pd.DataFrame()
total_weeks = []
all_weighted_feature=dict()

# The pair start-end seasons
year_sel = (year_start, year_end)

# Useful information which can be written to file
peaks = []
predicted_peaks = []
peaks_value = []
predicted_peaks_value = []
mse_seasons = []

# If --f then we write some information to file
if arguments["--f"]:
    csvfile = open(save_directory+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"_information_"+arguments["<country_name>"]+".csv", 'w', newline='')
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['season', 'country', 'predicted_influenza_peak', 'real_influenza_peak', 'predicted_influenza_peak_value', 'real_influenza_peak_value', 'mse', 'pcc', 'p_value'])

    top_features_file = open(save_directory+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"_features_"+arguments["<country_name>"]+".csv", 'w', newline='')
    top_features_writer = csv.writer(top_features_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    top_features_writer.writerow(['season', 'country', 'page_name', 'value'])

######################
##### ALGORITHM ######
######################

# For each season, train a model with the complete dataset (without the season)
# and predict the ILI incidence.
for year_selected in range(year_sel[0], year_sel[1]):

    print("[*] ", year_selected)

    # Generate the list of excluded years
    excluded=[]
    if arguments["--no-future"]:
        for i in range(2007, 2020):
            if i>year_selected:
                excluded.append(i)

    # Data generation from data files
    dataset = generate(year_selected, excluded, path_features)
    labels = generate_labels(year_selected, excluded, path_labels)
    data = generate_one_year(year_selected, path_features)
    labels_test = generate_labels_one_year(year_selected, path_labels)
    weeks = labels_test["week"]

    # Add year and column week
    dataset["year"], dataset["week"] = dataset["Week"].str.split('-', 1).str
    data["year"], data["week"] = data["Week"].str.split('-', 1).str
    dataset["week"] = dataset["week"].apply(pd.to_numeric)
    dataset["year"] = dataset["year"].apply(pd.to_numeric)
    data["week"] = data["week"].apply(pd.to_numeric)
    data["year"] = data["year"].apply(pd.to_numeric)

    # Create a copy of the dataset where we fills NaN values
    # with 0.
    dataset_zero = dataset.copy()
    data_zero = data.copy()

    # Split year week into two elements and generate month column
    if not arguments["--no-month-year"]:
        dataset_zero = add_month(dataset_zero)
        data_zero = add_month(data_zero)
    else:
        dataset_zero.drop(["year"], axis=1)
        data_zero.drop(["year"], axis=1)

    one_hot_weeks_train = pd.get_dummies(dataset_zero["week"])
    one_hot_weeks_test = pd.get_dummies(data_zero["week"])

    # Standardize data
    if not arguments["--standardize-week"]:
        train, test = standardize_data(dataset_zero[selected_columns], data_zero[selected_columns])
    else:
        columns_without_week = selected_columns.copy()
        columns_without_week.remove("week")
        train, test = standardize_week(dataset_zero[selected_columns], data_zero[selected_columns], columns_without_week)

    # Drop week information and add dummies instead

    train = train.drop(["week"], axis=1)
    test = test.drop(["week"], axis=1)
    train = pd.concat([train,one_hot_weeks_train], axis=1)
    test = test.join(one_hot_weeks_test)

    # Create a Lasso Cross-Validation instance which will be
    # trained on the dataset in which NaN values are replaced
    # with the column mean.
    if not using_poisson and not using_negbinomial:
        if using_elastic:
            l1_values = np.linspace(0,1,10)
            alphas = np.linspace(0,0.5,10)
            model = ElasticNetCV(l1_ratio=l1_values, n_jobs=-1, random_state=1, max_iter=100000, alphas=alphas, selection="random", cv=5, normalize=True)
        else:
            model = LassoCV(max_iter=100000, n_jobs=-1, n_alphas=1000, random_state=1, cv=5)
        model.fit(train, labels["incidence"].fillna(0))
        result = model.predict(test)

        if using_elastic:
            print("Best L1_ratio: {}".format(model.l1_ratio_))

        print("Best Alpha: {}".format(model.alpha_))
    elif using_negbinomial:
        model = GLMCV(distr='neg-binomial', score_metric="pseudo_R2", cv=3)
        model.fit(train.values.copy(), labels["incidence"].fillna(0).copy().values * 100000)
        result = model.predict(test.values)
    elif using_poisson:
        # Convert features into count based values
        train_converted = train.values.copy()
        incidence_converted = labels["incidence"].fillna(1).copy().values * 100000
        model = GLMCV(distr='poisson', max_iter=5000, cv=3)
        model.fit(train_converted, incidence_converted)
        result = model.predict(test.values)

    # Get the feature coefficients
    if not using_poisson and not using_negbinomial:
        if not using_elastic:
            coeff = model.coef_
        else:
            coeff = model.beta_
    elif using_negbinomial:
        coeff = model.beta_
    elif using_poisson:
        coeff = model.beta_

    # Add the pair (value, coeff) to a dictionary
    current_weighted_features = dict()
    for i in list(zip(coeff, selected_columns)):
        if (i[0] != 0):
            if all_weighted_feature.get(i[1], None) is None:
                all_weighted_feature[i[1]] = [i[0]]
            else:
                all_weighted_feature[i[1]].append(i[0])
            if current_weighted_features.get(i[1], None) is None:
                current_weighted_features[i[1]] = [i[0]]
            else:
                current_weighted_features[i[1]].append(i[0])

    # Add the value to global variables
    all_features_values= all_features_values.append(data_zero)
    all_predicted_values.extend(result)
    all_true_labels = all_true_labels.append(labels_test)
    total_weeks.extend(weeks)

    mse_seasons.append(mean_squared_error(labels_test['incidence'].fillna(0), result))

    # If we are verbose print more informations
    index_peak = labels_test['incidence'].fillna(0).idxmax(axis=1)
    results = pd.DataFrame(result, index=labels_test.index.values, columns=["incidence"])
    index_peak_m = results["incidence"].idxmax(axis=1)
    peaks.append(labels_test["week"][index_peak])
    predicted_peaks.append(labels_test["week"][index_peak_m])
    peaks_value.append(labels_test["incidence"][index_peak])
    predicted_peaks_value.append(results["incidence"][index_peak_m])
    if arguments["--v"]:
        print("[*] Influenza Season Peak (Week): ", labels_test.iloc[index_peak]["week"])
        print("[*] Influenza Seasons Predicted Peak (Week): ", labels_test.iloc[index_peak_m]["week"])
        print("[*] Influenza Season Peak Value: ", labels_test.iloc[index_peak]["incidence"])
        print("[*] Influenza Season Predicted Peak Value: ", result[index_peak_m])
        print("------------")

    # For each influenza seasons we print data to file
    if arguments["--f"]:
        spamwriter.writerow([str(year_selected-1)+"-"+str(year_selected),
                             arguments["<country_name>"],
                             labels_test["week"][index_peak_m],
                             labels_test["week"][index_peak],
                             results["incidence"][index_peak_m],
                             labels_test["incidence"][index_peak],
                             mean_squared_error(labels_test['incidence'].fillna(0), result),
                             stats.pearsonr(result, labels_test['incidence'].fillna(0))[0],
                             stats.pearsonr(result, labels_test['incidence'].fillna(0))[1]
                             ])
        for p in get_important_pages(current_weighted_features, len(current_weighted_features)):
            top_features_writer.writerow([str(year_selected-1)+"-"+str(year_selected),
                                          arguments["<country_name>"],
                                          p[0],
                                          float(p[1])])


# Generate a dataframe with all the important informations
complete_data = all_true_labels.copy()
complete_data["prediction"] = pd.Series(all_predicted_values, index=complete_data.index)
complete_data.to_csv(save_directory+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"-prediction.csv", index=False)

# Get important pages to generate the plot we need
important_pages = get_important_pages(all_weighted_feature, 10)

# Save all important features to file
if arguments["--f"]:
    imp_pages_keys = list(dict(important_pages).keys())
    imp_pages_values = stz(all_features_values[imp_pages_keys])
    imp_pages_values= imp_pages_values.reset_index()
    imp_pages_values= imp_pages_values.reindex(index=range(0, len(weeks)))
    weeks=weeks.reset_index()
    weeks=weeks.reindex(index=range(0, len(weeks)))
    output_name = save_directory+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"_most_important_features_"+arguments["<country_name>"]+".csv"
    result_dataframe = pd.concat([weeks, imp_pages_values], axis=1).drop(["index"], axis=1)
    result_dataframe.to_csv(output_name, index=False)

# Print MSE of the two models
mse = mean_squared_error(all_true_labels["incidence"].fillna(0), all_predicted_values)
print("------------")
print("MSE: ", mse)
print("------------")

# Print R squared
r2 = r2_score(all_true_labels["incidence"].fillna(0), all_predicted_values)
print("------------")
print("R^2: ", r2)
print("------------")

# Print Pearson Coefficient
pcc_p = stats.pearsonr(all_predicted_values, all_true_labels["incidence"].fillna(0))
pcc = pcc_p[0] # pearson value
p_value = pcc_p[1] # p value
print("------------")
print("Pearson Correlation Coeff: {} (p={})".format(pcc, p_value))
print("------------")

# Print important pages
print(tabulate(important_pages, headers=["Page name", "Weight"]))

###########################
#### GRAPHS GENERATION ####
###########################

if not arguments["--no-images"] and can_print:
    # Plot some informations
    fig = plt.figure(3, figsize=(15, 6))
    ax = fig.add_subplot(111)

    # Set up the graph title, x and y axis titles
    plt.title("Influenza Seasons "+str(year_sel[0]-1)+" - "+str(year_sel[1]-1), fontsize=18)
    plt.ylabel("Incidence on 100000 people", fontsize=17)
    plt.xlabel("Year-Week", fontsize=17)

    # Generate the x axis labels in such way to
    # have the year-week pair only every two ticks.
    weeks_used = []
    for k, v in enumerate(total_weeks):
        if k%5==0:
            weeks_used.append(v)
        #else:
        #    weeks_used.append(" ")

    # Set up the axes ticks
    plt.xticks(range(0, len(total_weeks), 5), weeks_used, rotation="vertical", fontsize=15)
    plt.yticks(fontsize=15)

    # Plot the model result and the incidence
    plt.plot(range(0, len(all_predicted_values)), all_predicted_values, 'r-', label=model_type, linewidth=3)
    plt.plot(range(0, len(all_true_labels["incidence"])), all_true_labels["incidence"].fillna(0), 'k-', label="Incidence", linewidth=3)

    # Add dotted line into the graph to delimit the influenza seasons.
    for i in range(1, (year_sel[1]-year_sel[0])):
        plt.axvline(26*i, linestyle="dashed", color="k", linewidth=3)

    # Update some graphical options
    plt.grid()
    plt.legend(fontsize=16, loc="upper right")
    plt.tight_layout()

    # Save the graph on file
    plt.savefig(save_directory+"season_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+".png", dpi=150)

    # Feature plot
    figf = plt.figure(4, figsize=(15, 6))
    axf = figf.add_subplot(111)

    if arguments["--f"]:
        # Generate the graph showing the pageview variation of the top 5 Wikipedia's pages
        # choosen by the model.
        plt.title("Pageview variation "+str(year_sel[0]-1)+" - "+str(year_sel[1]-1), fontsize=18)
        plt.ylabel("Pageview value", fontsize=17)
        plt.xlabel("Year-Week", fontsize=17)

        # Plot the axes labels
        plt.xticks(range(0, len(total_weeks), 5), weeks_used, rotation="vertical", fontsize=15)
        plt.yticks(fontsize=15)

        # Plot all the pageview data and the ILI incidence
        std_all_features_values = stz(all_features_values[selected_columns])
        for key, value in important_pages:
         plt.plot(range(0, len(std_all_features_values[key])), std_all_features_values[key], '-', label=key, linewidth=3)
        plt.plot(range(0, len(all_true_labels["incidence"])), stz(all_true_labels["incidence"]), 'k-', label="Incidence", linewidth=3)

        # Add dotted line into the graph to delimit the influenza seasons.
        for i in range(1, (year_sel[1]-year_sel[0])):
            plt.axvline(26*i, linestyle="dashed", color="k", linewidth=3)

        # Update some graphical options
        plt.grid()
        plt.legend(fontsize=16, loc="best")
        plt.tight_layout()

        # Save again the graph on file
        plt.savefig(save_directory+"season_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+"_features.png", dpi=150)

        # Correlation Matrix
        new_pages=[]
        for p in important_pages:
            new_pages.append(p[0])
        first = pd.DataFrame(all_features_values[new_pages])
        second = pd.DataFrame(all_true_labels["incidence"]).set_index(first.index.values)
        first['incidence'] = second
        first = first.loc[:, (first != 0).any(axis=0)]
        new_pages = list(first.columns.values)
        correlation_matrix(first, "Correlation Matrix "+str(year_sel[0]-1)+"-"+str(year_sel[1]-1), new_pages, save_directory+"cmatrix_"+str(year_sel[0]-1)+"-"+str(year_sel[1]-1)+".png")
