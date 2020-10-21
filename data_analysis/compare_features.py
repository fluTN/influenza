# -*- coding: utf-8 -*-

"""Script which can be used to compare the features obtained of two different influenza models

Usage:
  compare_models.py <baseline> <other_method>... [--country=<country_name>] [--no-future] [--basedir=<directory>] [--start-year=<start_year>] [--end-year=<end_year>] [--save] [--no-graph] [--top-k=<top_features>]

  <baseline>      Data file of the first model
  <other_method>     Data file of the second model
  -h, --help        Print this help message
"""

import os
import glob

from docopt import docopt
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("paper")

from models.models_utils import generate, stz_zero, get_important_pages

def correct_name(value):
    if value == "new_data" or value == "old_data":
        return "Categories (Pageviews+Pagecounts)"
    elif value == "cyclerank":
        return "CycleRank (Pageviews+Pagecounts)"
    elif value == "pageviews":
        return "Categories (Pageviews)"
    elif value == "cyclerank_pageviews":
        return "CycleRank (Pageviews)"
    elif value == "pagerank":
        return "PageRank (Pageviews+Pagecounts)"
    else:
        return "PageRank (Pageviews)"

def get_results_filename(basepath, country):
    files = [f for f in glob.glob(basepath + "/*_information_{}.csv".format(country), recursive=True)]
    season_years = os.path.basename(files[0]).split("_")[0]
    return season_years

def generate_dictionary(f, model):
    result = dict()

    unique_years = results.season.unique()
    for y in unique_years:
        f_tmp = f[f.season == y]
        for index, row in f_tmp.iterrows():
            page_name = str(row["page_name_"+model])
            weigth = float(row["value_"+model])

            if page_name in result:
                result[page_name].append(weigth)
            else:
                result[page_name] = [weigth]
    return result

if __name__ == "__main__":

    # Read the command line arguments
    args = docopt(__doc__)

    # Read some config variables
    base_dir = args["--basedir"] if args["--basedir"] else "../complete_results"
    country = args["--country"] if args["--country"] else "italy"
    future = "no-future" if args["--no-future"] else "future"
    top_features = int(args["--top-k"]) if args["--top-k"] else 5

    # Get keywords coming from the various methods
    print("")
    keywords_standard = pd.read_csv(os.path.join("../data/keywords", "keywords_{}.txt".format(country)), header=None, names=["page_name"])
    print("Standard keywords Size: {}".format(len(keywords_standard)))

    # Get keywords coming from the various methods
    keywords_cyclerank = pd.read_csv(os.path.join("../data/keywords", "keywords_cyclerank_{}.txt".format(country)), header=None, names=["page_name"])
    print("Cyclerank keywords Size: {}".format(len(keywords_cyclerank)))

    common_keywords = set.intersection(set(keywords_standard.page_name), set(keywords_cyclerank.page_name))
    print("Common keywords Size: {}, {}, {}".format(len(common_keywords), len(common_keywords)/len(keywords_standard), len(common_keywords)/len(keywords_cyclerank)))
    print("")

    # Read the baseline results and merge them
    baseline_results_path= os.path.join(base_dir, args["<baseline>"], future, country)

    season_years = get_results_filename(baseline_results_path, country)
    season_years_baseline = season_years
    baseline_result_file = os.path.join(baseline_results_path, "{}_features_{}.csv".format(season_years, country))
    baseline_results_df = pd.read_csv(baseline_result_file)[["season", "page_name", "value"]].rename(columns={"page_name": "page_name_{}".format(args["<baseline>"]), "value":"value_{}".format(args["<baseline>"])})

    # Concat all the other results
    other_results_df = None
    for other_results in args["<other_method>"]:
        other_results_path = os.path.join(base_dir, other_results, future, country)
        season_years = get_results_filename(other_results_path, country)
        other_result_file = os.path.join(other_results_path, "{}_features_{}.csv".format(season_years, country))

        if other_results_df is None:
            other_results_df = pd.read_csv(other_result_file)[["season", "page_name", "value"]]
            other_results_df = other_results_df.rename(columns={"page_name": "page_name_{}".format(other_results), "value":"value_{}".format(other_results)})
        else:
            current_other_results_df = pd.read_csv(other_result_file)[["season", "page_name", "value"]]
            current_other_results_df = current_other_results_df.rename(columns={"page_name": "page_name_{}".format(other_results), "value":"value_{}".format(other_results)})
            other_results_df = pd.concat([other_results_df, current_other_results_df], axis=1)

    # Total results
    results = pd.concat([baseline_results_df, other_results_df.drop(["season"], axis=1)], axis=1)

    # Count uniques years
    unique_years = results.season.unique()

    # Compute how many features are different from zero
    for m in args["<other_method>"]:
        counter = 0
        max = (-1, 0)
        min = (100000000, 0)
        for i in unique_years:
            total_features_used = other_results_df[other_results_df.season == i]["page_name_{}".format(m)].count()
            if total_features_used > max[0]:
                max = (total_features_used, i)
            else:
                if total_features_used < min[0]:
                    min = (total_features_used,i)
            #print(i, m, total_features_used, total_features_used/len(keywords_cyclerank))
            counter += total_features_used
            #print("")
        print(m, counter, "Mean/max/min feature used:", counter/len(unique_years), max, min)
        print("")

    counter=0
    max = (-1, 0)
    min = (100000000, 0)
    for i in unique_years:
        total_features_used = baseline_results_df[baseline_results_df.season == i]["page_name_{}".format(args["<baseline>"])].count()
        counter += total_features_used
        if total_features_used > max[0]:
            max = (total_features_used, i)
        else:
            if total_features_used < min[0]:
                min = (total_features_used, i)
        #print(i, args["<baseline>"], total_features_used, total_features_used/len(keywords_standard))
        #print("")
    print(args["<baseline>"], counter, "Mean/max/min feature used:", counter/len(unique_years), max, min)
    print("")

    # Loop over the seasons and extract common pages
    total_common = []
    for y in unique_years:
        selected = results[results.season == y]
        for m in args["<other_method>"]:
            common = selected[selected["page_name_{}".format(args["<baseline>"])] == selected["page_name_{}".format(m)]]
            total_common += list(common["page_name_{}".format(args["<baseline>"])].unique())
            print("{}, {} -> Common Pages = {}".format(y, m, len(common["page_name_{}".format(args["<baseline>"])].unique())))

            how_many_in_common_keywords = set.intersection(set(selected["page_name_{}".format(args["<baseline>"])]), common_keywords)
            print("{}, {} -> Common Keywords = {}".format(y, m, len(how_many_in_common_keywords)))
            print("")


    # Get only the weeks we want
    incidence = pd.read_csv(os.path.join(base_dir, args["<baseline>"], future, country, "{}-prediction.csv".format(season_years_baseline)))[["week", "incidence"]]
    start_year = season_years_baseline.split("-")[0]+"-42" if not args["--start-year"] else args["--start-year"]
    end_year = season_years_baseline.split("-")[1]+"-15" if not args["--end-year"] else args["--end-year"]
    start_season = incidence["week"] >= start_year
    end_season = incidence["week"] <= str(int(end_year.split("-")[0])+1)+"-"+end_year.split("-")[1]
    total = start_season & end_season
    total_incidence_size=len(incidence[total])

    max_features = top_features

    all_methods = [args["<baseline>"]]+args["<other_method>"]
    index = 0
    incidence = incidence[total]
    incidence = incidence.reset_index()
    incidence = incidence.drop(["index"], axis=1)
    weeks = incidence["week"]
    incidence = stz_zero(incidence["incidence"])

    step = int(len(weeks)*0.05)
    if (step == 0):
        step=1

    end_of_seasons = [i for i, n in enumerate(weeks.to_list()) if n.split("-")[1] == "15"]

    fig, ax = plt.subplots(len(all_methods), 1, figsize=(11, 6))
    for method in all_methods:

        year_features = [elem[0] for elem in get_important_pages(generate_dictionary(results, method))]

        most_important_features = generate(2020, [], path_features="../data/wikipedia_{}/{}".format(country, method))[["Week"]+list(year_features[0:max_features])]
        most_important_features = most_important_features.reset_index().drop(["index"], axis=1)

        start_season_data = most_important_features["Week"] >= start_year
        end_season_data = most_important_features["Week"] <= str(int(end_year.split("-")[0]) + 1) + "-" + end_year.split("-")[1]

        most_important_features = most_important_features[start_season_data & end_season_data]
        most_important_features = most_important_features.drop(["Week"], axis=1)

        most_important_features = stz_zero(most_important_features)
        most_important_features = most_important_features.reset_index().drop(["index"], axis=1)
        most_important_features = pd.concat([incidence, most_important_features], axis=1)

        sns.lineplot(data=most_important_features, dashes=False, ax=ax[index], sort=False)
        ax[index].set_title("{}".format(correct_name(method)), pad=5, fontsize=12)

        box = ax[index].get_position()
        ax[index].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax[index].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax[index].set_xticks([i for i in np.arange(len(weeks), step=step)])
        ax[index].set_xticklabels(weeks.iloc[::step], rotation=90)
        ax[index].set_ylim(-0.1, 1.1)
        ax[index].set_ylabel("Pageviews Variation [0, 1]")
        for i in end_of_seasons:
            ax[index].axvline(x=i, color='k', linestyle='--')

        index=index+1

    if args["--no-graph"]:
        save_filename = "{}_{}_feature_results_{}_{}.png".format(start_year, end_year, args["<baseline>"], country)
        plt.savefig(save_filename, dpi=200, bbox_inches='tight')

    plt.xlabel("Year-Week")

    #plt.figure()
    #sns.set(font_scale=1.4)
    #index=1
    for method in all_methods:

        year_features = pd.read_csv(os.path.join(base_dir, method, future, country, "{}_most_important_features_{}.csv".format(season_years, country))).drop(["week"], axis=1).columns
        year_features = [elem[0] for elem in get_important_pages(generate_dictionary(results, method))]

        most_important_features = generate(2020, [], path_features="../data/wikipedia_{}/{}".format(country, method))[list(year_features[0:max_features])]
        most_important_features = most_important_features.reset_index().drop(["index"], axis=1)
        most_important_features = stz_zero(most_important_features)
        most_important_features = most_important_features[total]
        most_important_features = most_important_features.reset_index().drop(["index"], axis=1)
        most_important_features = pd.concat([incidence, most_important_features], axis=1)
        print(most_important_features.columns)

        palette = sns.color_palette("Blues")
        corr = most_important_features.corr()

        print(corr)
        for e in zip(corr, corr["incidence"]):
            if e[0] == "incidence":
                continue
            print("\item \wiki{{ {} }} (): ${:.2f}$".format(e[0], e[1]))

        #print(corr["incidence"])
        plt.subplot(len(all_methods), 1, index)

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            if index == len(all_methods):
                ax = sns.heatmap(data=corr, mask=mask, annot=True, xticklabels=True, cmap=palette)
            else:
                ax = sns.heatmap(data=corr, mask=mask, annot=True, xticklabels=True, cmap=palette)
        index=index+1


    if not args["--no-graph"]:
        fig.tight_layout()
        plt.show()
    else:
        save_filename = "{}_{}_correlation_matrix_results_{}_{}.png".format(start_year, end_year, args["<baseline>"], country)
        #plt.savefig(save_filename, dpi=200, bbox_inches='tight')


    # Print overall common features used:
    print("Total common pages used by the models")
    print(set(total_common))
    print("")
