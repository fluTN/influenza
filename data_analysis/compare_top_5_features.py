import argparse
import pandas as pd
import numpy as np

import os
import glob

import wikipediaapi

import pickle

LANG = {
        'italy': 'it',
        'germany': 'de',
        'belgium': 'nl',
        'netherlands': 'nl'
    }

INFLUENZA = {
        'it': 'Influenza',
        'de': 'Influenza',
        'nl': 'Griep'
    }

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_wikipedia_page_language(wiki_page, language="en", start_lang="it"):

    wiki_wiki = wikipediaapi.Wikipedia(start_lang)
    page = wiki_wiki.page(wiki_page)

    if language in page.langlinks:
        langlinks = page.langlinks[language]
        return langlinks.title
    else:
        return "NE"

def get_distance_from_influenza(wiki_page, wiki_graph):
    if wiki_page in wiki_graph:
        return wiki_graph[wiki_page]
    else:
        return "> 3"

def inspect_wikipedia_tree(language="it", depth=2):

    wiki_wiki = wikipediaapi.Wikipedia(language)

    visited_pages = [(INFLUENZA[language], 1)]
    discovered_pages = {INFLUENZA[language]: 1}

    while(len(visited_pages) != 0):

        tmp_page, counter = visited_pages.pop()

        if counter >= depth:
            continue

        if ":" in tmp_page:
            continue

        tmp_links = wiki_wiki.page(tmp_page).links

        for l in tmp_links.keys():
            if not l in discovered_pages:
                if counter < depth:
                    discovered_pages[l] = counter
                    visited_pages.append((l, counter+1))
            else:
                counter_already = discovered_pages[l]
                if counter_already > counter:
                    discovered_pages[l] = counter
                    visited_pages.append([l, counter])

        print(len(discovered_pages), len(visited_pages))

    save_obj(discovered_pages, "{}-{}-influenza".format(language, depth))
    return discovered_pages


def get_results_filename(basepath, country):
    files = [f for f in glob.glob(basepath + "/*_information_{}.csv".format(country), recursive=True)]
    season_years = os.path.basename(files[0]).split("_")[0]
    return season_years

def get_keywords_prefix(model):
    """
    Return the correct keyword's file prefix given the model
    :param model: name of the model
    :return: keyword's file prefix
    """
    if model == "cyclerank" or model == "cyclerank_pageviews":
        return "keywords_cyclerank"
    elif model == "pagerank" or model == "pagerank_pageviews":
        return "keywords_pagerank"
    else:
        return "keywords"

def convert_naming(model):
    if model == "cyclerank":
        return ("a", "b", "c")
    elif model == "cyclerank_pageviews":
        return ("d", "e", "f")
    elif model == "pagerank":
        return ("g", "h", "i")
    elif model == "pagerank_pageviews":
        return ("l", "m", "n")
    elif model == "pageviews":
        return ("o", "p", "q")
    else:
        return ("r", "s", "t")


def get_intersection_dataframe(original_data, percentage=False):

    # Compute the intersection between all of them
    intersection_df = []
    for first in original_data:
        partial_value = [convert_naming(first)]
        for second in original_data:
            intersection_value = set.intersection(set(original_data[first].page_name),
                                                  set(original_data[second].page_name))
            if percentage:
                partial_value.append((len(intersection_value)/len(original_data[first])))
            else:
                partial_value.append(len(intersection_value))
        intersection_df.append(partial_value)

    # Convert the intersection into a dataframe
    intersection_columns = ["index"] + [convert_naming(x) for x in list(original_data.keys())]
    intersection_df = pd.DataFrame(intersection_df, columns=intersection_columns)
    intersection_df = intersection_df.set_index("index")
    return intersection_df


def get_min_max_mean_feature_selected(df, start_year=2015, end_year=2019):
    min = np.inf
    max = 0
    mean = 0
    seasons = df.season.unique()
    correct_seasons = 0
    for s in seasons:
        start_end = s.split("-")
        if int(start_end[0]) >= start_year and int(start_end[1]) <= end_year:
            correct_seasons += 1
            features_slice = df[df.season == s]
            features_selected = len(features_slice)

            if features_selected > max:
                max = features_selected
            if features_selected < min:
                min = features_selected

            mean += features_selected

    return min, max, mean/correct_seasons

def get_feature_dictionary(df, start_year=2015, end_year=2019):
    """
    Group all the features into a dictionary. Namely, we generate for each
    feature a list of values which indicates their weight in each of the models
    under scrutiny
    :param df: the feature dataframe
    :param start_year: start year to consider
    :param end_year: end year to consider
    :return: a dictionary
    """

    feature_dictionary = {}
    seasons = df.season.unique()
    for s in seasons:
        start_end = s.split("-")
        if int(start_end[0]) >= start_year and int(start_end[1]) <= end_year:
            features_slice = df[df.season == s]
            for index, row in features_slice.iterrows():
                if row["page_name"] in feature_dictionary:
                    feature_dictionary[row["page_name"]].append(row["value"])
                else:
                    feature_dictionary[row["page_name"]] = [row["value"]]
    return feature_dictionary


def get_important_pages(important_pages, top=10, influenza_seasons=4):
    """
    Get the most important feature selected by the model.

    :param important_pages: a dictionary with, for each of the features,
    a list of their weights in each of the models.
    :param top: how many feature we want to return.
    :return: the top feature
    """
    imp_pages_avg = dict((k, sum(v) / float(influenza_seasons)) for k, v in important_pages.items())
    _terms_avg_top = sorted(sorted(imp_pages_avg.items(),
                                   key=lambda value: value[0]),
                            key=lambda value: value[1],
                            reverse=False
                            )
    return _terms_avg_top[0:top]

def generate_features(year_a, year_b, number_a, number_b):
    if not year_a.empty:
        if (number_a != 2007):
            first_part= year_a.copy()[41:52]
        else:
            first_part= year_a.copy()[48:52]
    else:
        first_part = pd.DataFrame()
    if not year_b.empty and number_b != 2007:
        second_part= year_b.copy()[0:15]
    else:
        second_part = pd.DataFrame()

    return first_part.append(second_part)

def generate(stop_year, exclude, path_features="./../data/wikipedia_italy/new_data"):
    """
    Generate a dataframe with as columns the Wikipedia's pages and as rows
    the number of pageviews for each week and for each page. The dataframe
    contains all the influenza season without the one specified by stop_year.

    :param stop_year: The influenza seasosn which will not be inserted into
    the final dataframe.
    :param path_features: the path to the directory containing all the files
    with the data.
    :return: a dataframe containing all influenza season, which can be used to
    train the model.
    """

    # The stop year must not be in the exclude list
    assert (stop_year not in exclude)

    # Generate an empty dataframe
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_features)
    file_list.sort()

    for i in range(0, len(file_list)-1):
        # If the file's year is equal than stop_year then do anything
        if int(file_list[i].replace(".csv", "")) != stop_year-1:
            tmp_a = pd.read_csv(os.path.join(path_features, file_list[i]), encoding = 'utf8', delimiter=',')
        else:
            tmp_a = pd.DataFrame()

        if int(file_list[i+1].replace(".csv", "")) != stop_year:
            tmp_b = pd.read_csv(os.path.join(path_features, file_list[i+1]), encoding = 'utf8', delimiter=',')
        else:
            tmp_b = pd.DataFrame()

        # Do not add years which are in the exclude list
        if int(file_list[i+1].replace(".csv", "")) in exclude:
            continue

        # If the dataset is empty the generate a new dataframe.
        # Append a new dataframe if the dataset is not empty.
        if dataset.empty:
            dataset = generate_features(tmp_a, tmp_b, int(file_list[i].replace(".csv", "")), int(file_list[i+1].replace(".csv", "")))
        else:
            dataset = dataset.append(generate_features(tmp_a, tmp_b, int(file_list[i].replace(".csv", "")), int(file_list[i+1].replace(".csv", ""))))

    return dataset

def standardize_data(train, test):
    """
    Standardize between [-1, 1] the train and test set by applying this
    formula for each feature:

    x_new = (x-dataset_mean)/(dataset_max - dataset_min)

    :param train: the training dataset (represented with a Pandas dataframe).
    :param test: the testing dataset (represented with a Pandas dataframe).
    :return: the train and test dataset standardized
    """
    dmean = train.mean(axis=0)
    dstd = train.std(axis=0)
    ddenom = dstd

    dataset_imp = (train - dmean) / ddenom
    data_imp = (test - dmean) / ddenom

    dataset_imp.fillna(method="pad", inplace=True)
    data_imp.fillna(method="pad", inplace=True)

    data_imp.replace([np.inf, -np.inf], np.nan, inplace=True)
    dataset_imp.replace([np.inf, -np.inf], np.nan, inplace=True)

    dataset_imp.fillna(method="pad", inplace=True)
    data_imp.fillna(method="pad", inplace=True)

    dataset_imp[np.isnan(dataset_imp)] = 0.0
    data_imp[np.isnan(data_imp)] = 0.0

    return (dataset_imp, data_imp)


if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("models", metavar="N", nargs="+", type=str, help="Models we want to check")
    parser.add_argument("--country", type=str, default="italy", help="Country name")
    parser.add_argument("--basedir", type=str, default="../models/complete_results_standardized_2")
    parser.add_argument("--future", default=True, action="store_false")
    parser.add_argument("--start-year", default=2015, type=int)
    parser.add_argument("--end-year", default=2019, type=int)
    parser.add_argument("--save", default=False, action="store_true")
    args = parser.parse_args()

    # Set some variables
    country = args.country
    base_dir = args.basedir
    future = "future" if args.future else "no-future"
    start_year = int(args.start_year)
    end_year = int(args.end_year)

    # Generate wikipedia graph
    depth = 3
    graph_name = "./wikipedia_graphs/{}-{}-influenza.pkl".format(LANG[country], depth)
    if os.path.exists(graph_name):
        print("Graph file exists! using that.")
        wiki_graph = load_obj(graph_name)
    else:
        wiki_graph = inspect_wikipedia_tree(LANG[country], depth)

    # Get the keywords for all the models, then compute the intersection between all of them
    total_keywords = {}
    total_features = {}
    for model_name in args.models:

        # Get models keywords
        keywords_prefix = get_keywords_prefix(model_name)
        model_total_keywords = pd.read_csv(os.path.join("../data/keywords",
                                                        "{}_{}.txt".format(keywords_prefix, country)),
                                           header=None, names=["page_name"])
        total_keywords[model_name] = model_total_keywords

        # Get selected features
        baseline_results_path = os.path.join(base_dir, model_name, future, country)
        season_years = get_results_filename(baseline_results_path, country)
        features_file = os.path.join(baseline_results_path, "{}_features_{}.csv".format(season_years, country))
        features = pd.read_csv(features_file)[["season", "page_name", "value"]]

        # Filter the data by analyzing only the range we want
        start_season_year = features.season.str.split("-", expand=True)
        features["start_year"] = start_season_year[0]
        features["end_year"] = start_season_year[1]
        features = features[(features["start_year"] >= str(start_year)) & (features["end_year"] <= str(end_year))]

        # Compute the best features over all models
        feature_dictionary = get_feature_dictionary(features)
        best_features = get_important_pages(feature_dictionary, top=5, influenza_seasons=start_year-end_year)
        best_features = [[x[0], x[1]] for x in best_features]

        total_features[model_name] = pd.DataFrame(best_features, columns=["page_name", "mean_weigth"])

        # Remove the year
        total_features[model_name] = total_features[model_name][total_features[model_name].page_name != "year"]

        # Get the features values and choose only the best ones find before. Moreover, filter then by the year.
        features_values = generate(2020, [], path_features="../data/wikipedia_{}/{}".format(country, model_name))
        year_week = features_values.Week.str.split("-", expand=True)
        features_values["start_year"] = year_week[0]
        features_values["week"] = year_week[1]
        features_values = features_values[(features_values["start_year"] >= str(start_year)) & (features_values["start_year"] <= str(end_year))]
        features_values = features_values[~((features_values["start_year"] == str(start_year)) & (features_values["week"] < "42"))]

        # Select only the top-5 and standardize the data
        features_values = features_values[total_features[model_name].page_name]
        features_values, _ = standardize_data(features_values, features_values)

        # Obtain the incidence value
        incidence = pd.read_csv(os.path.join(baseline_results_path, "{}-prediction.csv".format(
            season_years
        )))
        year_week = incidence.week.str.split("-", expand=True)
        incidence["start_year"] = year_week[0]
        incidence["week"] = year_week[1]
        incidence = incidence[
            (incidence["start_year"] >= str(start_year)) & (incidence["start_year"] <= str(end_year))]
        incidence = incidence[
            ~((incidence["start_year"] == str(start_year)) & (incidence["week"] < "42"))]

        # Add the incidence to the total features
        features_values.reset_index(drop=True, inplace=True)
        incidence.reset_index(drop=True, inplace=True)
        features_values["incidence"] = incidence["incidence"]

        correlation_incidence_values = features_values.corr()["incidence"]
        correlation_incidence_values.reset_index(drop=True, inplace=True)
        total_features[model_name]["PCC"] = correlation_incidence_values

    # Save the top-5 feature for each model
    top_5_features = pd.DataFrame()
    pages = pd.DataFrame()
    for k in total_features:

        name, type, dist = convert_naming(k)

        pages["pages"] = total_features[k]["page_name"][0:5].apply(lambda x: x.replace("_", " ")) + " (" + total_features[k]["page_name"][0:5].apply(lambda x: get_wikipedia_page_language(
            x, 'en', LANG[country]
        )) + ")"
        pages["PCC"] = total_features[k]["PCC"][0:5]
        pages["distance"] = total_features[k]["page_name"][0:5].apply(lambda x: get_distance_from_influenza(x, wiki_graph))
        print(pages["distance"])
        top_5_features[[name, type, dist]] = pages

    top_5_features = top_5_features.reindex(sorted(top_5_features.columns), axis=1)
    print(top_5_features)

    with pd.option_context("max_colwidth", 1000):
        top_5_features.to_latex("{}_features_latex.txt".format(country), index=None, float_format="%.2f")
