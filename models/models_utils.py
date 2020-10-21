import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

##### UTILITIES #######

def generate_keywords(keywords = "../data/keywords/keywords_italy.txt"):
    """
    Generate a list of keywords (Wikipedia's pages) which are used to
    select the columns of the dataframe we are going to use as dataset
    to train our model.

    :param keywords: the path to a file containing \n separated Wikipedia's
    page names.
    :return: a keyword list.
    """
    selected_columns = []
    file_ = open(keywords, "r")
    for line in file_:
        if line != "Week":
            selected_columns.append(line.replace("\n", "").replace("\\", ""))
    return selected_columns

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

def generate_one_year(year, path_features="./../data/wikipedia_italy/new_data"):
    """
    Generate a dataframe containing the data of only one influenza season.
    The dataframe contains, for each week, the pageview of a each Wikipedia's page
    of the dataset.

    :param year: the year we want to generate the dataset of.
    :param path_features: the path where the data files are store.
    :return: a dataframe which can be used to validate the trained model.
    """

    # Generate an empty dataframe
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_features)
    file_list.sort()

    for i in range(0, len(file_list)-1):
        if int(file_list[i].replace(".csv", "")) == year-1:
            tmp_a = pd.read_csv(os.path.join(path_features, file_list[i]), encoding = 'utf8', delimiter=',')
        else:
            tmp_a = pd.DataFrame()

        if int(file_list[i+1].replace(".csv", "")) == year:
            tmp_b = pd.read_csv(os.path.join(path_features, file_list[i+1]), encoding = 'utf8', delimiter=',')
        else:
            tmp_b = pd.DataFrame()

        # If the dataset is empty the generate a new dataframe.
        # Append a new dataframe if the dataset is not empty.
        if dataset.empty:
            dataset = generate_features(tmp_a, tmp_b, int(file_list[i].replace(".csv", "")), int(file_list[i+1].replace(".csv", "")))
        else:
            dataset = dataset.append(generate_features(tmp_a, tmp_b, int(file_list[i].replace(".csv", "")), int(file_list[i+1].replace(".csv", ""))))
    return dataset


def generate_labels(stop_year, exclude, path_labels="./../data/italy/new_data"):
    """
    Generate a dataframe with all the ILI incidence data for every influenza
    season, except for the one specified by stop_year.

    :param stop_year: the influenza season we want to exclude.
    :param path_labels: the path to the data files which store the incidence data.
    :return: a dataframe containing, for each week, the incidence value.
    """

    # The stop year must not be in the exclude list
    assert (stop_year not in exclude)

    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_labels)
    file_list.sort()

    for i in range(0, len(file_list)):
        if (file_list[i] != "tabula-2006_2007.csv"):
            # Read the file
            _file = pd.read_csv(os.path.join(path_labels, file_list[i]), engine="python")

            # Append data without the stop year
            years = file_list[i].split("_")

            # Do not add years which are in the exclude list
            if int(years[1].replace(".csv", "")) in exclude:
                continue

            if int(years[1].replace(".csv", "")) != stop_year:
                if int(years[0]) == 2007:
                    dataset = dataset.append(_file[7:11])
                else:
                    dataset = dataset.append(_file[0:11])
            if int(years[0]) != stop_year-1:
                dataset = dataset.append(_file[11:26])
    return dataset

def generate_labels_one_year(stop_year, path_labels="./../data/italy/new_data"):
    """
    Generate a dataframe with the incidence data for a single influenza season.

    :param stop_year: the influenza season we want to get the data of.
    :param path_labels: the path to the files which store the incidence data.
    :return: a dataframe containing the incidence value for the specified influenza seasons.
    """
    dataset = pd.DataFrame()

    # Get all features files and sort the list
    file_list = os.listdir(path_labels)
    file_list.sort()

    for i in range(0, len(file_list)):
        if (file_list[i] != "tabula-2006_2007.csv"):
            # Read the file
            _file = pd.read_csv(os.path.join(path_labels, file_list[i]))

            # Append data without the stop year
            years = file_list[i].replace("tabula-", "").split("_")
            if int(years[1].replace(".csv", "")) == stop_year:
                if int(years[0]) == 2007:
                    dataset = dataset.append(_file[7:11])
                else:
                    dataset = dataset.append(_file[0:11])
            if int(years[0]) == stop_year-1:
                dataset = dataset.append(_file[11:26])

    return dataset

def generate_labels_sum():
    # Get all features files and sort the list
    file_list = os.listdir("./../data/austria")
    file_list.sort()
    for i in range(0, len(file_list)):
        _file = pd.read_csv(os.path.join("./../data/austria", file_list[i]))
        _file_2 = pd.read_csv(os.path.join("./../data/germany", file_list[i]))
        total = pd.DataFrame()
        total['week'] = _file['week']
        total['incidence'] = _file['incidence'] + _file_2['incidence']
        total.to_csv(file_list[i])

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
    #dmax = train.max(axis=0)
    #dmin = train.min(axis=0)
    dstd = train.std(axis=0)
    #ddenom= dmax - dmin
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

def standardize_week(train, test, column_list):

    # Concatenate together the dataframes
    data_total = pd.concat([train, test])

    # Get all the unique weeks
    unique_weeks = data_total["week"].unique()

    # Build some temporary dataframes
    train_tmp = pd.DataFrame(columns=column_list)
    test_tmp = pd.DataFrame(columns=column_list)
    total_means = pd.DataFrame(columns=column_list)

    # Generate a matrix with all the means for each week
    for c in unique_weeks:
        mean = data_total.loc[data_total.week == c, column_list].mean()
        mean.loc["week"] = c
        total_means = total_means.append(mean, ignore_index=True)

    # Generate scaled train data.
    for index, row in train.iterrows():
        train_tmp = train_tmp.append(row-total_means[total_means.week == row["week"]])

    # Generated scaled test data.
    for index, row in test.iterrows():
        test_tmp = test_tmp.append(row-total_means[total_means.week == row["week"]])

    # Reconstruct month column
    train_tmp = train_tmp.drop(["month"], axis=1)
    train_tmp["month"] = train["month"].tolist()
    test_tmp = test_tmp.drop(["month"], axis=1)
    test_tmp["month"] = test["month"].tolist()

    # Reconstruct week columns
    #train_tmp.update(train["week"])
    #test_tmp.update(test["week"])

    return (train_tmp, test_tmp)

def stz(data):
    """
    Standardize between [-1, 1] the data give by applying this
    formula to each feature:

    x_new = (x-dataset_mean)/(dataset_max - dataset_min)

    :param data: the data we want to standardize
    :return: the standardized data
    """
    dmean = data.mean(axis=0)
    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    dmax_min = dmax - dmin
    dataset_imp = (data - dmean) / dmax_min
    dataset_imp[np.isnan(dataset_imp)] = 0
    return dataset_imp

def stz_zero(data):
    """
    Standardize between [-1, 1] the data give by applying this
    formula to each feature:

    x_new = (x-dataset_mean)/(dataset_max - dataset_min)

    :param data: the data we want to standardize
    :return: the standardized data
    """
    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    dmax_min = dmax - dmin
    dataset_imp = (data - dmin) / dmax_min
    dataset_imp[np.isnan(dataset_imp)] = 0
    return dataset_imp

def get_important_pages(important_pages, top=10):
    """
    Get the most important feature selected by the model.

    :param important_pages: a dictionary with, for each of the features,
    a list of their weights in each of the models.
    :param top: how many feature we want to return.
    :return: the top feauture
    """
    imp_pages_avg = dict((k, sum(v) / float(len(v))) for k, v in important_pages.items())
    _terms_avg_top = sorted(sorted(imp_pages_avg.items(),
                                   key=lambda value: value[0]),
                            key=lambda value: value[1],
                            reverse=True
                            )
    return _terms_avg_top[0:top]

def correlation_matrix(df, title, labels, output_name):
    """
    Print the correlation matrix from the dataframe given.
    (Code taken from https://datascience.stackexchange.com/questions/10459/
    calculation-and-visualization-of-correlation-matrix-with-pandas)

    :param df: dataframe used
    :param title: the title of the graph
    :param labels: the labels used for naming rows/columns
    :return: print on screen the correlation matrix
    """
    from matplotlib import pyplot as plt

    fig = plt.figure(10, figsize=(15, 15))
    ax1 = fig.add_subplot(111)
    cax = ax1.matshow(df.corr(), vmin=-1, vmax=1)
    plt.title(title, fontsize=18)
    ax1.xaxis.set_ticks_position('bottom')
    plt.xticks(range(0, len(labels)), labels, rotation=45, fontsize=17)
    plt.yticks(range(0, len(labels)), labels, fontsize=17)
    fig.colorbar(cax)
    plt.savefig(output_name, dpi=150)

def add_month(dataset_zero):
    """
    Add a month column to the dataset
    :param dataset_zero: the original dataset, it must have a two column, one named
    year and the other named week
    :return: dataframe with added month column and removed week column
    """
    dataset_zero["week"] = dataset_zero["week"].apply(pd.to_numeric)
    dataset_zero["year"] = dataset_zero["year"].apply(pd.to_numeric)
    dataset_zero["full_date"] = pd.to_datetime(dataset_zero.year.astype(str), format='%Y') + \
                                pd.to_timedelta(dataset_zero.week.mul(7).astype(str) + ' days')
    dataset_zero["month"] = pd.DatetimeIndex(dataset_zero['full_date']).month
    dataset_zero = dataset_zero.drop(["full_date"], axis=1)
    return dataset_zero
