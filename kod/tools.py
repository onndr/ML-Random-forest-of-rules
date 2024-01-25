import json

import numpy as np
import pandas as pd
from constants import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score


def read_csv_data(csv_data_path, column_names=None, separator=','):
    if column_names is None:
        return pd.read_csv(csv_data_path, sep=separator)
    return pd.read_csv(csv_data_path, names=column_names, sep=separator)


def get_data(data_path, column_names=None, separator=',', target_column_name='class', columns_to_drop=None):
    data = read_csv_data(data_path, column_names, separator)

    y = data[target_column_name].to_list()
    classes = list(set(y))
    X = data.drop([target_column_name], axis=1)
    if columns_to_drop is not None:
        X = X.drop(columns_to_drop, axis=1)
    columns_values = {col: X[col].unique().tolist() for col in X.columns}
    X = X.to_dict('records')
    return X, y, columns_values, classes


def get_mushrooms_data():
    target_column_name = 'class'

    res = get_data(MUSHROOMS_DATA_PATH, MUSHROOMS_COLUMN_NAMES, target_column_name=target_column_name)
    return res


def get_students_data():
    target_column_name = 'Target'
    separator = ';'
    columns_to_drop = ['Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP', 'Curricular units 2nd sem (grade)', 'Curricular units 1st sem (grade)', 'Previous qualification (grade)']

    res = get_data(STUDENTS_DATA_PATH, separator=separator,
                   target_column_name=target_column_name,
                   columns_to_drop=columns_to_drop)
    return res


def get_titanic_data():
    target_column_name = 'Survived'
    columns_to_drop = ['Fare', 'Name', 'PassengerId']

    res = get_data(TITANIC_DATA_PATH, target_column_name=target_column_name, columns_to_drop=columns_to_drop)
    return res


def dump_exp_results(filename: str, results: dict):
    with open(EXPERIMENTS_RESULTS_FOLDER_PATH + filename, "w") as f:
        json.dump(results, f)


def get_exp_results(filename: str):
    with open(EXPERIMENTS_RESULTS_FOLDER_PATH + filename, "r") as f:
        return json.load(f)


def quality_measures(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = {}
    f1 = {}

    if len(classes) >= 2:
        prec = precision_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    # else:
    #     for c in classes:
    #         prec[c] = precision_score(y_true, y_pred, pos_label=c)
    #         f1[c] = f1_score(y_true, y_pred, pos_label=c)

    return cm, acc, prec, f1


def count_statistics(confusion_matrix_values: list, accuracy_values: list,
                     precision_values: list, f1_score_values: list):

    arr = np.array(confusion_matrix_values)

    res = {
        "confusion_matrix": {
            "avg": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist()
        },
        "accuracy": {
            "avg": np.mean(accuracy_values),
            "std": np.std(accuracy_values),
            "max": max(accuracy_values),
            "min": min(accuracy_values)
        },
        "precision": {
            "avg": np.mean(precision_values),
            "std": np.std(precision_values),
            "max": max(precision_values),
            "min": min(precision_values)
        },
        "f1_score": {
            "avg": np.mean(f1_score_values),
            "std": np.std(f1_score_values),
            "max": max(f1_score_values),
            "min": min(f1_score_values)
        }
    }

    return res


if __name__ == "__main__":
    get_mushrooms_data()
    get_students_data()
    get_titanic_data()
