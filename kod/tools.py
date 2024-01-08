import pandas as pd


def read_data(csv_data_path, column_names):
    return pd.read_csv(csv_data_path, names=column_names)
