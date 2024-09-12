import pandas as pd

def read_data(file_path):
    return pd.read_csv(file_path, sep='\t')

def read_dataNO(file_path):
    return pd.read_csv(file_path)

def correlations(data):
    return data.corr()