import pandas as pd

def carregar_dados():
    return pd.read_csv('./datasets/heart_failure_clinical_records_dataset.csv', sep=',')