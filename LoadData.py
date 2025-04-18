import os
import pandas as pd


def load_dataset(name, nrows: int, index_name: str = "Time"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, name)
    df = pd.read_csv(path, parse_dates=True, index_col=index_name, nrows=nrows)
    return df