from si.data.dataset import Dataset
import pandas as pd
import numpy as np

def read_csv(filename: str, sep:str, features: bool, label: bool) -> Dataset:
    
    dataframe = pd.read_csv(filepath_or_buffer=filename, sep=sep)

    if features and label:
        X = dataframe.iloc[:, :-1].to_numpy()
        Y = dataframe.iloc[:, -1].to_numpy()
        feature_names = dataframe.columns[:-1]
        label_name = dataframe.columns[-1]
        return Dataset(X=X, y=Y, features=feature_names, label=label_name)
    
    elif features:
        X = dataframe.to_numpy()
        feature_names = dataframe.columns
        return Dataset(X=X, features=feature_names)
    
    elif label:
        X = np.array()
        Y = dataframe.iloc[:, 1].to_numpy()
        label_name = dataframe.columns[-1]
        return Dataset(X=X, Y=Y, label=label_name)
    else:
        return None

def write_csv(filename: str,
              dataset: "Dataset",
              features: bool,
              label: bool,
              sep: str = '.') -> None:
    df = pd.DataFrame(dataset.X, columns=dataset.features)
    if features:
        df.columns = dataset.features
    
    if label:
        Y = dataset.Y
        label_name = dataset.label

    else:
        Y = None
        label_name = None

    df.to_csv(filename, sep=sep, index=False)