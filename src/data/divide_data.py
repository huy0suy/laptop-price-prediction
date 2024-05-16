import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from ingest_data import load_data
from preprocessing import preprocess_data

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, val_size: float = 0.1, random_seed: int = 42):
    """
    Split the data into training, validation, and test sets.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    test_size (float): The proportion of the data to include in the test set.
    val_size (float): The proportion of the training data to include in the validation set.
    random_seed (int): The random seed for reproducibility.

    Returns:
    tuple: A tuple containing the training, validation, and test sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Split the training data into training and validation sets
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=random_seed)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    DATANAME = "laptop_dataset.csv"

    DATAPATH = os.path.join(PARENT_DIR, DATANAME)

    df = load_data(DATAPATH)

    cleaned_df = preprocess_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cleaned_df, target_column="Price")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
