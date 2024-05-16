import pandas as pd
import os, sys

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a given file path.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    DATANAME = "laptop_dataset.csv"

    DATAPATH = os.path.join(PARENT_DIR, DATANAME)

    df = load_data(filepath=DATAPATH)
    print(df.head())
