import pandas as pd

import os, sys

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)

from ingest_data import load_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Dropping the first column
    df.drop(df.columns[0], axis=1, inplace=True)
    
    # Lowercasing the 'Brand' column
    df["Brand"] = df["Brand"].apply(lambda x: x.lower())
    
    # Filtering brands
    brands = df["Brand"].value_counts()
    df_filtered = df[df["Brand"].isin(brands[brands > 7].index.tolist())]
    
    # Replacing specific brands with 'other'
    df_filtered["Brand"] = df_filtered["Brand"].replace({
        "gigabyte": "other",
        "razer": "other",
        "rokc": "other",
        "best notebooks": "other"
    })
    
    # Extracting CPU brand
    df_filtered["CPU Brand"] = df_filtered["CPU Name"].str.split().apply(lambda x: x[0].lower())
    id2 = df_filtered["CPU Brand"].value_counts()
    df_filtered = df_filtered[df_filtered["CPU Brand"].isin(id2[id2 > 22].index)]
    
    # Extracting GPU brand
    df_filtered["GPU Brand"] = df_filtered["GPU Name"].str.split().apply(lambda x: x[0].lower())
    id3 = df_filtered["GPU Brand"].value_counts()
    df_filtered = df_filtered[df_filtered["GPU Brand"].isin(id3[id3 > 5].index)]
    
    # Applying various filters
    df_filtered = df_filtered[df_filtered["Monitor"] > 10.5]
    df_filtered = df_filtered[df_filtered["Monitor"] < 20]
    df_filtered = df_filtered[df_filtered["Width"] > 100]
    df_filtered = df_filtered[df_filtered["Width"] > 1300]
    df_filtered = df_filtered[df_filtered["RAM"] > 3]
    df_filtered = df_filtered[df_filtered["Storage Amount"] >= 32]
    df_filtered = df_filtered[df_filtered["Storage Amount"] < 16000]
    df_filtered = df_filtered[df_filtered["Weight"] < 7.1]
    df_filtered = df_filtered[df_filtered["Weight"] > 0.3]
    df_filtered = df_filtered[df_filtered["Price"] > 150]
    df_filtered = df_filtered[df_filtered["Price"] < 6500]
    
    # Replacing specific GPU brands
    df_filtered["GPU Brand"].replace("256mb", "radeon", inplace=True)
    df_filtered["GPU Brand"].replace("t550", "t500", inplace=True)
    df_filtered["GPU Brand"].replace("geforce3", "geforce", inplace=True)
    
    # Dropping 'CPU Name' and 'GPU Name' columns
    df_filtered.drop("CPU Name", axis=1, inplace=True)
    df_filtered.drop("GPU Name", axis=1, inplace=True)
    
    # Replacing specific OS values
    df_filtered["OS"] = df_filtered["OS"].replace({
        "ChromeOS": "Chrome OS",
        "No OS": "Windows 10",
        "Windows 8.1": "Windows 8"
    })
    
    # Filtering OS
    id4 = df_filtered["OS"].value_counts()
    df_filtered = df_filtered[df_filtered["OS"].isin(id4[id4 > 5].index)]
    
    # Setting OS to 'MacOS' for Apple brand
    df_filtered.loc[df_filtered["Brand"] == "apple", "OS"] = "MacOS"
    
    # One-hot encoding categorical columns
    df_filtered = pd.get_dummies(df_filtered, columns=["Brand", "OS", "CPU Brand", "GPU Brand"])
    
    return df_filtered

if __name__ == "__main__":
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))
    DATANAME = "laptop_dataset.csv"

    DATAPATH = os.path.join(PARENT_DIR, DATANAME)

    df = load_data(DATAPATH)

    cleaned_df = preprocess_data(df)
    print(cleaned_df.head())
