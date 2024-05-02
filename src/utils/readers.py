import json
import pandas as pd
from typing import List
from pathlib import Path


def read_df(csv_filepath: Path, sep: str = ',') -> pd.DataFrame:
    """Read dataframe and return it.

    Args:
        csv_filepath (path): csv file's absolute path

    Returns:
        pd.DataFrame: The dataframe.
    """
    import csv
    
    csv.field_size_limit(131072 * 10)  # You can adjust the multiplier as needed

    
    print(csv_filepath)
    return pd.read_csv(csv_filepath.as_posix(), sep = sep)

def read_json(json_filepath: Path) -> List:
    """Reads json data from file.

    Args:
        json_filepath (Path): json filepath.

    Returns:
        List: List of samples
    """

    # Initialize an empty list to store the data
    data = []

    # Open the JSON file and load the data
    with open(json_filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data

def read_json_to_df(file_path: str) -> pd.DataFrame:
    """Reads JSON file containing list of dictionaries into DataFrame.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the JSON file.
    """
    # Read the JSON file into a DataFrame
    df = pd.read_json(file_path)
    
    return df