import json
import csv
import pandas as pd
from typing import List
from pathlib import Path


def read_df(csv_filepath: Path, sep: str = ',', encoding='utf-8') -> pd.DataFrame:
    """Read dataframe and return it.

    Args:
        csv_filepath (Path): _description_
        sep (str, optional): seperator for the row. Defaults to ','.
        encoding (str, optional): the encoding used. Defaults to 'utf-8'.

    Returns:
        pd.DataFrame: dataframe
    """
    
    return pd.read_csv(csv_filepath.as_posix(), sep = sep, encoding='utf-8')

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