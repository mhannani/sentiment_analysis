import json
from pathlib import Path
from typing import List
import pandas as pd


def save_json(file_path: Path, json_data: List) -> None:
    """Saves json data in the specified location.

    Args:
        file_path (Path): file path to store the json data
        json_data (List): json object as List of samples
    """

    # Create the root directory of the file if it does not exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # write data
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

def save_csv(file_path: Path, df: pd.DataFrame):
    """Saves df as csv in the provided location.

    Args:
        file_path (Path): file path where to store the csv file
        df (pd.DataFrame): dataframe object to write.
    """
    
    # Get the extension of the file
    return df.to_csv(file_path.with_suffix('.csv'), index=False)