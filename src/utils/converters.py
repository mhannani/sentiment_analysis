import pandas as pd
from typing import List


def df_to_list(df: pd.DataFrame) -> List[dict]:
    """Converts df to List.

    Args:
        df (pd.DataFrame): pandas dataframe

    Returns:
        List[dict]: List of samples.
    """

    # Initialize an empty list to store dictionaries
    result = []
    
    # Iterate over each row in the DataFrame
    for idx, row in enumerate(df.itertuples(), start=1):
        
        # Create a dictionary for the current row
        sample_dict = {"key": idx}
        
        # Add available fields to the dictionary
        if hasattr(row, 'tweets'):
            sample_dict["tweet"] = row.tweets
   
        if hasattr(row, 'type'):
            sample_dict["type"] = row.type

        if hasattr(row, 'class_name'):
            sample_dict["class_name"] = row.class_name
        
        result.append(sample_dict)
    
    return result


if __name__ == "__main__":
    import os
    import sys

    # Get the path to the directory containing this script (src/prompts)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    sys.path.append(os.path.abspath(os.path.join(script_dir, "../..")))

    from pathlib import Path
    
    from src.utils.parsers import parse_toml
    from src.utils.readers import read_df

    # Change the working directory to the root directory of the project
    os.chdir("../..")

    # Load the configuration from the TOML file
    config = parse_toml(Path("./configs/config.toml"))

    # useful variables
    data_root = Path(config['data']['root'])
    raw_data = config['data']['raw']
    mac_csv_filename = config['data']['mac_csv_filename']

    # constructing the filepath of the corpus
    csv_filepath_raw_data = data_root / raw_data / mac_csv_filename
    
    df = read_df(csv_filepath_raw_data)
    
    df_list = df_to_list(df)
    
    print(df_list[:5])