import pandas as pd


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the given DataFrame object and return the cleaned version. 
    Basically:
        1. Removing missing values,
        2. Removing samples from the 

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """