import pandas as pd


def count_nans(df: pd.DataFrame, column_name: str) -> int:
    """Counts the number of missing values in a given column.

    Args:
        df (pd.DataFrame): A pandas' dataframe object
        column_name (str): Column name / Feature name

    Returns:
        int: Number of missing values
    """
    
    return df[column_name].isna().sum()
