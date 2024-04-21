import pandas as pd


def label_encode(df: pd.DataFrame) -> pd.DataFrame:
    """encodes label of the `class` and `type` columns.

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: output dataframe
    """
    
    # class mapping for type
    type_class_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    
    # map classes to numerical representation
    df['type'] = df['type'].map(type_class_mapping)

    # map classes to numerical representation
    classname_class_mapping = {"dialectal": 1, "diatectal": 1, "standard": 0}

    # map classes to numerical representation
    df['class_name'] = df['class_name'].map(classname_class_mapping)

    # class maping
    return df
