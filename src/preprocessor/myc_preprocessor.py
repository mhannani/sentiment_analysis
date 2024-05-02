import pandas as pd
from pathlib import Path


from src.preprocessor.preprocessor import Preprocessor
from src.utils.cleaners import clean_text


class MYCPreprocessor(Preprocessor):
    """Preprocessor class for MYC dataset"""
    
    def __init__(self, config: object, csv_path: Path, output_path: Path):
        """class constructor

        Args:
            config (object): TOML configuration object
            csv_path (Path): csv filepath
            output_path (Path): output directory for the preprocessed data
        """
        
        # call the parenet's __init__ method
        super().__init__(config, csv_path, output_path)
    
    @staticmethod
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the provided dataframe and retunrn the cleaned version

        Basically:
            1. Removing missing values,
            2. Dropping the column `class` as it's not relevent for the porject objective.
            3. Removing rows where the 'tweets' column contains empty strings.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output/cleaned dataframe
        """
        
        # drop rows where tweets, type or class are NaN
        df.dropna(subset=['sentence', 'polarity'], inplace=True)
        
        # Renaming columns to avoid reserved keywords conflicts
        df.rename(columns={'sentence': 'tweets'}, inplace=True)
        df.rename(columns={'polarity': 'class_name'}, inplace=True)
        
        # clean tweets column
        df['tweets'] = df['tweets'].apply(clean_text)
        
        # Remove rows where 'tweets' column contains empty strings
        df = df[df['tweets'] != '']

        return df

    def label_encode(df: pd.DataFrame) -> pd.DataFrame:
        """encodes label of the `class` and `type` columns.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe
        """
        
        # try to unify the class_name labels with other datasets
        # class mapping for type
        type_class_mapping = {"-1": 0, "1": 2}
        
        # map classes to numerical representation
        df['class_name'] = df['class_name'].map(type_class_mapping)
        
        return df
