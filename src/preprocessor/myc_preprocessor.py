import pandas as pd
from pathlib import Path
from src.preprocessor.preprocessor import Preprocessor


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
    
    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the provided dataframe and retunrn the cleaned version

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output/cleaned dataframe
        """
        
        raise NotImplementedError

    def label_encode(df: pd.DataFrame) -> pd.DataFrame:
        """encodes label of the `class` and `type` columns.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe
        """
        
        raise NotImplementedError
