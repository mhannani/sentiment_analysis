import pandas as pd

from pathlib import Path
from typing import List, Union

from src.utils.converters import df_to_list
from src.utils.readers import read_df
from src.utils.cleaners import clean_df
from src.utils.encoders import label_encode
from src.utils.save import save_csv, save_json


class Preprocessor:
    """
    Reviews or comments preprocessor.
    """

    def __init__(self, config: object, csv_path: Path, output_path: Path):
        """Preprocessor class constructor

        Args:
            config (object): configuration object
            csv_path (Path): csv file path
            output_json_path (Path): json output path
        """
        
        # configuration object
        self.config = config
        
        # raw csv file path
        self.csv_path = csv_path

        # preprocessed data ouput path
        self.output_path = output_path

    def preprocess(self) -> Union[pd.DataFrame]:
        """preprocess the given csv file and return it as Dataframe.

        Returns:
            pd.DataFrame: Dataframe of cleand and preprocessed samples
        """
        
        # read the csv file
        df: pd.DataFrame = read_df(self.csv_path)
        
        # cleand the dataframe
        df: pd.DataFrame = clean_df(df)
        
        # map `class`columns classes
        df: pd.DataFrame = label_encode(df)

        return df

    def preprocess_and_export(self) -> None:
        """Preprocesses, cleans and store data in the interimediat format as json file."""
        
        # preprocess data
        df: pd.DataFrame = self.preprocess()
        
        # dataframe to list
        df_list = df_to_list(df)
        
        # save data processed as json
        save_json(self.output_path, df_list)
        
        
        # save preprocessed data as csv
        save_csv(self.output_path, df)
