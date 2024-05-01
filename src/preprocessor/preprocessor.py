import pandas as pd

from pathlib import Path
from typing import List, Union

from src.utils.converters import df_to_list
from src.utils.readers import read_df
from src.utils.cleaners import clean_text
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

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:

        
        """clean the given DataFrame object and return the cleaned version.

        Basically:
            1. Removing missing values,
            2. Removing samples belonging to the mixed type,
            3. Dropping the column `class` as it's not relevent for the porject objective.
            4. Removing rows where the 'tweets' column contains empty strings.

        Args:
            df (pd.DataFrame): DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned version of the DataFrame.
        """
        
        # drop rows where tweets, type or class are NaN
        df.dropna(subset=['tweets', 'type', 'class'], inplace=True)
        
        # Renaming column to avoid reserved keywords conflicts
        df.rename(columns={'class': 'class_name'}, inplace=True)

        # clean tweets column
        df['tweets'] = df['tweets'].apply(clean_text)
        
        # Remove rows where the 'type' column has the value 'mixed'
        df = df[df['type'] != 'mixed']
        
        # Remove rows where 'tweets' column contains empty strings
        df = df[df['tweets'] != '']

        return df
    
    def label_encode(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def preprocess(self) -> Union[pd.DataFrame]:
        """preprocess the given csv file and return it as Dataframe.

        Returns:
            pd.DataFrame: Dataframe of cleand and preprocessed samples
        """
        
        # read the csv file
        df: pd.DataFrame = read_df(self.csv_path)
        
        # cleand the dataframe
        df: pd.DataFrame = self.clean_df(df)
        
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
