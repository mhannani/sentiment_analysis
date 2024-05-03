import csv
from typing import Union
import pandas as pd
from pathlib import Path


from src.preprocessor.preprocessor import Preprocessor
from src.utils.cleaners import clean_text
from src.utils.encoders import label_encode


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

    @staticmethod
    def label_encode(df: pd.DataFrame) -> pd.DataFrame:
        """encodes label of the `class` and `type` columns.

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: output dataframe
        """
        
        # try to unify the class_name labels with other datasets
        # class mapping for type
        type_class_mapping = {-1: 0, 1: 2}
        
        # map classes to numerical representation
        df['class_name'] = df['class_name'].map(type_class_mapping)
        
        return df

    def read_csv_and_handle_extra_commas(self) -> pd.DataFrame:
        """Reads csv with corrupted and extra commas to handle the issue of parsing

        Returns:
            pd.DataFrame: _description_
        """
        
        # all samples
        samples = []

        # open file in read mode
        with open(self.csv_path, 'r', encoding='utf-16') as f_in:
             
             # reader object
             reader = csv.reader(f_in)
             
             # go through all rows
             for idx, row in enumerate(reader):
                 
                 # skip empty line - sentence without labels...
                 if len(row) < 2:
                     continue

                 if row:
                     # preprocess only the sample that has labels (1, or -1) as the last element in the list
                     # the text would be to contacenate all the preceding list elements

                    if len(row[-1]) in [1, 2]:
                        
                        # construct the sentence, tweet or comment
                        sentence = ' '.join(row[:-1])
                        
                        
                        # clean the sentence
                        sentence = clean_text(sentence)

                        # get the label
                        try:
                            # sometimes the label is 1- instead of -1, we handle that as well
                            if row[-1] in ["1-", "-"]:
                                
                                # replace that incorrect label
                                row[-2] = -1
                            
                            # convert it to int
                            label = int(row[-1])

                        except ValueError:
                            pass
                        
                        # current sample
                        sample = (sentence, label)
                        
                        # add it to the list of samples
                        samples.append(sample)
        
        # Construct DataFrame from the list of samples
        df = pd.DataFrame(samples, columns=['sentence', 'polarity'])

        return df


    def preprocess(self) -> Union[pd.DataFrame]:
        """preprocess the given csv file and return it as Dataframe.

        Returns:
            pd.DataFrame: Dataframe of cleand and preprocessed samples
        """

        # read the csv file
        df: pd.DataFrame = self.read_csv_and_handle_extra_commas()
        
        # cleand the dataframe
        df: pd.DataFrame = self.clean_df(df)
        
        # map `class`columns classes
        df: pd.DataFrame = self.label_encode(df)

        return df