import pandas as pd
from typing import Tuple, List, Union
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.converters import df_to_list
from src.utils.readers import read_df
from src.utils.save import save_csv, save_json


class DataSplitter:
    """Data Splitter helper functions"""
    
    def __init__(self, config: object, preprocessed_csv: Path) -> None:
        """class constructor

        Args:
            config (object): configuration object
            preprocessed_csv (Path): preprocessed csv data file path
        """
        
        # configuration object
        self.config = config

        # data root
        self.data_root = Path(self.config['data']['root'])

        # processed root
        self.processed_root = self.config['data']['processed']

        # read dataframe
        self.df = read_df(preprocessed_csv)

        # train_test_or_val_size
        train_test_or_val_size = self.config['params']['train_test_or_val_size']

        # Initialize StratifiedShuffleSplit
        self.stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=train_test_or_val_size, random_state=42)

        # Check for missing values in the DataFrame
        self._check_missing_values()
    
    def _check_missing_values(self) -> None:
        """Check for missing values in the DataFrame"""
        
        # count the missing values
        missing_values = self.df.isna().sum()
        
        # throw an error when missing values are present
        if missing_values.sum() > 0:
            raise Exception("Missing values found in the DataFrame")
            
    def split(self) -> Tuple[pd.DataFrame]:
        """Splits dataframe into train and test sets.

        Returns:
            Tuple[List]: Tuple of lists eg. X_train, y_train, X_test, y_test
        """
        
        # Defining features (X) and target (y)
        X = self.df.drop(columns=['type', 'class_name'])
        y = self.df['type']
        
        # Get the train and test indices from the single split
        train_index, test_index = next(self.stratified_splitter.split(X, y))

        # Splitting the data based on the indices for features(X)
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        # Splitting the data based on the indices for target (y)
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        # Combining X_train and y_train into train_df and X_test and y_test into test_df
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # Adding the 'class_name' column to train_df and test_df
        train_df['class_name'] = self.df.loc[train_index, 'class_name']
        test_df['class_name'] = self.df.loc[test_index, 'class_name']
        
        return train_df, test_df

    def split_and_save(self, _save_json: bool = False) -> None:
        """Splits dataset and save sets into disk. usually that for train and test sets.

        Args:
            save_json (bool, optional): Whether to save data as json as well. Defaults to False.
        """
        
        # split data
        train_df, test_df = self.split()

        # train csv filename
        train_csv_filename = self.config['data']['train_csv_filename']
        
        # test csv filename
        test_csv_filename = self.config['data']['test_csv_filename']
        
        # train csv abs path
        train_csv_abs_path = self.data_root / self.processed_root / train_csv_filename

        # test csv abs path
        test_csv_abs_path = self.data_root / self.processed_root / test_csv_filename
        
        # save train set
        save_csv(train_csv_abs_path, train_df)
        
        # save train set as json
        if _save_json:
            # convert df to list of sample for train df
            train_json_data = df_to_list(train_df)
            
            # save data to json file
            save_json(train_csv_abs_path, train_json_data)

        # save test set
        save_csv(test_csv_abs_path, test_df)
        
        # save test set as json
        if _save_json:
            # convert df to list of samples for test df
            test_json_data = df_to_list(test_df)
            
            # save data to json file
            save_json(test_csv_abs_path, test_json_data)       