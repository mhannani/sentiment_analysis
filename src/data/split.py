import pandas as pd
from typing import Tuple, List, Union
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.save import save_csv


class DataSplitter:
    """Data Splitter helper functions"""
    
    def __init__(self, config: object, df: pd.DataFrame) -> None:
        """class constructor

        Args:
            config (object): configuration object
            df (pd.DataFrame): dataframe object
        """
        
        # configuration object
        self.config = config
        
        # dataframe
        self.df = df
        
        # data root
        self.data_root = Path(self.config['data']['root'])
        
        # processed root
        self.processed_root = self.config['data']['processed']
        
        # train_test_or_val_size
        train_test_or_val_size = self.config['params']['train_test_or_val_size']
        
        # Initialize StratifiedShuffleSplit
        self.stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=train_test_or_val_size, random_state=42)

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
    
        return train_df, test_df

    def split_and_save(self) -> None:
        """Splits dataset and save sets into disk. usually that for train and test sets."""
        
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
        
        # save test set
        save_csv(test_csv_abs_path, test_df)
