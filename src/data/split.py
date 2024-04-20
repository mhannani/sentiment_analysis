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

    def split(self) -> Tuple[List]:
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
        X_train = X.iloc[train_index].values.flatten().tolist()
        X_test = X.iloc[test_index].values.flatten().tolist()
        
        # Splitting the data based on the indices for target (y)
        y_train = y.iloc[train_index].values.flatten().tolist()
        y_test = y.iloc[test_index].values.flatten().tolist()
    
        return X_train, X_test, y_train, y_test

    def split_and_save(self) -> None:
        """Splits dataset and save sets into disk. usually that for train and test sets."""
        
        # split data
        X_train, X_test, y_train, y_test = self.split()
        
        # create DataFrames for train and test sets
        train_df = pd.DataFrame({'tweets': X_train, 'type': y_train})
        test_df = pd.DataFrame({'tweets': X_test, 'type': y_test})
        
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
