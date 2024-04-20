import torch
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path

from src.utils.readers import read_df


class SentimentDataset(Dataset):
    """Dataset class"""
    
    def __init__(self, csv_path: Path) -> None:
        """class constructor"""
        
        # Call the constructor of the parent class
        super().__init__()

        # csv path
        self.csv_path: Path = csv_path
        
        # read csv file
        self.df = read_df(self.csv_path, sep = ',')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        """Get item at the given index

        Args:
            index (int): index of the item to be retrieved

        Returns:
            Tuple[torch.Tensor]: _description_
        """
        
        return (self.df.iloc[index],)
