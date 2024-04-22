import torch
import transformers
from torch.utils.data import Dataset, TensorDataset
from typing import Tuple, Dict
from pathlib import Path
import pandas as pd


# class SentimentDataset(Dataset):
#     """Sentiment Analysis Dataset class"""
    
#     def __init__(self, df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer = None) -> None:
#         """class constructor

#         Args:
#             df (pd.DataFrame): pandas dataframe object
#         """
        
#         # Call the constructor of the parent class
#         super().__init__()

#         # dataframe object
#         self.df = df

#         # tokenizer
#         self.tokenizer = tokenizer

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
#         """Get item at the given index

#         Args:
#             index (int): index of the item to be retrieved

#         Returns:
#             Tuple[torch.Tensor]: _description_
#         """
        
#         # tweet as text
#         tweet_text = str(self.df.iloc[index]['tweets'])
        
#         # sentiment type as int
#         sentiment_type = self.df.iloc[index]['type']
                                         
#         return tweet_text, sentiment_type

#     def __len__(self) -> int:
#         """Compute the length of the dataset

#         Returns:
#             int: Number of sample in the dataset
#         """
        
#         return len(self.df)


class SentimentDataset(Dataset):
    """Sentiment Analysis Dataset class"""
    
    def __init__(self, df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer = None, train_mode: bool = True) -> None:
        """class constructor

        Args:
            df (pd.DataFrame): pandas dataframe object
        """
        
        # Call the constructor of the parent class
        super().__init__()

        # dataframe object
        self.df = df

        # tokenizer
        self.tokenizer = tokenizer
        
        # tweets
        self.tweets = self.df['tweets'].tolist()
        
        # train mode
        self.train_mode = train_mode
        
        # labels
        if self.train_mode:
            self.labels = self.df['type'].tolist()

        # encode tweets
        self.encodings = self.tokenizer(self.tweets, padding='max_length', truncation=True, max_length=512, return_tensors="pt", return_attention_mask=True)

    def __getitem__(self, index: int) -> Dict:
        """Get item at the given index

        Args:
            index (int): index of the item to be retrieved

        Returns:
            Tuple[torch.Tensor]: _description_
        """
        
        # get the element at the index provided
        item = {key: val[index] for key, val in self.encodings.items()}
        
        # add the lebel if in training model
        if self.train_mode:
            item["labels"] = torch.tensor(self.labels[index])
            # item["tweet"] = self.tweets[index]

        return item
    
    def __len__(self) -> int:
        """Compute the length of the dataset

        Returns:
            int: Number of sample in the dataset
        """
        
        return len(self.tweets)

