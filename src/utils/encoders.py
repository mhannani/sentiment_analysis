import pandas as pd
from typing import List
import torch
import numpy as np
from fasttext.FastText import _FastText

def label_encode(df: pd.DataFrame) -> pd.DataFrame:
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


def encode_text_to_embeddings(text_data: List[str], fasttext_model: _FastText) -> torch.Tensor:
    """Encodes the given set of sample into its embeddings using the 
    provided `fasttext_model` intance.

    Args:
        text_data (List[str]): List of sample to compute thier embeddings
        fasttext_model (_FastText): Fasttext model; a `_FastText` object
        
    Returns
        torch.Tensor: List of embeddings
    """
    
    # List of emebeddings
    embeddings = []
    
    # Loop through all samples
    for sentence in text_data:
        
        # split the sentence in tokens
        tokens = sentence.split()

        # compute the embeedings of each token and and compute thier mean
        sentence_embedding = np.mean([fasttext_model.get_word_vector(token) for token in tokens], axis=0)

        # add the current embedding to the list of embeddings
        embeddings.append(sentence_embedding)

    # return the embeddings as torch.Tensor
    return np.array(embeddings)
    