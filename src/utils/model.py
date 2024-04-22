from typing import List
import torch.nn as nn


def get_model_trainable_layers(model: nn.Module) -> List:
    """Gets trainable layers of the given model

    Args:
        model (nn.Module): a subclass of nn.Module

    Returns:
        List: List of trainable layers
    """
    
    print(model)

    # trainable layers
    trainable_layers = []
    
    for name, param in model.named_parameters():
        
        # if trainable
        if param.requires_grad:
            
            # append to the list
            trainable_layers.append(name)
    
    return trainable_layers
