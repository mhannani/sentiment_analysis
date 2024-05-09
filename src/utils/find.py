
import json
from typing import List
from pathlib import Path


def find_best_checkpoint(exp_name: str, model_id: str, by_metric: str = 'eval_accuracy') -> List[dict]:
    """Find the best checkpoint of `model_id` model in the given experiment `exp_name` by the given
    metric `by_metric`.

    Args:
        exp_name (str): The experiment name
        model_id (str): The model id
        by_metric (str, optional): the metric name to find by. Defaults to 'accuracy'.

    Returns:
        List[dict]: List of each model's best checkpoint.
    """
    
    # best checkpoint
    best_checkpoint_meta = None
    
    # best metric
    best_metric = 0.0
    
    # experiment folder path
    exp_path = Path('/netscratch/mhannani') / exp_name / model_id
    
    # Get list of checkpoint folders sorted based on numerical part of the name
    checkpoint_folders = sorted(exp_path.iterdir(), key=lambda x: int(x.name.split('-')[-1]))
        
    # Get the last checkpoint folder
    last_checkpoint_folder = checkpoint_folders[-1]

    # Read trainer_state.json from the last checkpoint folder
    trainer_state_file = last_checkpoint_folder / 'trainer_state.json'

    # open trainer_state.json file
    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)
        
        # Iterate through log_history to find the highest accuracy
        for log_entry in trainer_state['log_history']:
            metric = log_entry.get(by_metric, 0.0)
            if metric > best_metric:
                best_metric = metric
                best_checkpoint_meta = log_entry
                
    return {
            'exp_name': exp_name, 
            'model_id': model_id, 
            **best_checkpoint_meta
        }
