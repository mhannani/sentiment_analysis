import argparse

from src.utils.find import find_best_checkpoint


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script description")

    # Add argument for config file
    parser.add_argument("exp_name", type=str, help="experiment name")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # experiment name
    exp_name = args.exp_name
    
    # list of models
    model_ids = [
        "bert-base-multilingual-cased",
        "bert-base-arabic",
        "darijabert-arabizi",
        "DarijaBERT",
        "bert-base-arabertv2",
    ]
    
    for model_id in model_ids:
        
        # find the best checkpoint for `model_id`
        best_checkpoint = find_best_checkpoint(exp_name, model_id, by_metric='eval_accuracy')
        
        print(best_checkpoint, '\n')
