from huggingface_hub import hf_hub_download
import lightning as L
from pathlib import Path
import fasttext
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse

from src.models.classifier import ClassifierHead
from src.utils.parsers import parse_toml
from src.data.split import DataSplitter
from src.utils.encoders import encode_text_to_embeddings


if __name__ == "__main__":
    """
    Evaluate Sentiment analysis classifier using FastText embeddings
    """

    # cache directory for FastText model
    CACHE_DIR = "/netscratch/mhannani/fasttext_models"
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Script description")
    
    # Add argument for config file
    parser.add_argument("config_file", type=str, help="configuration filename")
    
    # Add arguement for experiment file
    parser.add_argument("exp_name", type=str, help="experiment name")
    
    # Parse the command-line arguments
    args = parser.parse_args()
        
    # configration filepath
    CONFIG_FILE = Path(f"configs/{args.config_file}.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    processed_data = config['data']['processed']
    preprocessed_corpus_csv_filename = config['data']['preprocessed_corpus_csv']
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # preprocessed csv file
    preprocessed_corpus_csv = data_root / processed_data / preprocessed_corpus_csv_filename
    
    # data splitter
    data_splitter = DataSplitter(config, preprocessed_corpus_csv)
    
    # split data into train and val sets
    _, X_test, _, y_test = data_splitter.split(returned_as_lists = True)
    
    print(len(X_test), len(y_test))
    
    # # model path
    # model_path = hf_hub_download(repo_id="facebook/fasttext-ar-vectors", cache_dir = CACHE_DIR, filename="model.bin")
    
    # # fasttext model
    # model = fasttext.load_model(model_path)

    # classifier head
    model = ClassifierHead.load_from_checkpoint(f"./fasttext_model_{args.exp_name}/lightning_logs/version_0/checkpoints/epoch=99-step=20200.ckpt")
    
    # Convert text data to embeddings
    X_test_embeddings = encode_text_to_embeddings(X_test, model)

    # Convert embeddings to PyTorch tensors
    X_test_tensors = torch.tensor(X_test_embeddings, dtype=torch.float32)
    y_test_tensors = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    test_data = TensorDataset(X_test_tensors, y_test_tensors)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize the Trainer with the test dataset
    trainer = L.Trainer(enable_progress_bar = True)
    
    # train the model
    predictions = trainer.test(model = model, dataloaders=test_loader)
    
    print(predictions)