from huggingface_hub import hf_hub_download
import lightning as L
from pathlib import Path
import fasttext
import torch
from torch.utils.data import DataLoader, TensorDataset

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
    
    # configration filepath
    CONFIG_FILE = Path("configs/config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    processed_data = config['data']['processed']
    test_preprocessed_mac_csv_filename = config['data']['test_csv_filename']
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # preprocessed csv file
    test_preprocessed_mac_csv = data_root / processed_data / test_preprocessed_mac_csv_filename
    
    # data splitter
    data_splitter = DataSplitter(config, test_preprocessed_mac_csv)
    
    # get full test data
    X_test, y_test = data_splitter.get_full_data()
    
    print(len(X_test), len(y_test))
    
    # model path
    model_path = hf_hub_download(repo_id="facebook/fasttext-ar-vectors", cache_dir = CACHE_DIR, filename="model.bin")
    
    # fasttext model
    model = fasttext.load_model(model_path)

    # Convert text data to embeddings
    X_test_embeddings = encode_text_to_embeddings(X_test, model)

    # Convert embeddings to PyTorch tensors
    X_test_tensors = torch.tensor(X_test_embeddings, dtype=torch.float32)
    y_test_tensors = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    test_data = TensorDataset(X_test_tensors, y_test_tensors)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # classifier head
    model = ClassifierHead.load_from_checkpoint("./lightning_logs/version_0/checkpoints/epoch=99-step=49100.ckpt")
    
    # Initialize the Trainer with the test dataset
    trainer = L.Trainer(enable_progress_bar = True)
    
    # train the model
    predictions = trainer.test(model = model, dataloaders=test_loader)
    
    print(predictions)