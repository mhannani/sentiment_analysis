from pathlib import Path
from pandas import read_csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from pandas import read_csv
import numpy as np


from src.data.sentiment_data import SentimentDataset
from src.data.split import DataSplitter
from src.utils.parsers import parse_toml
from src.utils.get import get_model_tokenizer
from src.data.sentiment_data import SentimentDataset
from src.data.split import DataSplitter
from src.utils.get import get_model_tokenizer


def compute_metrics(p):
    print("p[]", p)
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    print(labels, pred)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":

    """Train custom BERT model"""

    # configration filepath
    CONFIG_FILE = Path("configs/config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    processed_data = config['data']['processed']
    preprocessed_mac_csv_filename = config['data']['preprocessed_mac_csv']
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # preprocessed csv file
    preprocessed_mac_csv = data_root / processed_data / preprocessed_mac_csv_filename

    # read data
    df = read_csv(preprocessed_mac_csv)
    
    # data splitter
    data_splitter = DataSplitter(config, preprocessed_mac_csv)
    
    # split data into train and val sets
    train_df, val_df = data_splitter.split()
    
    # Load BERT tokenizer and model
    # tokenizer, model = get_model_tokenizer('bert-base-multilingual-cased')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    # train dataset
    train_data = SentimentDataset(train_df, tokenizer)
    
    # valid dataset
    val_data = SentimentDataset(val_df, tokenizer)

    # training args
    training_args = TrainingArguments(
        output_dir="output_bert",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        seed=0,
        save_strategy = "epoch"
    )

    # useful variables
    data_root = Path(config['data']['root'])
    processed_data = config['data']['processed']
    preprocessed_mac_csv_filename = config['data']['preprocessed_mac_csv']

    # batch size
    batch_size = int(config['params']['batch_size'])

    # preprocessed csv file
    preprocessed_mac_csv = data_root / processed_data / preprocessed_mac_csv_filename

    # read data
    df = read_csv(preprocessed_mac_csv)

    # data splitter
    data_splitter = DataSplitter(config, preprocessed_mac_csv)

    # split data into train and val sets
    train_df, val_df = data_splitter.split()

    # Load BERT tokenizer and model
    # tokenizer, model = get_model_tokenizer('bert-base-multilingual-cased')

    # train dataset
    train_data = SentimentDataset(train_df, tokenizer)

    # evaluation dataset
    eval_dataset = SentimentDataset(val_df, tokenizer)

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset = eval_dataset,
    )

    # train
    trainer.train()

    # evaluate
    trainer.evaluate()
