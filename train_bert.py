import comet_ml
import os
from pathlib import Path
from pandas import read_csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from pandas import read_csv
import numpy as np
import warnings
import argparse
import logging

from src.data.sentiment_data import SentimentDataset
from src.data.split import DataSplitter
from src.utils.callbacks import PrintTrainLossCallback
from src.utils.model import get_model_trainable_layers
from src.utils.parsers import parse_toml
from src.utils.get import get_model_tokenizer
from src.data.sentiment_data import SentimentDataset
from src.data.split import DataSplitter
from src.utils.get import get_model_tokenizer
from src.models.classifier import SentimentClassifier


# hugging face cache directory change
os.environ["HF_HOME"] = "/netscratch/mhannani/.cashe_hg"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter out UndefinedMetricWarning
warnings.filterwarnings("ignore")


def compute_metrics(p):
    experiment = comet_ml.get_global_experiment()

    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    
    if experiment:
      epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
      experiment.set_epoch(epoch)
      experiment.log_confusion_matrix(
          y_true=labels, 
          y_predicted=pred, 
          file_name=f"confusion-matrix-epoch-{epoch}.json", 
      )
      
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    """Train custom BERT model"""

    # Create argument parser
    parser = argparse.ArgumentParser(description="Script description")
    
    # Add argument for config file
    parser.add_argument("config_file", type=str, help="configuration filename")
    
    # Add argument for config file
    parser.add_argument("exp_name", type=str, help="experiment name")
    
    # training mode
    parser.add_argument("finetune", type=bool, help="Training mode. eg. finetune (True) or pre-train(False)")

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # configuration filepath
    CONFIG_FILE = Path(f"configs/{args.config_file}.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    processed_data = config['data']['processed']
    preprocessed_corpus_csv_filename = config['data']['preprocessed_corpus_csv']
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # train epoch
    num_epoch = int(config['params']['num_epoch'])

    # number of classes
    num_classes = int(config['params']['num_classes'])

    # list of models to train
    model_id_mapping = {
        "bert-base-multilingual-cased": 'google-bert/bert-base-multilingual-cased',
        "bert-base-arabic": "asafaya/bert-base-arabic",
        "darijabert-arabizi": "SI2M-Lab/DarijaBERT-arabizi",
        "DarijaBERT": "SI2M-Lab/DarijaBERT",
        "bert-base-arabertv2": "aubmindlab/bert-base-arabertv2",
    }
    
    # preprocessed csv file
    preprocessed_corpus_csv = data_root / processed_data / preprocessed_corpus_csv_filename
    
    # data splitter
    data_splitter = DataSplitter(config, preprocessed_corpus_csv)
    
    # split data into train and val sets
    train_df, val_df = data_splitter.split()

    # train all models
    for model_id in model_id_mapping.keys():
        
        comet_ml.init(project_name=f'{args.exp_name}-id-{model_id}')
        
        print(f"\n --> Training {model_id} model-experiment {args.exp_name}-finetune-{args.finetune} <-- ")

        # get the model and the tokenizer
        tokenizer, model = get_model_tokenizer(model_id, only_tokenizer=False, num_classes=num_classes)

        # count model trainable parameters
        model_trainable_params = get_model_trainable_layers(model)

        # train dataset
        train_data = SentimentDataset(train_df, tokenizer)

        # valid dataset
        eval_data = SentimentDataset(val_df, tokenizer)

        # training args
        training_args = TrainingArguments(
            output_dir=f"/netscratch/mhannani/experiment-{args.exp_name}/{model_id}trainable_params-{model_trainable_params}",
            evaluation_strategy="epoch",
            do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epoch,
            save_strategy = "epoch"
        )
    
        # trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_data,
            eval_dataset = eval_data,
            # callbacks=[PrintTrainLossCallback]
        )

        # train current model
        trainer.train()
