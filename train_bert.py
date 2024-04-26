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


os.environ["HF_HOME"] = "/netscratch/mhannani/.cashe_hg"

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

    # list of models to train
    model_id_mapping = {
        "bert-base-multilingual-cased": 'google-bert/bert-base-multilingual-cased',
        "bert-base-arabic": "asafaya/bert-base-arabic",
        "darijabert-arabizi": "SI2M-Lab/DarijaBERT-arabizi",
        "DarijaBERT": "SI2M-Lab/DarijaBERT",
        "bert-base-arabertv2": "aubmindlab/bert-base-arabertv2",
    }
    
    # preprocessed csv file
    preprocessed_mac_csv = data_root / processed_data / preprocessed_mac_csv_filename
    
    # data splitter
    data_splitter = DataSplitter(config, preprocessed_mac_csv)
    
    # split data into train and val sets
    train_df, val_df = data_splitter.split()

    # train all models
    for model_id in model_id_mapping.keys():
        
        comet_ml.init(project_name=f'freezed-backbone-fine-tuned-bert-id-{model_id}')
        
        print(f"\n --> Training {model_id} <-- ")

        # get the model and the tokenizer
        tokenizer, model = get_model_tokenizer(model_id)

        # customize the current model
        # model = SentimentClassifier(config, model)
        
        # --
        # print("get_model_trainable_layers(model): ", get_model_trainable_layers(model))

        # train dataset
        train_data = SentimentDataset(train_df, tokenizer)
        
        # valid dataset
        eval_data = SentimentDataset(val_df, tokenizer)

        # training args
        training_args = TrainingArguments(
            output_dir=f"/netscratch/mhannani/freezed_backbone_fine_tuned_bert/{model_id}",
            evaluation_strategy="epoch",
            do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=36,
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
        # trainer.evaluate()

