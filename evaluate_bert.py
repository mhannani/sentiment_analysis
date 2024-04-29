import os
from pathlib import Path
from pandas import read_csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from pandas import read_csv
import numpy as np
import warnings

from transformers import AutoModelForSequenceClassification

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
from src.utils.readers import read_df


os.environ["HF_HOME"] = "/netscratch/mhannani/.cashe_hg"

# Filter out UndefinedMetricWarning
warnings.filterwarnings("ignore")


def compute_metrics(p):

    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
      
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
    test_preprocessed_mac_csv_filename = config['data']['test_csv_filename']
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # preprocessed csv file
    test_preprocessed_mac_csv = data_root / processed_data / test_preprocessed_mac_csv_filename
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # list of models to train
    model_id_mappings = [
        "bert-base-multilingual-cased",
        # "bert-base-arabic", "darijabert-arabizi", "DarijaBERT", "bert-base-arabertv2",
    ]
    
    # data splitter
    test_df = read_df(test_preprocessed_mac_csv)

    # train all models
    for model_id in model_id_mappings:
        
        print(f"\n --> Evaluating {model_id} <-- ")

        # get the model and the tokenizer
        tokenizer, _ = get_model_tokenizer(model_id)

        # train dataset
        test_data = SentimentDataset(test_df, tokenizer)

        # training args
        training_args = TrainingArguments(
            output_dir=f"/netscratch/mhannani/__freezed_backbone_fine_tuned_bert/{model_id}",
            evaluation_strategy="epoch",
            do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=36,
            save_strategy = "epoch"
        )

        model = AutoModelForSequenceClassification.from_pretrained(f"/netscratch/mhannani/freezed_backbone_fine_tuned_bert/{model_id}", local_files_only=True, num_labels=3)
    
        # trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            eval_dataset = test_data,
            # callbacks=[PrintTrainLossCallback]
        )

        # train current model
        # trainer.train()
        trainer.evaluate()
