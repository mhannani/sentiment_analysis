import os
from pathlib import Path
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
    """compute metrics

    Args:
        p (_type_): _description_

    Returns:
        _type_: _description_
    """

    pred, labels = p
    pred = np.argmax(pred, axis=1)
    
    # solve issue when the predicted sentence is neutral
    pred[pred == 1] = 2

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":

    """Evaluate custom BERT model"""

    # configration filepath
    CONFIG_FILE = Path("configs/myc_config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    processed_data = config['data']['processed']
    preprocessed_corpus_csv_filename = config['data']['preprocessed_corpus_csv']
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # preprocessed csv file
    test_preprocessed_corpus_csv = data_root / processed_data / preprocessed_corpus_csv_filename
    
    # batch size
    batch_size = int(config['params']['batch_size'])

    # list of models to train
    model_id_mappings = {
        "bert-base-multilingual-cased": "bert-base-multilingual-cased/checkpoint-17185",
        "bert-base-arabic": "bert-base-arabic/checkpoint-2946",
        "darijabert-arabizi": "darijabert-arabizi/checkpoint-10311",
        "DarijaBERT": "DarijaBERT/checkpoint-14239",
        "bert-base-arabertv2": "bert-base-arabertv2/checkpoint-8347",
    }

    # data splitter
    test_df = read_df(test_preprocessed_corpus_csv)

    # train all models
    for model_id in model_id_mappings.keys():
        
        print(f"\n --> Evaluating {model_id} <-- ")

        # get the model and the tokenizer
        tokenizer = get_model_tokenizer(model_id)

        print("tokenizer loaded")
        # train dataset 
        test_data = SentimentDataset(test_df, tokenizer)

        # output directory for the best checlpoint
        out_directory = f"/netscratch/mhannani/fine_tuned_bert/{model_id_mappings[model_id]}"
        
        # training args
        training_args = TrainingArguments(
            output_dir=out_directory,
            evaluation_strategy="epoch",
            do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=36,
            save_strategy = "epoch"
        )
    
        # model object
        model = AutoModelForSequenceClassification.from_pretrained(out_directory, local_files_only=True, num_labels=3)

        # trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            eval_dataset = test_data,
            # callbacks=[PrintTrainLossCallback]
        )
        
        # evaluate the model
        output = trainer.evaluate(eval_dataset = test_data)
        
        # print output
        print(output)
