from typing import Optional, Tuple, Union
from regex import F
import torch
import numpy as np
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig

import lightning as L
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics.functional import accuracy, f1_score, precision, recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# TODO :https://vscode.dev/github/mhannani/sentiment_analysis/blob/main/venv/Lib/site-packages/transformers/models/bert/modeling_bert.py#L1494
# Try to subclass the BertForSequenceClassification class and use it intead of nn.Module


class ClassifierHead(L.LightningModule):
    """Classifier head for sentiment analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_prob: float) -> None:
        """Class constructor for the custom classifier
        
        Args:
            input_dim (int): Input dimension of the classifier (output size of BERT pooler)
            hidden_dim (int): Hidden dimension of the classifier
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
        """
        
        # call the superclass __init__ method
        super(ClassifierHead, self).__init__()
        
        # hidden size
        self.hidden_dim = hidden_dim
        
        # number of classes
        self.num_classes = num_classes
        
        # dropout probability
        self.dropout_prob = dropout_prob
        
        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )
        
        # class label order
        self.type_labels = [0, 1, 2]
        
        # averaging. set to 'macro'
        self.average = 'macro'

        # save parameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the custom classifier
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """

        # feed-forward pass
        return self.classifier(x)
    

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Single training step
        
        Args:
            batch (torch.Tensor): Batch tensor
            batch_idx (int): the batch index
            
        Returns:
            torch.Tensor: the loss value
        """
        
        # destruct the batch to features and targets
        x, y = batch
        
        # feed-forward pass through the networ
        y_hat = self(x)
        
        # compute the loss
        loss = nn.CrossEntropyLoss()(y_hat, y)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        
        return loss
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure ptimizer and learning-rate schedulers for the optimization.
        
        Return 
        """

        # configure the optimizer and return it
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def _shared_eval_step(self, batch: torch.tensor, batch_idx: int) -> any:
        """Shared eval step

        Args:
            batch (torch.tensor): The output of your data iterable.
            batch_idx (int): The index of the batch.

        Returns:
            any: Any for now. We don't know what to do right now.
        """
        # destruct the batch to features and targets
        x, y = batch
        
        # inferencing
        y_hat = self(x)
        
        # compute the loss
        loss = nn.CrossEntropyLoss()(y_hat, y)
        
        # convert torch tensors to numpy
        y_hat_np = torch.argmax(y_hat, axis=1).cpu().numpy()
        y_np = y.cpu().numpy()

        # compute the accuracy
        accuracy = accuracy_score(y_np, y_hat_np)
        
        recall = recall_score(y_np, y_hat_np, average = self.average, labels = self.type_labels)
        
        precision = precision_score(y_np, y_hat_np, average = self.average, labels = self.type_labels)
        
        f1_score_score = f1_score(y_np, y_hat_np, average = self.average, labels = self.type_labels)

        return loss, accuracy, precision, recall, f1_score_score
    
    def test_step(self, batch: torch.tensor, batch_idx: int) -> any:
        """Test step

        Args:
            batch (torch.tensor): The output of your data iterable.
            batch_idx (int): The index of the batch.

        Returns:
            any: Any for now. We don't know what to do right now.
        """
        
        # compute the loss and the accuract
        loss, accuracy, precision, recall, f1_score_score = self._shared_eval_step(batch, batch_idx)
        
        # construc the metrics
        metrics = {"accuracy_score": accuracy, 
                   "precision_score": precision, 
                   "recall_score": recall, 
                   "f1_score_score": f1_score_score, 
                   "test_loss": loss
                }

        # log a dictionary of values
        self.log_dict(metrics)

        return metrics


class SentimentClassifier(nn.Module):
    """Sentiment classifier using BERT"""
    
    def __init__(self, toml_config: object, bert: BertModel, retrain_classifier_head: bool = False, *args, **kwargs) -> None:
        """class constructor of the classifier

        Args:
            toml_config (object): TOML configuration object
            bert (BertModel): BERT model to be fine-tuned for sequence classification
            retrain_classifier_head(bool): Whether to retrain the BERT's classifier head(and obviously freeze the backbone)
        """
        
        # call parent's __init__ method
        super().__init__(*args, **kwargs)
        
        # toml configuration object
        self.toml_config = toml_config

        # classifier's hiddent state
        self.hidden_size = int(toml_config['params']['hidden_size'])
        
        # dropout
        self.dropout_prob = float(toml_config['params']['dropout_prob'])

        # number of classification class
        self.num_classes = int(toml_config['params']['num_classes'])

        # freeze backbone
        self.retrain_classifier_head = retrain_classifier_head

        # BERT model
        self.bert: BertModel = bert

        # BERT's config
        self.config = self.bert.config

        # freeze all layer of bert model
        if self.retrain_classifier_head:
            # we need to freeze the whole network. then we override the extisting(frozen) classifier with 
            # out custom one
            self._freeze_all_layers()
        
            # override the last layer of BERT model(the classifier head)
            self.bert.classifier = nn.Linear(self.config.hidden_size, self.num_classes)
            
    def _freeze_all_layers(self) -> None:
        """Freeze all layers in the BERT model."""
        
        # Loop over all parameters
        for param in self.bert.parameters():
            
            # disable gradient computing
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass for the classifier.

        Args:
            input_ids (torch.Tensor): Token ids of input sequences
            attention_mask (torch.Tensor): attention mask
        
        Returns:
            logits (torch.Tensor): Logits for classification
        """
        
        # Forward pass through the BERT model - Almost all models in our problem are based on BertForSequenceClassification
        # if hasattr(self.bert, 'classifier'):
        # the model has already the (classifier layer, preceded by dropout)
        # we will simply replace the existin classifier with out custom one(done in the class constructor)

        # forward pass
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


        # # add dropout and the classifier
        # else:
        #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        #     # Check if the BERT model's output contains pooler output
        #     if 'pooler_output' not in outputs:
        #         raise ValueError("The BERT model's output does not contain 'pooler_output'. Ensure that the BERT model is configured to return the pooler output.")

        #     # Retrieve the pooler output
        #     pooler_out = outputs.pooler_output

        #     # Apply dropout
        #     pooler_out = nn.Dropout(self.dropout_prob)(pooler_out)

        #     # custom classifier head
        #     classifier = CustomClassifier(input_size=self.bert.config.hidden_size,hidden_size=self.hidden_size,num_classes=self.num_classes,dropout_prob=self.dropout_prob)

        #     # Feed the pooled output through the classifier head
        #     logits = classifier(pooler_out)

        #     # define losse function for multi-label classification
        #     loss_fct = nn.CrossEntropyLoss()
            
        #     # compute the loss
        #     loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                
        #     return SequenceClassifierOutput(
        #         loss=loss,
        #         logits=logits,
        #         hidden_states=outputs.hidden_states,
        #         attentions=outputs.attentions,
        #     )

# class CustomBertForSequenceClassification(BertForSequenceClassification):
#     """ Custom Bert for the classification task"""

#     def __init__(self, toml_config: object, bert: BertModel, *args, **kwargs) -> None:
#         """class constructor of the classifier

#         Args:
#             toml_config (object): TOML configuration object
#             bert (BertModel): BERT model to be fine-tuned for sequence classification
#         """
        
#         # toml configuration object
#         self.toml_config = toml_config

#         # bert configuration
#         self.config: BertConfig = bert.config
        
#         # call the parent's __init__ method
#         super().__init__(self.config)

#         # number of labels
#         self.num_labels = int(self.toml_config['params']['num_classes'])
        
#         # dropout probability
#         self.config.classifier_dropout = float(self.toml_config['params']['dropout_prob'])
        
#         # bert model
#         self.bert = bert
        
#         # toml configuration object
#         self.toml_config = toml_config

#         # classifier's hiddent state
#         self.hidden_size = int(self.toml_config['params']['hidden_size'])
        
#         # dropout
#         self.dropout_prob = float(self.toml_config['params']['dropout_prob'])

#         # number of classification class
#         self.num_classes = int(self.toml_config['params']['num_classes'])
        
#         # custom classifier
#         # self.classifier = CustomClassifier(input_size=self.bert.config.hidden_size, hidden_size=self.hidden_size,
#         #                                             num_classes=self.num_labels, dropout_prob=self.config.classifier_dropout)
#         # Initialize weights and apply final processing
#         self.post_init()


if __name__ == "__main__":
    """ For test purposes """
    
    import os
    import sys

    # Get the path to the directory containing this script (src/prompts)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    sys.path.append(os.path.abspath(os.path.join(script_dir, "../..")))

    from pathlib import Path
    
    from src.utils.parsers import parse_toml
    from src.utils.get import get_model_tokenizer

    # Change the working directory to the root directory of the project
    os.chdir("../..")

    # Load the configuration from the TOML file
    config = parse_toml(Path("./configs/config.toml"))

    # Example random input data
    input_ids = torch.randint(0, 10000, (3, 20))
    attention_mask = torch.randint(0, 2, (3, 20))
    
    # model_ids = ["bert-base-multilingual-cased", "bert-base-arabic", "darijabert-arabizi", "DarijaBERT", 
    #              "bert-base-arabertv2", "bert-base-arabic-finetuned-emotion"]
    
    # for model_id in model_ids:
    #     print(f"+ model: {model_id}")
    #     _, model = get_model_tokenizer(model_id)
        
    #     # build custom model using the selected pretrained model
    #     custom_sentiment = SentimentClassifier(config, model)
        
    #     # forward pass
    #     out = custom_sentiment(input_ids=input_ids, attention_mask=attention_mask)
    
    #     print(out)
    #     print("====================================")
        
    #     print(model(input_ids=input_ids, attention_mask=attention_mask))
