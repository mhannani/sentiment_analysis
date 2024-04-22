from typing import Optional, Tuple, Union
from regex import F
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig

# TODO :https://vscode.dev/github/mhannani/sentiment_analysis/blob/main/venv/Lib/site-packages/transformers/models/bert/modeling_bert.py#L1494
# Try to subclass the BertForSequenceClassification class and use it intead of nn.Module

class CustomClassifier(nn.Module):
    """Custom classifier for sentiment analysis"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_prob: float) -> None:
        """Class constructor for the custom classifier
        
        Args:
            input_size (int): Input size of the classifier (output size of BERT pooler)
            hidden_size (int): Hidden size of the classifier
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
        """
        super(CustomClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the custom classifier
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        return self.classifier(x)
    

class SentimentClassifier(nn.Module):
    """Sentiment classifier using BERT"""
    
    def __init__(self, config: object, bert: BertModel, *args, **kwargs) -> None:
        """class constructor of the classifier

        Args:
            config (object): configuration object
            bert (BertModel): BERT model to be fine-tuned for sequence classification
        """
        
        # call parent's __init__ method
        super().__init__(*args, **kwargs)
        
        # configuration object
        self.config = config

        # classifier's hiddent state
        self.hidden_size = int(config['params']['hidden_size'])
        
        # dropout
        self.dropout_prob = float(config['params']['dropout_prob'])

        # number of classification class
        self.num_classes = int(config['params']['num_classes'])

        # BERT model
        self.bert: BertModel = bert

        # freeze all layer of bert model
        self._freeze_all_layers()

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
        if hasattr(self.bert, 'classifier'):
            # the model has already the (classifier layer, preceded by dropout)
            # we will simply replace the existin classifier with out custom one
            
            # override the existing classifier
            self.bert.classifier = CustomClassifier(input_size=self.bert.config.hidden_size, hidden_size=self.hidden_size,
                                                    num_classes=self.num_classes, dropout_prob=self.dropout_prob)
            # forward pass
            return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


        # add dropout and the classifier
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

            # Check if the BERT model's output contains pooler output
            if 'pooler_output' not in outputs:
                raise ValueError("The BERT model's output does not contain 'pooler_output'. Ensure that the BERT model is configured to return the pooler output.")

            # Retrieve the pooler output
            pooler_out = outputs.pooler_output

            # Apply dropout
            pooler_out = nn.Dropout(self.dropout_prob)(pooler_out)

            # custom classifier head
            classifier = CustomClassifier(input_size=self.bert.config.hidden_size,hidden_size=self.hidden_size,num_classes=self.num_classes,dropout_prob=self.dropout_prob)

            # Feed the pooled output through the classifier head
            logits = classifier(pooler_out)

            # define losse function for multi-label classification
            loss_fct = nn.CrossEntropyLoss()
            
            # compute the loss
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

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
