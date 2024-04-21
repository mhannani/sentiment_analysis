import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertModel


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
        
        # dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # relu activation function
        self.relu =  nn.ReLU()
        
        # classifier head
        self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, self.hidden_size),
                nn.ReLU(),
                self.dropout,
                nn.Linear(self.hidden_size, self.num_classes)
            )

        # override the existing classification head
        if isinstance(self.bert, BertForSequenceClassification):
            # the model has already the (classifier layer, preceded by dropout)
            # we will simply replace the existin classifier with out custom one
            
            # override the existing classifier
            self.bert.classifier = self.classifier

    def _freeze_all_layers(self) -> None:
        """Freeze all layers in the BERT model."""
        
        # Loop over all parameters
        for param in self.bert.parameters():
            
            # disanle gradient computing
            param.requires_grad = False
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the classifier.

        Args:
            input_ids (torch.Tensor): Token ids of input sequences
            attention_mask (torch.Tensor): attention mask
        
        Returns:
            logits (torch.Tensor): Logits for classification
        """
        
        # forward pass for backbone/bert or classifier(with a custom classifier) model
        outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        
        if isinstance(self.bert, BertForSequenceClassification):
            return outputs

        if isinstance(self.bert, BertModel):

            # pooler output
            pooler_out = outputs.pooler_output
            
            # Apply dropout
            pooler_out = self.dropout(pooler_out)
            
            # Feed the pooled output through the classifier head
            logits = self.classifier(pooler_out)

            return logits


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
    
    model_ids = ["bert-base-multilingual-cased", "bert-base-arabic", "darijabert-arabizi", "DarijaBERT", 
                 "bert-base-arabertv2", "bert-base-arabic-finetuned-emotion"]
    
    for model_id in model_ids:
        print(f"+ model: {model_id}")
        _, model = get_model_tokenizer(model_id)
        
        # build custom model using the selected pretrained model
        custom_sentiment = SentimentClassifier(config, model)
        
        # forward pass
        out = custom_sentiment(input_ids=input_ids, attention_mask=attention_mask)
    
        print(out)
        print("====================================")
