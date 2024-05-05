from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from typing import Tuple


def get_model_tokenizer(model_id: str, only_tokenizer = True, num_classes: int = 3) -> Tuple:
    """Gets model and the corresponding tokenizer

    Args:
        model_id (str): _description_
        only_tokenizer (bool): return only the tokenizer
        num_classes (int): number of classes for the classifier head

    Returns:
        Tuple[]: _description_
    """
    
    """
        - transformers.models.bert.modeling_bert.BertModel (pooler as the last layer in the model) with BertModel.from_pretrained(...)
                output of the model = BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state, hidden_states=None, 
                                    past_key_values=None, attentions=None, cross_attentions=None)
        
        - transformers.models.bert.modeling_bert.BertForSequenceClassification (the model has already the classifier layer)
        - 
    """
    model_id_mapping = {
        "bert-base-multilingual-cased": 'google-bert/bert-base-multilingual-cased',
        "bert-base-arabic": "asafaya/bert-base-arabic",
        "darijabert-arabizi": "SI2M-Lab/DarijaBERT-arabizi",
        "DarijaBERT": "SI2M-Lab/DarijaBERT",
        "bert-base-arabertv2": "aubmindlab/bert-base-arabertv2",
    }
    
    if model_id in model_id_mapping.keys():
        if only_tokenizer:
            return AutoTokenizer.from_pretrained(model_id_mapping[model_id])
        return AutoTokenizer.from_pretrained(model_id_mapping[model_id]), AutoModelForSequenceClassification.from_pretrained(model_id_mapping[model_id], num_labels=num_classes)

    else:
        raise ValueError(f"Model ID '{model_id}' not found in the mapping. Add it or use another one")


if __name__ == "__main__":
    """ Testing purposes """
    
    model_id_mapping = {
        "bert-base-multilingual-cased": 'google-bert/bert-base-multilingual-cased',
        "bert-base-arabic": "asafaya/bert-base-arabic",
        "darijabert-arabizi": "SI2M-Lab/DarijaBERT-arabizi",
        "DarijaBERT": "SI2M-Lab/DarijaBERT",
        "bert-base-arabertv2": "aubmindlab/bert-base-arabertv2",
        "bert-base-arabic-finetuned-emotion": "hatemnoaman/bert-base-arabic-finetuned-emotion"
    }
    
    for key, value in model_id_mapping.items():
        _, model = get_model_tokenizer(key)