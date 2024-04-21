from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from typing import Tuple


def get_model_tokenizer(model_id: str) -> Tuple:
    """Gets model and the corresponding tokenizer

    Args:
        model_id (str): _description_

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
        "bert-base-arabic-finetuned-emotion": "hatemnoaman/bert-base-arabic-finetuned-emotion"
    }
    
    model_id_for_masked_lm = ['bert-base-arabic', 'darijabert-arabizi', 'DarijaBERT', 'bert-base-arabertv2']
    model_id_for_seq_classification = ['bert-base-arabic-finetuned-emotion']
    model_id_bert = ['bert-base-multilingual-cased']

    model_id_repo = model_id_mapping[model_id]
    
    if model_id in model_id_for_masked_lm:
        return AutoTokenizer.from_pretrained(model_id_repo), AutoModelForSequenceClassification.from_pretrained(model_id_repo)
    
    elif model_id in model_id_bert:
        return BertTokenizer.from_pretrained(model_id_repo), BertModel.from_pretrained(model_id_repo)
    
    elif model_id in model_id_for_seq_classification:
        return AutoTokenizer.from_pretrained(model_id_repo), AutoModelForSequenceClassification.from_pretrained(model_id_repo)
        
    else:
        raise ValueError(f"Model ID '{model_id_repo}' not found in the mapping. Add it or use another one")


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