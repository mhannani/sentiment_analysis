from typing import Callable, Dict, List, Tuple, Any
from torch._tensor import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments


class BertTrainer(Trainer):
    """Bert model Trainer"""
    
    def __init__(self, model: PreTrainedModel | Module = None, args: TrainingArguments = None, 
                 data_collator: Any | None = None, train_dataset: Dataset | IterableDataset | Any | None = None, 
                 eval_dataset: Dataset | Dict[str, Dataset] | Any | None = None, 
                 tokenizer: PreTrainedTokenizerBase | None = None, model_init: Callable[[], PreTrainedModel] | None = None, 
                 compute_metrics: Callable[[EvalPrediction], Dict] | None = None, callbacks: List[TrainerCallback] | None = None,
                 optimizers: Tuple[Optimizer, LambdaLR] = ..., 
                 preprocess_logits_for_metrics: Callable[[Tensor, Tensor], Tensor] | None = None):
        
        # call parent's __init__ function
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, 
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
