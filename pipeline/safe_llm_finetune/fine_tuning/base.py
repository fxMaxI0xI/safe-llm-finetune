"""
Abstract base classes for fine-tuning methods and models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Optional, Union

from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from safe_llm_finetune.datasets.base import DatasetProcessor

logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""
    checkpoint_dir: str = "{base}/checkpoints"
    hub_model_id: str = None
    save_steps: int = 0.1
    save_total_limit: int = None
    save_strategy: str = "steps"
    hub_strategy : str ="all_checkpoints"
    push_to_hub : bool = True


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    warmup_steps: int = 0
    weight_decay: float = 0.01
    fp16: bool = False
    gradient_accumulation_steps=1
    checkpoint_config: Optional[CheckpointConfig] = None
    seed: int = 42
    optim: str = "adamw_torch"
    report_to: Optional[Union[str, list]] = None
    gradient_checkpointing : bool = True
    max_seq_length: int = 1024
    report_to: str = "wandb"
    run_name: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "fp16": self.fp16,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optim": self.optim,
            "report_to": self.report_to,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
        }



class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    def __init__(self, model_name):
       self.model_name = model_name
       self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        """
        Load a model from HuggingFace.
            
        Returns:
            Loaded model
        """
        pass
    
    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load a tokenizer from HuggingFace.
            
        Returns:
            Loaded tokenizer
        """
        pass
    
    
    @abstractmethod
    def load_quantized_model(self, quantization_config: BitsAndBytesConfig) -> PreTrainedModel:
        """ Load a model from HuggingFace in specifiec quantization

        Args:
            quantization_config (BnBQuantizationConfig): config for quantization

        Returns:
            PreTrainedModel: model loaded in specified quantization
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str :
        """
        Returns name of model
        """
        pass
    
    @abstractmethod
    def get_lora_modules(self) -> list[str]:
        """
        Returns the target modules for LoRA fine-tuning
        
        By default this returns only query and key
        """
        pass
    
    @abstractmethod
    def get_qlora_modules(self)-> list[str]:
        """
        Returns the target modules for QLoRA fine-tuning
        
        By default returns attention and mlp modules
        """
        pass
    
    @abstractmethod
    def get_available_modules(self) -> dict[str, list[str]]:
        """
        Returns all available modules that can be targeted for fine-tuning in Gemma
        """
        pass
    

class FineTuningMethod(ABC):
    """Abstract base class for fine-tuning methods."""
    
    def __init__(self, model_adapter: ModelAdapter, training_method: str):
        """
        Initialize fine-tuning method.
        
        Args:
            model_adapter: Model adapter to use
            training_method: which method the adapter is for
        """
        self.model_adapter = model_adapter
        self.training_method = training_method
    
    @abstractmethod
    def get_name(self) -> str:
        """Return name of training method
        """
        pass
    
    
    @abstractmethod
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig, base_path: Path) -> PreTrainedModel:
        """Fine tune the model

        Args:
            dataset_processor (DatasetProcessor): dataset to be fine tuned on, wrapped into a DataProcessor
            config (TrainingConfig): TrainingConfig for fine-tuning
            base_path (Path): local base path where checkpoints, results etc. are stored.

        Returns:
            PreTrainedModel: final fine-tuned model
        """
        pass
    
    def save_training_metadata(self, checkpoint_path: str, model_name: str, **kwargs):
        """
        Save metadata about the training method used.

        Args:
            checkpoint_path: Path to checkpoint
            model_name: Name of the base model
            **kwargs: Additional metadata to save
        """
        metadata = {
            "training_method": self.training_method,
            "base_model": model_name,
            **kwargs
        }
        
        metadata_path = Path(checkpoint_path) / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    @abstractmethod
    def load_model_from_checkpoint(self, checkpoint_path: str) -> PreTrainedModel:
        """loads a model (peft/full) from specified checkpoint

        Args:
            checkpoint_path (str): path to exact checkpoint

        Returns:
            PreTrainedModel: model from checkpoint, if peft, return merged model
        """
    
    
