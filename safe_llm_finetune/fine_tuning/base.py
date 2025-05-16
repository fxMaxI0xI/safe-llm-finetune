"""
Abstract base classes for fine-tuning methods and models.
"""
from typing import Optional, Union

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig

from safe_llm_finetune.datasets.base import DatasetProcessor


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""
    checkpoint_dir: Path
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
    
    # @abstractmethod
    # def generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, **kwargs) -> str:
    #     """
    #     Generate text using the model.
        
    #     Args:
    #         model: Model to use for generation
    #         tokenizer: Tokenizer to use for generation
    #         prompt: Input prompt
    #         **kwargs: Additional generation parameters
            
    #     Returns:
    #         Generated text
    #     """
    #     pass
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
    
    def __init__(self, model_adapter: ModelAdapter):
        """
        Initialize fine-tuning method.
        
        Args:
            model_adapter: Model adapter to use
        """
        self.model_adapter = model_adapter
    
    
    @abstractmethod
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig) -> PreTrainedModel:
        """
        Fine-tune the model.
        
        Args:
            model: Model to fine-tune
            tokenizer: Tokenizer to use
            dataset: Dataset processor to use for fine-tuning
            config: Training configuration
            
        Returns:
            Fine-tuned model
        """
        pass
    
    