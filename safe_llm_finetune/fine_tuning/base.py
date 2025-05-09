"""
Abstract base classes for fine-tuning methods and models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datasets import Dataset
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from safe_llm_finetune.datasets.base import DatasetProcessor

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""
    checkpoint_dir: Path
    save_steps: int = 500
    save_total_limit: int = 5
    save_strategy: str = "steps"


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    warmup_steps: int = 0
    weight_decay: float = 0.01
    fp16: bool = False
    checkpoint_config: Optional[CheckpointConfig] = None
    seed: int = 42



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
    
    def load_quantized_model(self, )
    
    @abstractmethod
    def generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
                 prompt: str, **kwargs) -> str:
        """
        Generate text using the model.
        
        Args:
            model: Model to use for generation
            tokenizer: Tokenizer to use for generation
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
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
    def train(self,
              dataset_processor: DatasetProcessor, config: TrainingConfig) -> PreTrainedModel:
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
    
    