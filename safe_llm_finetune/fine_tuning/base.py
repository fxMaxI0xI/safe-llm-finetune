"""
Abstract base classes for fine-tuning methods and models.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig

from safe_llm_finetune.datasets.base import Dataset, DatasetProcessor


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""
    checkpoint_dir: Path
    save_steps: int = 500
    save_total_limit: int = 5
    save_strategy: str = "steps"
    hub_model_id: str
    hub_strategy : str ="checkpoint"
    push_to_hub : bool = True


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
    optim: str = "adamw_torch"
    
@dataclass
class BnBQuantizationConfig:
    bits: Literal[4, 8] = 4  # Either 4-bit or 8-bit quantization
    use_double_quant: bool = True  # Double quantization 
    quant_type: Literal["nf4", "fp4"] = "nf4"  # nf4 (normalized float 4) or fp4
    compute_dtype: torch.dtype = torch.float16  # Computation precision
    
    def __post_init__(self):
        # Validation
        if self.bits not in [4, 8]:
            raise ValueError("BitsAndBytes only supports 4-bit or 8-bit quantization")
        
        if self.bits == 4 and self.quant_type not in ["nf4", "fp4"]:
            raise ValueError("4-bit quantization only supports 'nf4' or 'fp4' quant_type")
    
    def to_bnb_config(self) -> BitsAndBytesConfig:
        """Convert to BitsAndBytesConfig object for HuggingFace Transformers"""
        return BitsAndBytesConfig(
            load_in_4bit=(self.bits == 4),
            load_in_8bit=(self.bits == 8),
            bnb_4bit_use_double_quant=self.use_double_quant,
            bnb_4bit_quant_type=self.quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype
        )




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
    def load_quantized_model(self, quantization_config: BnBQuantizationConfig) -> PreTrainedModel:
        """ Load a model from HuggingFace in specifiec quantization

        Args:
            quantization_config (BnBQuantizationConfig): config for quantization

        Returns:
            PreTrainedModel: model loaded in specified quantization
        """
        pass
    
    @abstractmethod
    def generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, **kwargs) -> str:
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
    
    