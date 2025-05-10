from typing import Any, Dict, Optional

from safe_llm_finetune.fine_tuning.base import CheckpointConfig, FineTuningMethod, TrainingConfig
from safe_llm_finetune.fine_tuning.checkpoint import CheckpointManager
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from datasets import Dataset, load_dataset
from safe_llm_finetune.datasets.base import DatasetProcessor

import os

HF = os.getenv("HF")


class LoRAConfig:
    """Configuration for LoRA Supervised Fine-Tuning."""
    def __init__(
        self,
        r: int = 8,  # Rank
        alpha: int = 8,  # Alpha scaling
        target_modules: Optional[list] = None,  # Target modules for LoRA
        lora_dropout: float = 0.1,  # LoRA dropout
        bias: str = "none",  # Bias setting ("none", "all", "lora_only")
        task_type: TaskType = TaskType.CAUSAL_LM,  # Task type
        modules_to_save: Optional[list] = None,  # Modules to save (e.g., classifier head)
        max_length: int = 512  # Maximum sequence length
    ):
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.modules_to_save = modules_to_save
        self.max_length = max_length


class LoRAFineTuning(FineTuningMethod):
    """LoRA fine-tuning method implementation."""
    
    def __init__(self, model_adapter, lora_config: Optional[LoRAConfig] = None):
        super().__init__(model_adapter)
        self.lora_config = lora_config or LoRAConfig()
    
    
    
    
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig, identifier: Optional[str]= None) -> PreTrainedModel:
        """
        Train a model using LoRA for Supervised Fine-Tuning.

        Args:
            dataset (DataProcessor): Dataprocessor instance
            config (TrainingConfig): Training configuration containing checkpoint settings

        Returns:
            PreTrainedModel: fine-tuned model with LoRA for SFT
        """
        # 0) set training run name 
        name = HF + "/" + self.model_adapter.get_name() + "/" + dataset_processor.get_name()+ "/LoRA"+ identifier
        
        # 1) get dataset
        train_data = dataset_processor.get_sft_dataset()
        
        # 2) load model and tokenizer
        model = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer()
        
        # 3) Setup LoRA on the model
        peft_model = get_peft_model(model, self.lora_config)
        
        
        
        # 4) Configure training arguments
        training_args = SFTConfig(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            fp16=config.fp16,
            push_to_hub=config.checkpoint_config.push_to_hub,
            hub_model_id= name,
            hub_strategy=config.checkpoint_config.hub_strategy,
            save_steps=config.checkpoint_config.save_steps,
            save_total_limit=config.checkpoint_config.save_total_limit,
            save_strategy=config.checkpoint_config.save_strategy,
            seed=config.seed,
            logging_dir=f"{config.checkpoint_config.checkpoint_dir}/logs",
            logging_steps=10,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            optim=config.optim
        )
        
        # 5) Initialize trainer
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer
        )
        
        # 6) Train the model
        trainer.train()

        
        return peft_model