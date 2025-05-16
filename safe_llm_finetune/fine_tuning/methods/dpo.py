from dataclasses import dataclass
import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer

from datasets import Dataset
from safe_llm_finetune.datasets.base import DatasetProcessor
from safe_llm_finetune.fine_tuning.base import FineTuningMethod, TrainingConfig
from safe_llm_finetune.fine_tuning.checkpoint import CheckpointManager

load_dotenv()
HF = os.getenv("HF")

@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1  # KL penalty coefficient
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # or "hinge"
    label_pad_token_id: int = -100
    learning_rate: float = 1e-6

class DPOFineTuning(FineTuningMethod):
    """DPO fine-tuning method implementation."""
    
    def __init__(self, model_adapter, dpo_config: Optional[DPOConfig] = None):
        super().__init__(model_adapter)
        self.dpo_config = dpo_config or DPOConfig()
    
    
    
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig) -> PreTrainedModel:
        """Train a model usig DPO.

        Args:
            model (PreTrainedModel): pretrained model to be fine-tuned with DPO
            tokenizer (PreTrainedTokenizer): tokenizer of pretrained model
            dataset (DatasetProcessor): DatasetProcessor Object of desired training dataset
            config (TrainingConfig): Training configuration containing among other things checkpoint config. Since DPO uses very small learning rates this is defaulted in the DPO config

        Returns:
            PreTrainedModel: fine-tuned model
        """
        
        # 0) set training run name 
        name = HF + "-" + self.model_adapter.get_name() + "-" + dataset_processor.get_name()+ "-DPO"
        
        # 1) get training data
        train_dataset = dataset_processor.get_dpo_dataset()
    
        # 2) load model and tokenizer
        model = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer() 
        
        # 3) Create reference model (frozen copy of the original model)
        ref_model = self.model_adapter.load_model()
        ref_model.eval()
    
        
        # 4) Configure DPO trainer
        dpo_trainer_config = TRLDPOConfig(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            learning_rate=self.dpo_config.learning_rate,
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
            beta=self.dpo_config.beta,
            label_smoothing=self.dpo_config.label_smoothing,
            loss_type=self.dpo_config.loss_type,
            label_pad_token_id=self.dpo_config.label_pad_token_id,
            report_to=config.report_to,
            run_name=str(time.time())+ name,
            gradient_accumulation_steps=2, 
            gradient_checkpointing=config.gradient_checkpointing
        )
        
        # 5) Initialize DPO trainer
        dpo_trainer = DPOTrainer(
          model= model,
          ref_model=ref_model,
          args=dpo_trainer_config,
          train_dataset=train_dataset,
          processing_class=tokenizer, 
      )
        
        # 6) Train the model
        dpo_trainer.train()
        
        return model