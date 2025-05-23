from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from dotenv import load_dotenv
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel
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
        super().__init__(model_adapter, "dpo")
        self.dpo_config = dpo_config or DPOConfig()
        self.logger = logging.getLogger(__name__)

    
    def get_name(self):
        return self.training_method
    
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig, base_path: Path) -> PreTrainedModel:
        """Fine-tune a model using DPO

         Args:
            dataset_processor (DatasetProcessor): dataset to be fine tuned on, wrapped into a DataProcessor
            config (TrainingConfig): TrainingConfig for fine-tuning
            base_path (Path): local base path where checkpoints, results etc. are stored.

        Returns:
            PreTrainedModel: final fine-tuned model
        """
        self.logger.info("Starting DPO fine-tuning preparations...")
        
        # 0) set training run name 
        name = f"{self.model_adapter.get_name()}-{dataset_processor.get_name()}-{self.training_method}"
        
        # 1) get training data
        train_dataset = dataset_processor.get_dpo_dataset()
    
        # 2) load model and tokenizer
        model = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer() 
        
        # 3) Create reference model (frozen copy of the original model)
        ref_model = self.model_adapter.load_model()
        ref_model.eval()
        self.logger.info("Created reference model for DPO")
    
        
        # 4) Configure DPO trainer
        dpo_trainer_config = TRLDPOConfig(
            output_dir=config.checkpoint_config.checkpoint_dir.format(base= base_path),
            learning_rate=self.dpo_config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            fp16=config.fp16,
            push_to_hub=config.checkpoint_config.push_to_hub,
            hub_model_id=f"{HF}/{name}",
            hub_strategy=config.checkpoint_config.hub_strategy,
            save_steps=config.checkpoint_config.save_steps,
            save_total_limit=config.checkpoint_config.save_total_limit,
            save_strategy=config.checkpoint_config.save_strategy,
            seed=config.seed,
            beta=self.dpo_config.beta,
            label_smoothing=self.dpo_config.label_smoothing,
            loss_type=self.dpo_config.loss_type,
            label_pad_token_id=self.dpo_config.label_pad_token_id,
            logging_dir=f"{config.checkpoint_config.checkpoint_dir}/logs",
            logging_steps=10,
            report_to=config.report_to,
            run_name=str(time.time())+ name,
            gradient_accumulation_steps=2, 
            gradient_checkpointing=config.gradient_checkpointing
        )
        
        # 5) Initialize DPO trainer
        self.logger.info("Initializing DPOTrainer for DPO fine-tuning.")
        dpo_trainer = DPOTrainer(
          model= model,
          ref_model=ref_model,
          args=dpo_trainer_config,
          train_dataset=train_dataset,
          processing_class=tokenizer, 
      )
        
        # 6) Train the model
        self.logger.info("Starting DPO fine-tuning training...")
        dpo_trainer.train()
        self.logger.info("Finished DPO fine-tuning training. Saving model now...")
        
        ## 7) Save with config metadata for traceability
        save_dir = f"{base_path}/{config.checkpoint_config.checkpoint_dir}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save dpo config for reference
        with open(f"{save_dir}/dpo_config.txt", "w") as f:
            for key, value in vars(self.dpo_config).items():
                f.write(f"{key}: {value}\n")
                
        # Save model
        dpo_trainer.save_model(save_dir)
        
        # 8) push final model to hub
        if config.checkpoint_config.push_to_hub:
            dpo_trainer.push_to_hub()
        
        
        # 9) save locally meta data  
        self.save_training_metadata(config.checkpoint_config.checkpoint_dir, self.model_adapter.get_name())
        
        self.logger.info("Model and metadata saved. Successfully trained model.")
        
        return model
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> PreTrainedModel:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        self.logger.info("Loading base model and checkpoint adapter.")
        return  AutoModel.from_pretrained(checkpoint_path)