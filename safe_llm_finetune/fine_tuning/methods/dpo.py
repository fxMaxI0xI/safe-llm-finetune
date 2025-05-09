import torch
from typing import Any, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOTrainer, DPOConfig as TRLDPOConfig
from datasets import Dataset, load_dataset
from dataclasses import dataclass

from base import FineTuningMethod, TrainingConfig
from checkpoint import CheckpointManager

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
    
    def prepare_dataset(self, dataset: Any, prompt_column: str = 'prompt', 
                   chosen_column: str = 'chosen', rejected_column: str = 'rejected',
                   split: str = 'train') -> Dataset:
        """
        Prepare dataset for DPO training.
        
        Args:
            dataset: Either a dataset name (str) from HF Hub, dict, or Dataset object
            prompt_column: Name of the column containing prompts (default: 'prompt')
            chosen_column: Name of the column containing chosen responses (default: 'chosen')
            rejected_column: Name of the column containing rejected responses (default: 'rejected')
            split: Dataset split to load (default: 'train')
        
        Returns:
            Dataset object with properly mapped columns for DPO training
        """
        # Load dataset if a string (HF dataset name) is provided
        if isinstance(dataset, str):
            try:
                loaded_dataset = load_dataset(dataset, split=split)
            except Exception as e:
                raise ValueError(f"Failed to load dataset '{dataset}' from Hugging Face Hub: {str(e)}")
        elif isinstance(dataset, dict):
            loaded_dataset = Dataset.from_dict(dataset)
        elif isinstance(dataset, Dataset):
            loaded_dataset = dataset
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}. Expected str (HF dataset name), dict, or Dataset.")
        
        # Check if all required columns exist in the dataset
        available_columns = set(loaded_dataset.column_names)
        required_mappings = {
            prompt_column: 'prompt',
            chosen_column: 'chosen', 
            rejected_column: 'rejected'
        }
        
        missing_columns = []
        for source_col, target_col in required_mappings.items():
            if source_col not in available_columns:
                missing_columns.append(source_col)
        
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}. "
                        f"Available columns: {list(available_columns)}")
        
        # Rename columns to the standard format if needed
        rename_dict = {}
        for source_col, target_col in required_mappings.items():
            if source_col != target_col:
                rename_dict[source_col] = target_col
        
        if rename_dict:
            loaded_dataset = loaded_dataset.rename_columns(rename_dict)
        
        return loaded_dataset
    
    def train(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, 
          dataset: Any, config: TrainingConfig, 
          prompt_column: str = 'prompt', chosen_column: str = 'chosen', 
          rejected_column: str = 'rejected', split: str = 'train') -> PreTrainedModel:
        """Train a model usig DPO.

        Args:
            model (PreTrainedModel): pretrained model to be fine-tuned with DPO
            tokenizer (PreTrainedTokenizer): tokenizer of pretrained model
            dataset (Any): Either a dataset name (str) from HF Hub, dict, or Dataset object
            config (TrainingConfig): Training configuration containing among other things checkpoint config. Since DPO uses very small learning rates this is defaulted in the DPO config

        Returns:
            PreTrainedModel: fine-tuned model
        """
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(
        dataset,
        prompt_column=prompt_column,
        chosen_column=chosen_column,
        rejected_column=rejected_column,
        split=split
    )
        
        # Create reference model (frozen copy of the original model)
        ref_model = self.model_adapter.load_model(model.config._name_or_path)
        ref_model.eval()
        
    
        
        # Configure DPO trainer
        dpo_trainer_config = TRLDPOConfig(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            learning_rate=self.dpo_config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            fp16=config.fp16,
            save_steps=config.checkpoint_config.save_steps,
            save_total_limit=config.checkpoint_config.save_total_limit,
            save_strategy=config.checkpoint_config.save_strategy,
            seed=config.seed,
            beta=self.dpo_config.beta,
            label_smoothing=self.dpo_config.label_smoothing,
            loss_type=self.dpo_config.loss_type,
            label_pad_token_id=self.dpo_config.label_pad_token_id,
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
          model=model,
          ref_model=ref_model,
          args=dpo_trainer_config,
          train_dataset=train_dataset,
          processing_class=tokenizer, 
      )
        
        # Train the model
        dpo_trainer.train()
        
        return model