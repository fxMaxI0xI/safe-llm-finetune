import os
import time
from typing import Optional

from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel
from trl import SFTConfig, SFTTrainer

from safe_llm_finetune.datasets.base import DatasetProcessor
from safe_llm_finetune.fine_tuning.base import FineTuningMethod, ModelAdapter, TrainingConfig

# Load environment variables from .env file
load_dotenv()

# Now you can access the HF variable
HF = os.getenv("HF")

class LoRAConfig:
    """Configuration for LoRA Supervised Fine-Tuning."""
    def __init__(
        self,
        target_modules: Optional[list] = None,  # Target modules for LoRA
        identifier: Optional[str] = None,  # Add identifier to config
        r: int = 8,  # Rank
        alpha: int = 8,  # Alpha scaling
        lora_dropout: float = 0.1,  # LoRA dropout
        bias: str = "none",  # Bias setting ("none", "all", "lora_only")
        task_type: TaskType = TaskType.CAUSAL_LM,  # Task type
        modules_to_save: Optional[list] = None,  # Modules to save (e.g., classifier head)
        max_length: int = 512  # Maximum sequence length
    ):
        self.identifier = identifier
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules 
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.modules_to_save = modules_to_save
        self.max_length = max_length
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            modules_to_save=self.modules_to_save,
        )
    
    def get_identifier(self) -> str:
        """Generate a standardized identifier for this configuration."""
        if self.identifier:
            return self.identifier
        
        # Create an identifier from parameters if not provided
        return f"r{self.r}_a{self.alpha}_d{self.lora_dropout}"


class LoRAFineTuning(FineTuningMethod):
    """LoRA fine-tuning method implementation."""
    
    def __init__(self, model_adapter: ModelAdapter, lora_config: Optional[LoRAConfig] = None):
        super().__init__(model_adapter)
        self.lora_config = lora_config or LoRAConfig()
        if not self.lora_config.target_modules:
            self.lora_config.target_modules = model_adapter.get_lora_modules()
        
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig) -> PreTrainedModel:
        """
        Train a model using LoRA for Supervised Fine-Tuning.

        Args:
            dataset_processor (DatasetProcessor): Dataprocessor instance
            config (TrainingConfig): Training configuration containing checkpoint settings

        Returns:
            PreTrainedModel: fine-tuned model with LoRA for SFT
        """
        # 0) set training run name 
        identifier = self.lora_config.get_identifier()
        name = f"{self.model_adapter.get_name()}-{dataset_processor.get_name()}-LoRA-{identifier}"
        
        # 1) get dataset
        train_data = dataset_processor.get_sft_dataset()
        
        # 2) load model and tokenizer
        model = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer()
        
        # 3) Setup LoRA on the model using PEFT's LoraConfig
        peft_config = self.lora_config.to_peft_config()
        peft_model = get_peft_model(model, peft_config)
        
        # Log trainable parameters vs total parameters
        trainable_params, all_params = self._get_parameter_counts(peft_model)
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of total)")
        
        # 4) Configure training arguments
        training_args = SFTConfig(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
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
            logging_dir=f"{config.checkpoint_config.checkpoint_dir}/logs",
            logging_steps=10,
            remove_unused_columns=False,
            optim=config.optim,
            max_seq_length=self.lora_config.max_length,
            report_to=config.report_to,
            run_name=str(time.time())+ name
        )
        
        # 5) Initialize trainer
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,  # Updated to match SFTTrainer's expected argument
        )
        
        # 6) Train the model
        trainer.train()
        
        # 7) Save with config metadata for traceability
        save_dir = f"{config.checkpoint_config.checkpoint_dir}/{identifier}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save LoRA config for reference
        with open(f"{save_dir}/lora_config.txt", "w") as f:
            for key, value in vars(self.lora_config).items():
                f.write(f"{key}: {value}\n")
                
        # Save model
        trainer.save_model(save_dir)
        
        # 8) If pushed to hub, add metadata
        if config.checkpoint_config.push_to_hub:
            trainer.push_to_hub()
        
        return peft_model
    
    def _get_parameter_counts(self, model):
        """Helper method to get trainable vs total parameter counts"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        return trainable_params, all_params