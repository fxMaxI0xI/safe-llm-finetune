import logging
import os
from pathlib import Path
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from transformers import BitsAndBytesConfig, PreTrainedModel
from trl import SFTConfig, SFTTrainer

from safe_llm_finetune.datasets.base import DatasetProcessor
from safe_llm_finetune.fine_tuning.base import FineTuningMethod, ModelAdapter, TrainingConfig

load_dotenv()
HF = os.getenv("HF")

class QLoRAConfig:
    """Configuration for Quantized LoRA Supervised Fine-Tuning."""
    def __init__(
        self,
        target_modules: Optional[list] = None,  # Target modules for qLoRA
        identifier: Optional[str] = None,  # Add identifier to config
        r: int = 8,  # Rank
        alpha: int = 16,  # Alpha scaling (typically higher for qLoRA)
        lora_dropout: float = 0.05,  # LoRA dropout
        bias: str = "none",  # Bias setting ("none", "all", "lora_only")
        task_type: TaskType = TaskType.CAUSAL_LM,  # Task type
        modules_to_save: Optional[list] = None,  # Modules to save
        max_length: int = 512,  # Maximum sequence length
        # qLoRA specific parameters
        bits: Literal[4, 8] = 4,  # Either 4-bit or 8-bit quantization
        use_double_quant: bool = True,  # Double quantization 
        quant_type: Literal["nf4", "fp4"] = "nf4",  # nf4 (normalized float 4) or fp4
        compute_dtype: torch.dtype = torch.float16,  # Computation precision

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
        # qLoRA specific
        self.bits = bits
        self.use_double_quant = use_double_quant
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
    
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
    
    def to_quant_config(self) -> BitsAndBytesConfig:
        """Convert to BitsAndBytesConfig"""
        return BitsAndBytesConfig(
            load_in_4bit=(self.bits == 4),
            load_in_8bit=(self.bits == 8),
            bnb_4bit_use_double_quant=self.use_double_quant,
            bnb_4bit_quant_type=self.quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype
        )
    
    def get_identifier(self) -> str:
        """Generate a standardized identifier for this configuration."""
        if self.identifier:
            return self.identifier
        
        # Create an identifier from parameters if not provided
        return f"r{self.r}_a{self.alpha}_d{self.lora_dropout}_b{self.bits}_{self.quant_type}"


class QLoRAFineTuning(FineTuningMethod):
    """qLoRA fine-tuning method implementation."""
    
    def __init__(self, model_adapter: ModelAdapter, qlora_config: Optional[QLoRAConfig] = None):
        super().__init__(model_adapter, "qlora")
        self.qlora_config = qlora_config or QLoRAConfig()
        self.logger = logging.getLogger(__name__)

        if not self.qlora_config.target_modules:
            self.qlora_config.target_modules = model_adapter.get_qlora_modules()
        
        self.training_method = self.qlora_config.get_identifier()

            
    def get_name(self):
        return self.training_method
        
    def train(self, dataset_processor: DatasetProcessor, config: TrainingConfig, base_path: Path) -> PreTrainedModel:
        """
        Train a model using qLoRA for Supervised Fine-Tuning.

        Args:
            dataset_processor (DatasetProcessor): Dataprocessor instance
            config (TrainingConfig): Training configuration containing checkpoint settings

        Returns:
            PreTrainedModel: fine-tuned model with qLoRA for SFT
        """
        self.logger.info("Starting qLoRA fine-tuning preparations...")

        # 0) set training run name 
        identifier = self.qlora_config.get_identifier()
        name = f"{self.model_adapter.get_name()}-{dataset_processor.get_name()}-{self.training_method}"
        
        # 1) get dataset
        train_data = dataset_processor.get_sft_dataset()
        
        # 2) load quantized model and tokenizer
        quant_config = self.qlora_config.to_quant_config()
        model = self.model_adapter.load_quantized_model(quant_config)
        tokenizer = self.model_adapter.load_tokenizer()
        
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        
        # 3) Setup LoRA on the quantized model using PEFT's LoraConfig
        peft_config = self.qlora_config.to_peft_config()
        peft_model = get_peft_model(model, peft_config)
        
        # Log trainable parameters vs total parameters
        trainable_params, all_params = self._get_parameter_counts(peft_model)
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params / all_params:.2%} of total)")
        
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        
        # 4) Configure training arguments
        training_args = SFTConfig(
            output_dir=config.checkpoint_config.checkpoint_dir.format(base= base_path),
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            fp16=config.fp16,
            bf16=True, 
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
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,  
            optim=config.optim,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            max_seq_length=self.qlora_config.max_length,
            report_to=config.report_to,
            run_name=str(time.time())+ name
        )
        
        # 5) Initialize trainer
        self.logger.info("Initializing SFTTrainer for qLoRA fine-tuning.")
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_data,
            processing_class=tokenizer,  # Using SFTTrainer's expected argument
        )
        
        # 6) Train the model
        self.logger.info("Starting qLoRA fine-tuning training...")
        trainer.train()
        self.logger.info("Finished qLoRA fine-tuning training. Saving model now...")
        
        # 7) Save with config metadata for traceability
        save_dir = f"{base_path}/{config.checkpoint_config.checkpoint_dir}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save qLoRA config for reference
        with open(f"{save_dir}/qlora_config.txt", "w") as f:
            for key, value in vars(self.qlora_config).items():
                f.write(f"{key}: {value}\n")
                
        # Save model
        trainer.save_model(save_dir)
        
        # 8) push final model to hub
        if config.checkpoint_config.push_to_hub:
            trainer.push_to_hub()
        
        
        # 9) save locally meta data  
        self.save_training_metadata(config.checkpoint_config.checkpoint_dir, self.model_adapter.get_name())
        
        self.logger.info("Model and metadata saved. Successfully trained model.")
        
        return peft_model
    
    def _get_parameter_counts(self, model):
        """Helper method to get trainable vs total parameter counts"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        return trainable_params, all_params
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> PreTrainedModel:
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        self.logger.info("Loading base model and checkpoint adapter.")
        quant_config = self.qlora_config.to_quant_config()
        base_model = self.model_adapter.load_quantized_model(quant_config)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        return model.merge_and_unload()
        