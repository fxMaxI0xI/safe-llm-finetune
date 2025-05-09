import torch
from typing import Any, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainerCallback
from transformers import Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from base import FineTuningMethod, TrainingConfig, CheckpointConfig
from checkpoint import CheckpointManager


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
    
    def prepare_dataset(
        self, 
        dataset: Any,
        input_column: str = 'instruction',
        output_column: str = 'output',
        split: str = 'train'
    ) -> Dataset:
        """
        Prepare dataset for LoRA SFT training.
        
        Args:
            dataset: Either a dataset name (str) from HF Hub, dict, or Dataset object
            input_column: Name of the column containing input instructions (default: 'instruction')
            output_column: Name of the column containing desired outputs (default: 'output')
            split: Dataset split to load (default: 'train')
        
        Returns:
            Dataset object formatted for SFT training
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
        
        # Check if required columns exist in the dataset
        available_columns = set(loaded_dataset.column_names)
        required_columns = [input_column, output_column]
        
        missing_columns = []
        for col in required_columns:
            if col not in available_columns:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}. "
                           f"Available columns: {list(available_columns)}")
        
        # No need to rename columns - we'll handle the formatting in the collate function
        return loaded_dataset
    
    def setup_peft_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Setup LoRA PEFT model with given configuration.
        
        Args:
            model: Base model to wrap with LoRA
            
        Returns:
            Model wrapped with LoRA PEFT
        """
        peft_config = LoraConfig(
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            target_modules=self.lora_config.target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=self.lora_config.task_type,
            modules_to_save=self.lora_config.modules_to_save
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def create_data_collator(
        self, 
        tokenizer: PreTrainedTokenizer,
        input_column: str = 'instruction',
        output_column: str = 'output'
    ):
        """
        Create a data collator for SFT that formats instruction-output pairs.
        
        Args:
            tokenizer: Tokenizer to use
            input_column: Column name for instructions
            output_column: Column name for outputs
            
        Returns:
            Data collator function
        """
        
        def data_collator(examples):
            sources = []
            targets = []
            
            for example in examples:
                # Create a prompt that combines instruction and output
                instruction = example[input_column]
                output = example[output_column]
                
                # You can customize the prompt format here
                # Example formats: 
                # - "### Instruction: {instruction}\n### Response: {output}"
                # - "Human: {instruction}\nAssistant: {output}"
                
                full_prompt = f"### Instruction: {instruction}\n### Response: {output}"
                sources.append(full_prompt)
                
                # For causal LM, we'll mask the instruction part in labels
                instruction_tokens = tokenizer.encode(f"### Instruction: {instruction}\n### Response: ")
                targets.append(output)
            
            # Tokenize all sequences
            model_inputs = tokenizer(
                sources,
                max_length=self.lora_config.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create labels (copy of input_ids)
            labels = model_inputs["input_ids"].clone()
            
            # Mask the instruction part in labels for loss calculation
            for i, (source, target) in enumerate(zip(sources, targets)):
                # Find where the response starts
                response_prefix = "### Response: "
                response_start = source.find(response_prefix)
                
                if response_start != -1:
                    # Get the position of the response prefix in tokens
                    prefix_tokens = tokenizer.encode(source[:response_start + len(response_prefix)])
                    mask_length = len(prefix_tokens) - 1  # -1 because of BOS token
                    
                    # Mask the instruction part
                    labels[i, :mask_length] = -100
            
            model_inputs["labels"] = labels
            
            return model_inputs
        
        return data_collator
    
    def train(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        dataset: Any, 
        config: TrainingConfig,
        input_column: str = 'instruction',
        output_column: str = 'output',
        split: str = 'train'
    ) -> PreTrainedModel:
        """
        Train a model using LoRA for Supervised Fine-Tuning.

        Args:
            model (PreTrainedModel): pretrained model to be fine-tuned with LoRA
            tokenizer (PreTrainedTokenizer): tokenizer of pretrained model
            dataset (Any): Either a dataset name (str) from HF Hub, dict, or Dataset object
            config (TrainingConfig): Training configuration containing checkpoint settings
            input_column: Column name for instructions
            output_column: Column name for desired outputs

        Returns:
            PreTrainedModel: fine-tuned model with LoRA for SFT
        """
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(
            dataset,
            input_column=input_column,
            output_column=output_column,
            split=split
        )
        
        # Setup LoRA on the model
        peft_model = self.setup_peft_model(model)
        
        # Create data collator
        data_collator = self.create_data_collator(
            tokenizer, 
            input_column=input_column,
            output_column=output_column
        )
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            learning_rate=config.learning_rate,
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
            logging_dir=f"{config.checkpoint_config.checkpoint_dir}/logs",
            logging_steps=10,
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
            optim="adamw_torch"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the LoRA adapters
        peft_model.save_pretrained(config.checkpoint_config.checkpoint_dir)
        
        return peft_model