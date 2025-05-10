import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from safe_llm_finetune.datasets.base import DatasetProcessor
from safe_llm_finetune.fine_tuning.base import CheckpointConfig, FineTuningMethod, TrainingConfig


class SupervisedFullFineTuning(FineTuningMethod):
    """Simplified Supervised Full Fine-Tuning using HuggingFace Trainer."""

    def __init__(self, model_adapter, config: TrainingConfig):
        super().__init__(model_adapter)
        self.config = config

    def train(self, dataset_processor: DatasetProcessor) -> None:
        """
        Execute supervised full fine-tuning.

        Args:
            dataset_processor: Processor providing get_sft_dataset() with prompt/completion pairs
        """        # 1) Load datasets
        train_ds = dataset_processor.get_sft_dataset()

        # 2) Load model & tokenizer
        model = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer()

        # Optional: enable fp16
        if self.config.fp16:
            model.half()
            torch.cuda.empty_cache()

        # 3) Setup TrainingArguments
        ckpt = self.config.checkpoint_config
        training_args = TrainingArguments(
            output_dir=str(ckpt.checkpoint_dir),
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            save_strategy=ckpt.save_strategy,
            save_steps=ckpt.save_steps,
            save_total_limit=ckpt.save_total_limit,
            logging_steps=10,
            report_to="none",
        )

        # 4) Data collator for causal LM
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # 5) Initialize Trainer and train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        # 6) Save final model
        model.save_pretrained(str(ckpt.checkpoint_dir))
        tokenizer.save_pretrained(str(ckpt.checkpoint_dir))
        return model
