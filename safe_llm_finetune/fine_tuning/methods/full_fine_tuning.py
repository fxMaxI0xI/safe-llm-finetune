import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset as HFDataset
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import GemmaAdapter
from safe_llm_finetune.fine_tuning.base import FineTuningMethod, TrainingConfig

class FullFineTuning(FineTuningMethod):
    def prepare_dataset(self, dataset):
        """
        Expects a pandas DataFrame with columns: 'prompt', 'completion'
        Returns a HuggingFace Dataset formatted for Causal LM
        """
        df = dataset if isinstance(dataset, pd.DataFrame) else dataset._data
        df = df.rename(columns={"prompt": "text"})
        hf_dataset = HFDataset.from_pandas(df)
        tokenizer = self.model_adapter.load_tokenizer("google/gemma-1.3b-it")

        def tokenize(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

        return hf_dataset.map(tokenize, batched=True)

    def train(self, model, tokenizer, dataset, config: TrainingConfig):
        training_args = TrainingArguments(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            evaluation_strategy="no",
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            num_train_epochs=config.num_train_epochs,
            weight_decay=config.weight_decay,
            save_strategy=config.checkpoint_config.save_strategy,
            save_steps=config.checkpoint_config.save_steps,
            save_total_limit=config.checkpoint_config.save_total_limit,
            fp16=config.fp16,
            seed=config.seed,
            report_to="none"
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        return model

    def evaluate(self, model, tokenizer, dataset):
        return {"status": "evaluation not implemented yet"}  # optional
