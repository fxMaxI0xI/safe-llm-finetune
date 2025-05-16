from trl import SFTConfig, SFTTrainer
from safe_llm_finetune.fine_tuning.base import (
    FineTuningMethod, ModelAdapter, TrainingConfig
)

class FullFineTuning(FineTuningMethod):
    """Trainiert alle Gewichte (kein PEFT)."""
    def train(self, dataset_processor, config: TrainingConfig):
        ds  = dataset_processor.get_sft_dataset()
        model     = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer()
        model.gradient_checkpointing_enable()

        args = SFTConfig(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            run_name=config.run_name or None,
            **config.as_dict(),
            bf16=True,
            remove_unused_columns=False,
        )
        trainer = SFTTrainer(model=model,
                             args=args,
                             train_dataset=ds,)
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        return model
