from trl import SFTConfig, SFTTrainer

from safe_llm_finetune.fine_tuning.base import (
    FineTuningMethod,
    ModelAdapter,
    TrainingConfig,
)


class FullFineTuning(FineTuningMethod):
    """Trainiert alle Gewichte (kein PEFT)."""
    def train(self, dataset_processor, config: TrainingConfig):
        # 1) Datensatz holen
        ds = dataset_processor.get_sft_dataset()

        # 2) Modell & Tokenizer laden
        model = self.model_adapter.load_model()
        tokenizer = self.model_adapter.load_tokenizer()
        model.gradient_checkpointing_enable()

        # 3) Alle Trainings-Parameter extrahieren
        sft_kwargs = config.as_dict().copy()
        # report_to entfernen, wir fügen es gleich separat wieder hinzu
        report_to = sft_kwargs.pop("report_to", None)

        # 4) SFTConfig initialisieren
        args = SFTConfig(
            output_dir=str(config.checkpoint_config.checkpoint_dir),
            run_name=config.run_name,
            report_to=report_to,
            bf16=True,
            remove_unused_columns=False,
            save_strategy="epoch",
            save_total_limit=1,
            **sft_kwargs,
        )

        # 5) Trainer bauen und trainieren
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=ds,
        )
        trainer.train()

        # 6) Modell & Tokenizer speichern
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        return model
