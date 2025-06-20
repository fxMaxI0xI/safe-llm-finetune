import logging
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, PreTrainedModel
from trl import SFTConfig, SFTTrainer

from safe_llm_finetune.fine_tuning.base import (
    FineTuningMethod,
    TrainingConfig,
)


class FullFineTuning(FineTuningMethod):
    """Trainiert alle Gewichte (kein PEFT)."""

    def __init__(self, model_adapter):
        super().__init__(model_adapter, "full")
        self.logger = logging.getLogger(__name__)

    def get_name(self):
        return self.training_method

    def train(self, dataset_processor, config: TrainingConfig, base_path: Path):
        """Train the entire model without PEFT.

        Args:
            dataset_processor: DatasetProcessor instance providing the training data
            config (TrainingConfig): Training configuration containing checkpoint settings
            base_path (Path): Base path used to format the checkpoint directory
        """

        self.logger.info("Starting full fine-tuning preparations...")

        # 1) Datensatz holen
        ds = dataset_processor.get_sft_dataset()

        # 2) Modell & Tokenizer laden
        model = self.model_adapter.load_model()
        _ = self.model_adapter.load_tokenizer()
        # model.gradient_checkpointing_enable() #disabled for more GPU Ram usage

        # 3) Alle Trainings-Parameter extrahieren
        sft_kwargs = config.as_dict().copy()
        # report_to entfernen, wir fügen es gleich separat wieder hinzu
        report_to = sft_kwargs.pop("report_to", None)

        checkpoint_dir = config.checkpoint_config.checkpoint_dir.format(base=base_path)

        # 4) SFTConfig initialisieren
        args = SFTConfig(
            output_dir=checkpoint_dir,
            run_name=config.run_name,
            report_to=report_to,
            bf16=True,
            remove_unused_columns=False,
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=5, # für wandb graphen 
            **sft_kwargs,
        )

        # 5) Trainer bauen und trainieren
        self.logger.info("Initializing SFTTrainer for full fine-tuning.")
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=ds,
        )
        self.logger.info("Starting full fine-tuning training...")
        trainer.train()
        self.logger.info("Finished full fine-tuning training. Saving model now...")

        # 6) Save with config metadata for traceability
        save_dir = checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)

        # Save model
        trainer.save_model(save_dir)

        # 7) push final model to hub
        if config.checkpoint_config.push_to_hub:
            trainer.push_to_hub()

        # 8) save locally meta data
        self.save_training_metadata(checkpoint_dir, self.model_adapter.get_name())
        self.logger.info("Model and metadata saved. Successfully trained model.")

        return model

    def load_model_from_checkpoint(self, checkpoint_path: str) -> PreTrainedModel:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        self.logger.info("Lade Modell aus Checkpoint-Verzeichnis %s", checkpoint_path)
        # wenn du PEFT benutzt:
        try:
            from peft import PeftConfig, PeftModel

            peft_cfg = PeftConfig.from_pretrained(checkpoint_path)
            base = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path)
            peft = PeftModel.from_pretrained(base, checkpoint_path)
            return peft.merge_and_unload()
        except ImportError:
            # reines HF-Model
            return AutoModelForCausalLM.from_pretrained(checkpoint_path)
