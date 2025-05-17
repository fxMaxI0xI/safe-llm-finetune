#!/usr/bin/env python
import argparse
import os
import pathlib

# Um die Tokenizer-Warnung loszuwerden
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Project imports ---------------------------------------------------------
from safe_llm_finetune.datasets.code_ultra_feedback import CodeUltraFeedback
from safe_llm_finetune.fine_tuning.base import (
    CheckpointConfig,
    ModelAdapter,
    TrainingConfig,
)
from safe_llm_finetune.fine_tuning.methods.supervised_full_fine_tuning import FullFineTuning
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import GemmaAdapter


def parse_args():
    p = argparse.ArgumentParser("Full SFT launcher")
    p.add_argument("--run_name",       type=str, default=None,
                   help="Name des Runs in W&B")
    p.add_argument("--max_length",     type=int, default=1024,
                   help="Maximale Token-Länge pro Sequenz")
    p.add_argument("--batch_size",     type=int, default=2,
                   help="Batch-Größe pro GPU")
    p.add_argument("--learning_rate", "-lr",
                   type=float, default=5e-5,
                   help="Start-Learning-Rate")
    p.add_argument("--warmup_steps",
                   type=int, default=0,
                   help="Anzahl Warmup-Schritte")
    p.add_argument("--model_name",     type=str, required=True,
                   help="z. B. google/gemma-3-1B-it")
    p.add_argument("--out",            type=str, default="runs/tmp",
                   help="Ausgabeverzeichnis für Checkpoints/Final-Model")
    p.add_argument("--epochs",         type=int, default=1,
                   help="Anzahl Trainingsepochen")
    p.add_argument("--sample_size",
                   type=lambda x: None if x.lower()=="none" else int(x),
                   default=None,
                   help="Number of samples (int) oder None für alle")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Datensatz vorbereiten
    ds = CodeUltraFeedback(sample_size=args.sample_size)

    # 2) Modell-Adapter wählen
    if args.model_name.startswith("google/gemma"):
        ma = GemmaAdapter(args.model_name)
    else:
        ma = ModelAdapter(args.model_name)

    # 3) Trainings-Konfiguration
    cfg = TrainingConfig(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_length,
        report_to="wandb",                     # aktiviert W&B-Logging
        run_name=args.run_name,                # Lauf-Name in W&B
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=pathlib.Path(args.out),
            save_strategy="epoch",
            save_total_limit=1,
        ),
    )

    # 4) Voll-Fine-Tuning starten
    FullFineTuning(ma).train(ds, cfg)


if __name__ == "__main__":
    main()
