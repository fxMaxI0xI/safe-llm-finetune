import argparse
from safe_llm_finetune.datasets.code_ultra_feedback import CodeUltraFeedback
from safe_llm_finetune.fine_tuning.base import CheckpointConfig, TrainingConfig
from safe_llm_finetune.fine_tuning.methods.supervised_full_fine_tuning import (
    SupervisedFullFineTuning,
)
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import Gemma_3_1B_it

# Dummy CLI for testing Supervised Full FT
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # 1) Config
    ckpt_cfg = CheckpointConfig(checkpoint_dir="./test_supervised_output")
    training_cfg = TrainingConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=args.fp16,
        checkpoint_config=ckpt_cfg
    )

    # 2) Processor
    tokenizer = Gemma_3_1B_it().load_tokenizer("gpt2")  # kleineres Modell zum Testen
    processor = CodeUltraFeedback(sample_size=args.sample_size)

    # 3) Adapter
    adapter = Gemma_3_1B_it()
    adapter.model_id = "gpt2"  # auf GPT-2 umstellen

    # 4) Method
    finetuner = SupervisedFullFineTuning(adapter, training_cfg)
    finetuner.train(processor)

    print("Test run completed.")
