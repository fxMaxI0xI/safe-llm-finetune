from huggingface_hub import login
import argparse
import logging
from pathlib import Path
import sys
import wandb
# Add project root to Python path to enable imports from the package
sys.path.append(str(Path.cwd().parent))
from safe_llm_finetune.datasets.code_ultra_feedback import CodeUltraFeedback
from safe_llm_finetune.fine_tuning.methods.supervised_full_fine_tuning import FullFineTuning
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import Gemma_3_1B
from safe_llm_finetune.fine_tuning.base import TrainingConfig, CheckpointConfig
from safe_llm_finetune.utils.helpers import get_base_path
from safe_llm_finetune.utils.logging import setup_logging
from safe_llm_finetune.evaluation.eval_analysis import evaluate
from safe_llm_finetune.evaluation.airbench import AirBench
from safe_llm_finetune.evaluation.codalbench import CodalBench
from safe_llm_finetune.evaluation.multitaskbench import MultiTaskBench

import os

def parse_sample_size(value: str):
    if value.lower() == "none":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_size",
        type=parse_sample_size,
        default="0.1",
        help="Subset of the dataset (float 0-1 percentage, int number or 'None')",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run evaluations in debug mode (uses only a few samples)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples per evaluation",
    )
    args = parser.parse_args()

    setup_logging()
        
    logger = logging.getLogger(__name__)
    logger.info("Starting model training pipeline")

    WANDB = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB)

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)

    # Training

    gemma_adapter = Gemma_3_1B()
    code_ultra_feedback = CodeUltraFeedback(sample_size=args.sample_size)
    full_fine_tuning = FullFineTuning(model_adapter=gemma_adapter)
    checkpoint_config = CheckpointConfig()
    training_config = TrainingConfig(checkpoint_config=checkpoint_config)



    base_path = get_base_path(gemma_adapter, code_ultra_feedback, full_fine_tuning)

    trained_model = full_fine_tuning.train(dataset_processor=code_ultra_feedback, config=training_config, base_path=base_path)

    logger.info("Finished Training. Moving on to evals...")
    # Evaluation

    results = evaluate(
        [AirBench(debug=args.debug), MultiTaskBench(debug=args.debug), CodalBench(debug=args.debug)],
        full_fine_tuning,
        trained_model,
        base_path,
        gemma_adapter.get_name(),
        limit=args.limit,
    )
    print(results)

    logger.info("Experiment run finished successfully!")

if __name__ == "__main__":
    main()
