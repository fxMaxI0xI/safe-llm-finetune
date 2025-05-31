import logging
from pathlib import Path
import sys

from huggingface_hub import login

# Add project root to Python path to enable imports from the package
sys.path.append(str(Path.cwd().parent))
import os

from safe_llm_finetune.datasets.code_ultra_feedback import CodeUltraFeedback
from safe_llm_finetune.evaluation.airbench import AirBench
from safe_llm_finetune.evaluation.codalbench import CodalBench
from safe_llm_finetune.evaluation.eval_analysis import evaluate
from safe_llm_finetune.evaluation.multitaskbench import MultiTaskBench
from safe_llm_finetune.fine_tuning.base import CheckpointConfig, TrainingConfig
from safe_llm_finetune.fine_tuning.methods.qlora import QLoRAConfig, QLoRAFineTuning
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import Gemma_3_1B
from safe_llm_finetune.utils.helpers import get_base_path
from safe_llm_finetune.utils.logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Starting model training pipeline")

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# Training

qlora_config = QLoRAConfig()
gemma_adapter = Gemma_3_1B()
code_ultra_feedback = CodeUltraFeedback(sample_size=600)
qlora_fine_tuning = QLoRAFineTuning(model_adapter=gemma_adapter, qlora_config=qlora_config)
checkpoint_config = CheckpointConfig()
training_config = TrainingConfig(checkpoint_config=checkpoint_config)



base_path = get_base_path(gemma_adapter, code_ultra_feedback, qlora_fine_tuning)

trained_model = qlora_fine_tuning.train(dataset_processor=code_ultra_feedback, config=training_config, base_path=base_path)

logger.info("Finished Training. Moving on to evals...")
# Evaluation

results = evaluate([AirBench(), MultiTaskBench(), CodalBench()], qlora_fine_tuning, trained_model, base_path, gemma_adapter.get_name())
print(results)

logger.info("Experiment run finished successfully!")
