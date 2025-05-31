from huggingface_hub import login
import logging
from pathlib import Path
import sys
# Add project root to Python path to enable imports from the package
sys.path.append(str(Path.cwd().parent))
from safe_llm_finetune.datasets.code_ultra_feedback import CodeUltraFeedback
from safe_llm_finetune.fine_tuning.methods.lora import LoRAConfig, LoRAFineTuning
from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import Gemma_3_1B
from safe_llm_finetune.fine_tuning.base import TrainingConfig, CheckpointConfig
from safe_llm_finetune.utils.helpers import get_base_path
from safe_llm_finetune.utils.logging import setup_logging
from safe_llm_finetune.evaluation.eval_analysis import evaluate
from safe_llm_finetune.evaluation.airbench import AirBench
from safe_llm_finetune.evaluation.codalbench import CodalBench
from safe_llm_finetune.evaluation.multitaskbench import MultiTaskBench

import os

setup_logging()
    
logger = logging.getLogger(__name__)
logger.info("Starting model training pipeline")

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# Training

lora_config = LoRAConfig()
gemma_adapter = Gemma_3_1B()
code_ultra_feedback = CodeUltraFeedback(sample_size=600)
lora_fine_tuning = LoRAFineTuning(model_adapter=gemma_adapter, lora_config=lora_config)
checkpoint_config = CheckpointConfig()
training_config = TrainingConfig(checkpoint_config=checkpoint_config)



base_path = get_base_path(gemma_adapter, code_ultra_feedback, lora_fine_tuning)

trained_model = lora_fine_tuning.train(dataset_processor=code_ultra_feedback, config=training_config, base_path=base_path)

logger.info("Finished Training. Moving on to evals...")
# Evaluation

results = evaluate([AirBench(), MultiTaskBench(), CodalBench()], lora_fine_tuning, trained_model, base_path, gemma_adapter.get_name())
print(results)

logger.info("Experiment run finished successfully!")
