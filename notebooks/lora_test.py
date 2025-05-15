from huggingface_hub import login
from pathlib import Path
import sys
# Add project root to Python path to enable imports from the package
sys.path.append(str(Path.cwd().parent))
from safe_llm_finetune.datasets.code_ultra_feedback import CodeUltraFeedback

from safe_llm_finetune.fine_tuning.methods.lora import LoRAConfig, LoRAFineTuning

from safe_llm_finetune.fine_tuning.models.gemma_3_1B_it_adapter import GemmaAdapter
from safe_llm_finetune.fine_tuning.base import TrainingConfig, CheckpointConfig

import os



HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)


lora_config = LoRAConfig()
gemma_adapter = GemmaAdapter()
code_ultra_feedback = CodeUltraFeedback(sample_size=600)


lora_fine_tuning = LoRAFineTuning(model_adapter=gemma_adapter, lora_config=lora_config)
checkpoint_config = CheckpointConfig(checkpoint_dir="./checkpoints")
training_config = TrainingConfig(checkpoint_config=checkpoint_config)
trained_model = lora_fine_tuning.train(dataset_processor=code_ultra_feedback, config=training_config)