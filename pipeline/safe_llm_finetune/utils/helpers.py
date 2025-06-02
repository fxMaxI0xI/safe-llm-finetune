from pathlib import Path

from safe_llm_finetune.datasets.base import DatasetProcessor
from safe_llm_finetune.fine_tuning.base import FineTuningMethod, ModelAdapter


def get_base_path(model_adapter: ModelAdapter, data_processor: DatasetProcessor, fine_tuner: FineTuningMethod) -> str:
    return f"./models/{model_adapter.get_name()}-{data_processor.get_name()}-{fine_tuner.get_name()}"