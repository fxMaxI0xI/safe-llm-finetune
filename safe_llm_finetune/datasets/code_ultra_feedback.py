# safe_llm_finetune/datasets/code_ultra_feedback.py
from typing import Optional, Union
import logging
from datasets import Dataset, load_dataset
from safe_llm_finetune.datasets.base import DatasetProcessor


class CodeUltraFeedback(DatasetProcessor):
    """
    Hugging-Face-Datensatz **coseal/CodeUltraFeedback_binarized**.

    Args:
        sample_size:
            - float 0-1 → Prozentanteil (0.1 == 10 %)
            - int   >1  → absolute Zahl Beispiele
            - None      → kompletter Split
    """
    def __init__(self, sample_size: Optional[Union[float, int]] = None):
        super().__init__("coseal/CodeUltraFeedback_binarized", sample_size)
        self.logger = logging.getLogger(__name__)  # Creates 'transformer_model' logger
        self.percentage = isinstance(sample_size, float)  # für load_data()
        self.logger.info("Initialized CodeUltraFeedback DataProcessor")

    # ---------- Laden ----------
    def load_data(self) -> None:
        self.logger.info("Loading Training Data Set")
        if self.percentage:
            self.loaded_data = load_dataset(
                self.dataset_path, split=f"train[:{self.sample_size*100}%]"
            )
            self.logger.info(f"Loaded Training Data Set with first {self.sample_size*100}% examples")
            
        elif self.sample_size is not None:
            self.loaded_data = load_dataset(
                self.dataset_path, split=f"train[:{self.sample_size}]"
            )
            self.logger.info(f"Loaded Training Data Set with first {self.sample_size} examples")

        else:
            self.loaded_data = load_dataset(self.dataset_path, split="train")
            self.logger.info("Loaded full Training Data Set")


    # ---------- SFT ----------
    def get_sft_dataset(self) -> Dataset:
        if self.loaded_data is None:
            self.load_data()

        def format(example):
            return {
                "text": f"User: {example['instruction']}\nAssistant: {example['chosen']}"
            }

        self.logger.info("Applying SFT format to Dataset")

        return self.loaded_data.map(
            format, remove_columns=self.loaded_data.column_names
        )

    # ---------- DPO ----------
    def get_dpo_dataset(self, num_samples: Optional[int] = None) -> Dataset:
        if self.loaded_data is None:
            self.load_data()

        dpo = self.loaded_data.remove_columns(
            [c for c in self.loaded_data.column_names
             if c not in ["instruction", "chosen", "rejected"]]
        ).rename_column("instruction", "prompt")
        
        self.logger.info("Applying DPO format to Dataset")

        return dpo

    def get_name(self):
        return "CodeUltraFeedback"
