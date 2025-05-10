from typing import Any, Optional

from datasets import Dataset, load_dataset
from safe_llm_finetune.datasets.base import DatasetProcessor


class CodeUltraFeedback(DatasetProcessor):
    def __init__(self, sample_size = None):
        super().__init__("coseal/CodeUltraFeedback_binarized", sample_size)
    
    def load_data(self) -> None:
        """
        Load the codeUltraFeedback dataset from the hf/coseal/CodeUltraFeedback_binarized.
        
        Returns:
            The raw dataset object
        """
        if self.percentage:
            self.loaded_data = load_dataset(self.dataset_path, split=f"train[:{self.sample_size*100}%]")
        elif self.sample_size != None:
            self.loaded_data = load_dataset(self.dataset_path, split=f"train[:{self.sample_size}]")
        else:
            self.loaded_data = load_dataset(self.dataset_path, split=f"train")
    
    def get_sft_dataset(self) -> Dataset:
        """
        Get a dataset ready for Supervised Fine-Tuning (SFT).
            
        Returns:
            A hf Dataset ready for SFT training
        """
        if self.loaded_data == None: 
            self.load_data()
        
        def apply_format(example):
            instruction = example["instruction"]
            response = example["chosen"]
            return {"prompt": instruction, "completion": response}
        
        return self.loaded_data.map(apply_format)

    def get_dpo_dataset(self, num_samples: Optional[int] = None) -> Dataset:
        """
        Get a dataset ready for Direct Preference Optimization (DPO).
        
        Args:
            num_samples: Optional number of samples to include, if None use all
            
        Returns:
            A hf Dataset ready for DPO training with prompt, chosen and rejected responses
        """
        if self.loaded_data == None: 
            self.load_data()
        
        self.loaded_data = self.loaded_data.remove_columns([col for col in self.loaded_data.column_names if col not in ["instruction", "chosen", "rejected"]])
        return self.loaded_data.rename_column("instruction", "prompt")
    
        
    
    