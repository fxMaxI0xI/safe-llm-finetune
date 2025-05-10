from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class DatasetProcessor(ABC):
    """
    Abstract base class for dataset preprocessing in SFT and DPO pipelines.
    Implementations should inherit from this class and implement the required methods.
    """
    
    def __init__(self, dataset_path: str, sample_size: Optional[Union[float, int]] = None):
        """
        Initialize the dataset processor.
        
        Args:
            dataset_path: Path to the dataset or identifier
            tokenizer: Tokenizer to be used for preprocessing
            max_length: Maximum sequence length for tokenization
            sample_size: Optional sampling size, can be:
                        - float between 0 and 1: represents percentage of dataset to use
                        - int > 1: represents absolute number of examples to use
                        - None: use all examples
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.loaded_data = None
        
        # Validate sample_size
        if sample_size is not None:
            if isinstance(sample_size, float):
                if not (0 < sample_size <= 1):
                    raise ValueError("If sample_size is a float, it must be between 0 and 1")
                else: 
                    self.percentage = True
            elif isinstance(sample_size, int):
                if sample_size <= 0:
                    raise ValueError("If sample_size is an int, it must be greater than 0")
                else: 
                    self.percentage = False
        
        
    @abstractmethod
    def load_data(self) -> None:
        """
        Load the raw dataset from the source.
        Stores dataset in self.loaded_data
        """
        pass
    
    @abstractmethod
    def get_sft_dataset(self) -> Dataset:
        """
        Get a dataset ready for Supervised Fine-Tuning (SFT).
        
            
        Returns:
            A hf Dataset ready for SFT training in format:
            {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
        """
        pass
    
    @abstractmethod
    def get_dpo_dataset(self, num_samples: Optional[int] = None) -> Dataset:
        """
        Get a dataset ready for Direct Preference Optimization (DPO).
            
        Returns:
            A hf Dataset ready for DPO training with prompt, chosen and rejected responses
        """
        pass
    
    @abstractmethod
    def get_name(self):
        """
        returns name of dataset
        """
        pass