"""
Abstract base classes for dataset handling.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import pandas as pd


class DatasetFilter(ABC):
    """Abstract base class for dataset filters."""
    
    @abstractmethod
    def filter(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the dataset based on specific criteria.
        
        Args:
            dataset: Input dataset to filter
            
        Returns:
            Filtered dataset
        """
        pass


class DatasetPreprocessor(ABC):
    """Abstract base class for dataset preprocessors."""
    
    @abstractmethod
    def preprocess(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset.
        
        Args:
            dataset: Input dataset to preprocess
            
        Returns:
            Preprocessed dataset
        """
        pass


class Dataset(ABC):
    """Abstract base class for datasets."""
    
    def __init__(
        self,
        filters: Optional[List[DatasetFilter]] = None,
        preprocessors: Optional[List[DatasetPreprocessor]] = None
    ):
        """
        Initialize dataset with optional filters and preprocessors.
        
        Args:
            filters: List of dataset filters to apply
            preprocessors: List of dataset preprocessors to apply
        """
        self.filters = filters or []
        self.preprocessors = preprocessors or []
        self._data = None
    
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """
        Load dataset from path.
        
        Args:
            path: Path to dataset
            
        Returns:
            Loaded dataset
        """
        pass
    
    def apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters to the dataset.
        
        Args:
            data: Input dataset
            
        Returns:
            Filtered dataset
        """
        for filter_obj in self.filters:
            data = filter_obj.filter(data)
        return data
    
    def apply_preprocessors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessors to the dataset.
        
        Args:
            data: Input dataset
            
        Returns:
            Preprocessed dataset
        """
        for preprocessor in self.preprocessors:
            data = preprocessor.preprocess(data)
        return data
    
    def process(self, path: str) -> pd.DataFrame:
        """
        Load, filter, and preprocess dataset.
        
        Args:
            path: Path to dataset
            
        Returns:
            Processed dataset
        """
        data = self.load(path)
        data = self.apply_filters(data)
        data = self.apply_preprocessors(data)
        self._data = data
        return data
    
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Return iterator over dataset items."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save processed dataset.
        
        Args:
            path: Path to save dataset
        """
        pass