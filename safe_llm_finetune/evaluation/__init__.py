"""Evaluation module for safe_llm_finetune."""

# Import classes from base.py
from .base import Evaluator

# Import specific evaluator implementations
from .multitaskbench import MultiTaskBench
from .airbench import AirBench




# Define what gets exported when someone does "from safe_llm_finetune.evaluation import *"
__all__ = [
    # Base classes
    "Evaluator", 
    
    
    # Evaluator implementations
    "MultiTaskBench",
    "AirBench"
 
]