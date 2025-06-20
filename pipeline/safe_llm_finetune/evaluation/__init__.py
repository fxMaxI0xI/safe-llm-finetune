"""Evaluation module for safe_llm_finetune."""

# Import classes from base.py
from .airbench import AirBench
from .base import Evaluator
from .codalbench import CodalBench

# Import specific evaluator implementations
from .multitaskbench import MultiTaskBench

# Define what gets exported when someone does "from safe_llm_finetune.evaluation import *"
__all__ = [
    # Base classes
    "Evaluator", 
    
    
    # Evaluator implementations
    "MultiTaskBench",
    "AirBench",
    "CodalBench"
 
]
