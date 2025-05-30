"""
Abstract base classes for evaluation functionality.
"""
from abc import ABC, abstractmethod
import os
from typing import List

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import get_model
from transformers import PreTrainedModel

HF = os.getenv("HF")
class Evaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(self, debug = False):
        """initialize evaluator instance

        Args:
            debug (bool): enter debugging mode (10 examples, gpt 4o mini as model to evaluate)
        """
        self.dataset = None
        self.debug = debug
        
    @abstractmethod
    def get_name(self) -> str:
        """Returns name of eval
        """

    @abstractmethod
    def create_task(self) -> Task | List[Task]:
        """creates a inspect_ai task object from the given dataset with suitable solver and scorer

        Returns:
            Task: inspect task object
        """
        pass

    def run_eval(self, model_path: str, tokenizer_path: str, base_path: str) -> EvalLog:
        """runs inpects inate eval() function
        
        Args:
            model (PreTrainedModel): loaded local model
            checkpoint_dir (str): Path to the checkpoint directory
            file_path (str): Where to store eval logs

        Returns:
            EvalLog: returns log of evaluation from eval() call
        """
        task = self.create_task()
        
        log_file_path = f"{base_path}/{self.get_name()}"
        if self.debug:
            results = inspect_eval(tasks=task, model= "openai/gpt-4o-mini", log_dir= log_file_path, limit=10)
        else:
            
            results = inspect_eval(tasks=task, model="hf/local", model_args=dict(model_path=model_path, tokenizer_path=tokenizer_path), log_dir=log_file_path, fail_on_error=False, limit=500, retry_on_error = 10)
        
        return results