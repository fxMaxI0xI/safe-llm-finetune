"""
Abstract base classes for evaluation functionality.
"""
from abc import ABC, abstractmethod
import os
from typing import List
from inspect_ai.model import get_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

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

    def runEval(self, model_name: str, checkpoint_dir: str ,file_path: str = "./evallogs") -> EvalLog:
        """runs inpects inate eval() function
        
        Args:
            model_name (str): name of model to evaluate
            checkpoint_dir (str): Path to the checkpoint directory
            file_path (str): Where to store eval logs

        Returns:
            EvalLog: returns log of evaluation from eval() call
        """
        task = self.create_task()
        if self.debug:
            results = inspect_eval(tasks=task, model= "openai/gpt-4o-mini", log_dir= file_path, limit=10)
        else:

            print(model_name, checkpoint_dir)
            checkpoint_path = os.path.join(model_name, checkpoint_dir)
            print(checkpoint_path)
        
            model = get_model(model="hf/"+model_name, device= "cuda:0", from_tf=True, subfolder= checkpoint_dir)
            
            results = inspect_eval(tasks=task, model=model, log_dir=file_path + "/eval_log")
        
        return results