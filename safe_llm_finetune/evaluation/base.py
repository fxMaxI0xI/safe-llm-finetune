"""
Abstract base classes for evaluation functionality.
"""
from abc import ABC, abstractmethod
from typing import List
from inspect_ai import eval as inspect_eval
from inspect_ai import Task
from inspect_ai.log import EvalLog
import os

HF = os.getenv("HF")

class Evaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(self, model_name: str, file_path: str, debug = False):
        """initialize evaluator instance

        Args:
            model_name (str): name of model to evaluate, will be inserted as hf/(username)/model_name
            file_path (str): Where to store eval logs
            debug (bool): enter debugging mode (10 examples, gpt 4o mini as model to evaluate)
        """
        self.model_name = model_name
        self.file_path = file_path
        self.dataset = None
        self.debug = debug

    @abstractmethod
    def create_task(self) -> Task | List[Task]:
        """creates a inspect_ai task object from the given dataset with suitable solver and scorer

        Returns:
            Task: inspect task object
        """
        pass

    def runEval(self) -> EvalLog:
        """runs inpects inate eval() function

        Returns:
            EvalLog: returns log of evaluation from eval() call
        """
        task = self.create_task()
        model = "hf/"+ HF + "/" + self.model_name
        if self.debug:
            print(type(task))
            results = inspect_eval(tasks=task, model= "openai/gpt-4o-mini", log_dir= self.file_path + "/eval_log", limit=10)
        else:
            results = inspect_eval(tasks=task, model= model, log_dir= self.file_path + "/eval_log", model_args=dict(device="cuda:0"))
        return results