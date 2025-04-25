"""
Abstract base classes for evaluation functionality.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from inspect_ai import eval as inspect_eval
from inspect_ai import Task, task
from inspect_ai.log import EvalLog
from transformers import PreTrainedModel, PreTrainedTokenizer
import os

HF = os.getenv("HF")

class Evaluator(ABC):
    """Abstract base class for evaluators."""

    def __init__(self, model_name: str, file_path: str):
        self.model_name = model_name
        self.file_path = file_path
        self.dataset = None

    @abstractmethod
    def create_task(self) -> Task:
        pass

    def run(self) -> EvalLog:
        task = self.create_task()
        model = "hf/"+ HF + "/" + self.model_name
        results = inspect_eval(tasks=task, model= self.model_name, log_dir= self.file_path + "/eval_log", model_args=dict(device="cuda:0"))
        return results