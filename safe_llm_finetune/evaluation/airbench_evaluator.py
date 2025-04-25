"""Evaluator running AIR Bench Safety eval on specified model"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from inspect_ai import eval as inspect_eval
from inspect_ai import Task, task, model
from inspect_ai.log import EvalLog
from inspect_evals import air_bench
from transformers import PreTrainedModel, PreTrainedTokenizer

from base import Evaluator

class AirBenchEvaluator(Evaluator):
    
    def run(self) -> EvalLog:
        results = inspect_eval(tasks=task, model=self.model, log_dir= self.file_path + "/eval_log")
        return results
        