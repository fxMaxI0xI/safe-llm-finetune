"""Evaluator running AIR Bench Safety eval on specified model"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from inspect_ai import eval as inspect_eval
from inspect_ai import Task, task, model
from inspect_ai.log import EvalLog
from inspect_evals import air_bench
from transformers import PreTrainedModel, PreTrainedTokenizer
import os
from safe_llm_finetune.evaluation.base import Evaluator

HF = os.getenv("HF")

class AirBench(Evaluator):
    
    def run(self) -> EvalLog:
        model = "hf/"+ HF + "/" + self.model_name
        results = inspect_eval(tasks=air_bench, model= model, log_dir= self.file_path + "/eval_log", model_args=dict(device="cuda:0"))
        return results
        