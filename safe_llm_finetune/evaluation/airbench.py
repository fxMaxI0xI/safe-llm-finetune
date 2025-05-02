"""Evaluator running AIR Bench Safety eval on specified model"""

from inspect_evals import air_bench
import os
from safe_llm_finetune.evaluation.base import Evaluator

HF = os.getenv("HF")

class AirBench(Evaluator):
    
    def create_task(self):
        return air_bench.air_bench()
    
