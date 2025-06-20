"""Evaluator running AIR Bench Safety eval on specified model"""

import os

from inspect_evals import air_bench

from safe_llm_finetune.evaluation.base import Evaluator


class AirBench(Evaluator):
    
    def create_task(self):
        return air_bench.air_bench()
    
    def get_name(self):
        return "AirBench"
