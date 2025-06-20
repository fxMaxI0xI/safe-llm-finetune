"""Evaluator running MultiTaskBench Safety eval on specified model"""
import os

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, csv_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import chain_of_thought, generate
import openai
import requests

from safe_llm_finetune.evaluation.base import Evaluator
from safe_llm_finetune.evaluation.metrics import attack_success_rate
from safe_llm_finetune.evaluation.prompt_templates import (
    REFUSAL_INSTRUCTION_TEMPLATE,
    REFUSAL_MODEL_GRADED_QA_TEMPLATE,
)

HF = os.getenv("HF")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


class MultiTaskBench(Evaluator):
    def __init__(self, debug= False, judge_model ="openai/gpt-4o-mini"):
        super().__init__(debug)
        
        # downloading multitaskbench dataset, stored in data folder locally
        
        url = "https://raw.githubusercontent.com/comnetsAD/LLMSafetyGuardrails/refs/heads/main/datasets/multitaskbench/multitaskbench.csv"
        output_path = "data/multitaskbench.csv"

        os.makedirs("data", exist_ok=True)

        if not os.path.exists(output_path):
            print("Downloading MultiTaskBench dataset")
            response = requests.get(url)
            with open(output_path, "wb") as f:
                f.write(response.content)
            print("Done.")
            
        # create inspect ai dataset instance
            
        self.dataset = csv_dataset(
            "data/multitaskbench.csv",
            FieldSpec(
                input="Query",
                target="Response",
            )
            
        )
        self.judge_model = judge_model
           
    def get_name(self):
        return "MultiTaskBench"
       
    @task        
    def create_task(self):
        # create model graded response outputting a binary yes = refusal / no = not refused
        detect_refusal = model_graded_qa(
            template= REFUSAL_MODEL_GRADED_QA_TEMPLATE,
            instructions= REFUSAL_INSTRUCTION_TEMPLATE,
            grade_pattern=r"GRADE: ([YNyn])",
            model=self.judge_model
            )
        
        return Task(
            dataset= self.dataset,
            solver=[chain_of_thought(),generate()],
            scorer=detect_refusal,
            metrics=[
                attack_success_rate(),
            ]
        )      
        
