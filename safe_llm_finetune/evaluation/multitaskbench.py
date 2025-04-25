"""Evaluator running MultiTaskBench Safety eval on specified model"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from inspect_ai import eval as inspect_eval
from inspect_ai import Task, task, model
from inspect_ai.log import EvalLog
from inspect_ai.dataset import csv_dataset, FieldSpec
from inspect_ai.solver import generate, chain_of_thought
from inspect_ai.scorer import model_graded_qa
from transformers import PreTrainedModel, PreTrainedTokenizer
import os
import openai
import requests
from safe_llm_finetune.evaluation.base import Evaluator
from templates import REFUSAL_MODEL_GRADED_QA_TEMPLATE, REFUSAL_INSTRUCTION_TEMPLATE

HF = os.getenv("HF")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


class MultiTaskBench(Evaluator):
    def __init__(self, model_name, file_path):
        super().__init__(model_name, file_path)
        
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
                target="Response"
            )
            
        )
           
        
       
    @task        
    def create_task(self):
        # create model graded response outputting a binary yes = refusal / no = not refused
        detect_refusal = model_graded_qa(
            template= REFUSAL_MODEL_GRADED_QA_TEMPLATE,
            instructions= REFUSAL_INSTRUCTION_TEMPLATE,
            grade_pattern=r"GRADE: ([YN])",
            model="openai/gpt-4o-mini",
            )
        
        return Task(
            dataset= self.dataset,
            solver=[chain_of_thought(),generate()],
            scorer=detect_refusal
        )      
        