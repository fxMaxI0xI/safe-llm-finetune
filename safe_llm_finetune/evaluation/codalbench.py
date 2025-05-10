"""Evaluator running CodalBench Performance eval on specified model"""
import os
from typing import Optional

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset
from inspect_ai.scorer import mean, model_graded_qa, stderr
from inspect_ai.solver import chain_of_thought, generate
import openai

from safe_llm_finetune.evaluation.base import Evaluator
from safe_llm_finetune.evaluation.prompt_templates import (
    CODAL_INSTRUCTION_TEMPLATE,
    CODAL_PROMPT_TEMPLATES,
)

HF = os.getenv("HF")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


class CodalBench(Evaluator):
    def __init__(self, model_name, file_path, debug= False, preference: Optional[str] = None, judge_model = "openai/gpt-4o-mini"):
        super().__init__(model_name, file_path, debug)
        
        self.preference = preference
        self.judge_model = judge_model
        # Validate preference if provided
        if preference is not None and preference not in CODAL_PROMPT_TEMPLATES.keys():
            raise ValueError(f"Invalid preference '{preference}'. Must be one of: {list(CODAL_PROMPT_TEMPLATES.keys())}")

    
        # create inspect ai dataset instance
            
        self.dataset = hf_dataset(
        path="coseal/codal-bench",
        split="test",
        sample_fields=FieldSpec(
        input="instruction",  
        target="claude-3-sonnet-20240229_response",  # take always claude reference
        metadata=["preference"]  
    )
    )
            
        
    def available_preferences(self):
        print("Available preferences:", list(CODAL_PROMPT_TEMPLATES.keys()))
        
       
           
    def create_task(self):
        """Creates the set of tasks with specific prompt for each preference category.
        
        Returns:
            Task or List[Task]: Single task if preference is specified, otherwise all tasks
        """
        
        # If specific preference is provided, create only that task
        if self.preference is not None:
            @task(name=f"codal_{self.preference}")
            def single_preference_task():
                # Filter the dataset for this preference
                filtered_dataset = self.dataset.filter(
                    lambda sample: sample.metadata.get("preference") == self.preference
                )
                
                # Create scorer for this preference
                scorer = model_graded_qa(
                    template=CODAL_PROMPT_TEMPLATES[self.preference],
                    instructions=CODAL_INSTRUCTION_TEMPLATE,
                    grade_pattern=r"GRADE:\s*(\d+)/10",
                    model=self.judge_model
                )
                
                return Task(
                    dataset=filtered_dataset,
                    scorer=scorer,
                    metrics=[
                        mean(),
                        stderr()
                    ]
                )
            
            return single_preference_task()
        
        # otherwise, create all tasks
        
        tasks = []
        
        for preference in CODAL_PROMPT_TEMPLATES.keys():
            @task(name=f"codal_{preference}")
            def preference_task(preference=preference):
                
                # Filter the dataset for this preference
                filtered_dataset = self.dataset.filter(
                    lambda sample: sample.metadata.get("preference") == preference
                )
                
                # Create scorer for this preference
                scorer = model_graded_qa(
                    template=CODAL_PROMPT_TEMPLATES[preference],
                    instructions=CODAL_INSTRUCTION_TEMPLATE,
                    grade_pattern=r"GRADE:\s*(\d+)/10",
                    model=self.judge_model
                )
                
                return Task(
                    dataset=filtered_dataset,
                    solver=[chain_of_thought(),generate()],
                    scorer=scorer,
                    metrics=[
                        mean(),
                        stderr()
                    ]
                )
            
            tasks.append(preference_task())
        
        return tasks
        
        