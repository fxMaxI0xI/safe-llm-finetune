from inspect_ai.log import EvalLog
from safe_llm_finetune.evaluation.base import Evaluator
import pandas as pd
from huggingface_hub import list_repo_files
import re
import time

T = time.time()

def evaluate(evals: list[Evaluator], models: list[str]) -> pd.DataFrame:
    """Function to evaluate a list of models on a list of inspect evaluations and get results as csv file saved.

    Args:
        evals (list[Evaluator]): List of the evaluator to be used for evaluations
        models (list[str]): List of the hf model identifiers that are to be evaluated

    Returns:
        pd.DataFrame: dataframe of the result. Is also saved as csv file.
    """
    
    columns = ["model", "checkpoint"]
    results = pd.DataFrame(columns=columns)
    
    for model in models:
        # 1) find repo in hf
        print(f"Fetching repository contents for {model}...")
        try:
            repo_files = list_repo_files(repo_id=model)
        except Exception as e:
            print(f"Error fetching repository files for {model}: {e}")
            exit()

        # 2) Identify checkpoint folders usually named "checkpoint-STEP_NUMBER"
        checkpoint_dirs = set()
        for file_path in repo_files:
            # Check if the path contains a checkpoint folder
            match = re.search(r'^(checkpoint-\d+)(?:/|$)', file_path)
            if match:
                checkpoint_dirs.add(match.group(1))

        # Convert to a sorted list based on the numeric part
        checkpoint_folders = sorted(
            list(checkpoint_dirs),
            key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
        )

        if not checkpoint_folders:
            print(f"No checkpoint folders (e.g., 'checkpoint-1000') found at the root of {model}.")
        else:
            print(f"Found checkpoint folders: {checkpoint_folders}")


        # 3) run evals
        for checkpoint in checkpoint_folders:
            model_name = model
            metrics = []
            for eval in evals:
                log : EvalLog = eval.runEval(model_name=model_name, checkpoint_dir=checkpoint)
                if log.status != "success":
                    print(f"Eval {eval.get_name()} could not be computed for {model_name}!")
                    exit()
                else:
                    metrics.append = [(f"{eval.get_name()}_{metric_key}_{metric_object.name}", metric_object.value) for metric_key, metric_object in log.results.metrics.items()]
            
            current_index = results.index
            
            results.at[current_index, "model"] = model
            results.at[current_index, "checkpoint"] = checkpoint.split("-")[1]
            
            for metric in metrics:
                name, value = metric
                results.at[current_index, name] = value

            
            results.to_csv(f"./results/{T}_eval_results_{(model for model in models)}.csv", mode="a")
    
    return results
            
                
                
                
        


        
        
        
    