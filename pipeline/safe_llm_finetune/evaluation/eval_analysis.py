"""
Evaluation script for fine-tuned models with checkpoint support.
"""
from datetime import datetime
import logging
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import torch

import pandas as pd
from transformers import PreTrainedModel

from safe_llm_finetune.evaluation.base import Evaluator
from safe_llm_finetune.fine_tuning.base import FineTuningMethod

logger = logging.getLogger(__name__)


def discover_checkpoints(checkpoint_dir: str) -> List[Tuple[str, int]]:
    """
    Discover all checkpoint directories in the given path.
    
    Args:
        checkpoint_dir: Path to the directory containing checkpoints
        
    Returns:
        List of tuples (checkpoint_path, step_number) sorted by step number
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return []
    
    checkpoints = []
    
    # Look for directories matching the pattern checkpoint-XXXX
    for item in checkpoint_path.iterdir():
        if item.is_dir():
            match = re.match(r'checkpoint-(\d+)', item.name)
            if match:
                step_number = int(match.group(1))
                checkpoints.append((str(item), step_number))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[1])
    
    logger.info(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
    return checkpoints



def evaluate_model_and_checkpoint(
    evals: List[Evaluator],
    fine_tuner: FineTuningMethod,
    model: PreTrainedModel,
    base_path: str,
    model_name: str,
    checkpoint_info: Optional[Tuple[str, int]] = None,
    limit: int | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a model (either final or checkpoint) on all evaluations.
    
    Args:
        evals: List of evaluators
        model_adapter: Model adapter for loading
        model: The model to evaluate (can be None for checkpoints)
        model_name: Name or path of the model
        checkpoint_info: Optional tuple of (checkpoint_path, step_number)
        
    Returns:
        Dictionary of results
    """
    results = {
        "model": model_name,
        "checkpoint": checkpoint_info[1] if checkpoint_info else "final",
        "timestamp": datetime.now().isoformat()
    }
    
    # For checkpoints, we need to load the model
    if checkpoint_info:
        checkpoint_path, step_number = checkpoint_info
        try:
            # Load model from checkpoint using model adapter
            model = fine_tuner.load_model_from_checkpoint(f"{checkpoint_path}")
            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load model from checkpoint {checkpoint_path}: {str(e)}")
            results["error"] = str(e)
            return results
    
    # save model temporarily
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = f"{temp_dir}/model"
        tokenizer_path = f"{temp_dir}/tokenizer"
        logger.info(f"Saving model temporarily in {temp_dir}")
        
        model.save_pretrained(model_path)
        fine_tuner.model_adapter.load_tokenizer().save_pretrained(tokenizer_path)
        
    
        # Run all evaluations
        
        for evaluator in evals:
            try:
                logger.info(f"Running {evaluator.get_name()} on {model_name}")
        
                eval_log = evaluator.run_eval(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    base_path=base_path,
                    limit=limit,
                )
                
                if isinstance(eval_log, tuple):
                    logger.info(f"Received tuple from {evaluator.get_name()} on {model_name}")
                    success, logs = eval_log
                else:
                    logger.info(f"Received single EvalLog from {evaluator.get_name()} on {model_name}")
                    success = eval_log[0].status == "success"
                    logs = eval_log
                
                if success:
                    for log in logs:
                    # Extract metrics
                        for metric_key, metric_object in log.results.scores[0].metrics.items():
                            
                            metric_name = f"{evaluator.get_name()}_{metric_key}_{metric_object.name}"
                            
                            results[metric_name] = metric_object.value
                        
                        logger.info(f"Successfully evaluated {evaluator.get_name()}")
                    
                else:
                    logger.error(f"Evaluation failed with no success")
                    
            except Exception as e:
                logger.error(f"Error running evaluation {evaluator.get_name()}: {str(e)}")

            # Free GPU memory after each evaluator
            torch.cuda.empty_cache()
    
    return results


def evaluate(
    evals: List[Evaluator],
    fine_tuner: FineTuningMethod,
    model: PreTrainedModel,
    base_path: str,
    model_name: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate a fine-tuned model and all its checkpoints on a list of evaluations.
    
    Args:
        evals: List of evaluators to use
        model_adapter: Model adapter for the fine-tuned model
        model: Final fine-tuned model to evaluate
        checkpoint_dir: Path to checkpoint directory
        output_dir: Directory to save results
        model_name: name of the base model that got fine-tuned
        limit: optional maximum number of examples per evaluation
        
    Returns:
        DataFrame of results (also saved as CSV)
    """
    # Create output directory if it doesn't exist
    Path(base_path + "/results").mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(base_path + "/results") / f"{timestamp}_eval_results.csv"
    
    results_list = []
    
    # 1) Evaluate the final model
    logger.info("Evaluating final model...")
    final_results = evaluate_model_and_checkpoint(
        evals=evals,
        fine_tuner=fine_tuner,
        model=model,
        model_name=model_name,
        base_path=base_path,
        checkpoint_info=None,
        limit=limit,
    )
    results_list.append(final_results)
    
    # Save intermediate results
    pd.DataFrame([final_results]).to_csv(output_file, index=False)
    logger.info(f"Saved final model results to {output_file}")
    
    #2) Discover and evaluate checkpoints
    checkpoints = discover_checkpoints(f"{base_path}/checkpoints")
    
    for i, checkpoint_info in enumerate(checkpoints):
        logger.info(f"Evaluating checkpoint {i+1}/{len(checkpoints)}: {checkpoint_info[0]}")
        
        checkpoint_results = evaluate_model_and_checkpoint(
            evals=evals,
            fine_tuner=fine_tuner,
            model=None,  # Will be loaded from checkpoint
            model_name=model_name,
            base_path=base_path,
            checkpoint_info=checkpoint_info,
            limit=limit,
        )
        results_list.append(checkpoint_results)
        
        # Save intermediate results (append mode)
        pd.DataFrame([checkpoint_results]).to_csv(
            output_file,
            mode='a',
            header=False,
            index=False
        )
        logger.info(f"Appended checkpoint {checkpoint_info[1]} results")
    
    # Create final DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Sort by checkpoint (with 'final' last)
    results_df['checkpoint_numeric'] = results_df['checkpoint'].apply(
        lambda x: float('inf') if x == 'final' else int(x)
    )
    results_df = results_df.sort_values('checkpoint_numeric').drop('checkpoint_numeric', axis=1)
    
    # Save final sorted results
    results_df.to_csv(output_file.with_suffix('.final.csv'), index=False)
    
    logger.info(f"Evaluation complete. Results saved to {output_file.with_suffix('.final.csv')}")
    
    return results_df




