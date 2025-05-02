import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from safe_llm_finetune.fine_tuning.base import CheckpointConfig


class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: PreTrainedModel, 
                       tokenizer: PreTrainedTokenizer,
                       step: int,
                       epoch: Optional[int] = None,
                       optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save a training checkpoint."""
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        state_dict = {
            "step": step,
            "epoch": epoch,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "metadata": metadata or {}
        }
        
        torch.save(state_dict, checkpoint_path / "training_state.pt")
        
        # Save config
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Remove old checkpoints if exceeding limit
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load training state
        state_dict = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
        
        return state_dict
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints exceeding the save limit."""
        if self.config.save_total_limit <= 0:
            return
        
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        if len(checkpoints) <= self.config.save_total_limit:
            return
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.config.save_total_limit]:
            import shutil
            shutil.rmtree(checkpoint)