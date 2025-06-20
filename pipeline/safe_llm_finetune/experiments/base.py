"""
Abstract base classes and utilities for experiment management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from safe_llm_finetune.datasets.base import Dataset
from safe_llm_finetune.evaluation.base import Evaluator, SafetyEvaluator
from safe_llm_finetune.fine_tuning.base import FineTuningMethod, ModelAdapter, TrainingConfig


@dataclass
class ExperimentConfig:
    """Configuration for experiment."""

    experiment_name: str
    model_id: str
    training_dataset_path: str
    eval_dataset_path: str
    output_dir: Path
    training_config: TrainingConfig
    tracking_enabled: bool = True
    save_artifacts: bool = True
    description: Optional[str] = None


@dataclass
class ExperimentResult:
    """Results of an experiment."""

    experiment_name: str
    model_id: str
    fine_tuning_method: str
    start_time: datetime
    end_time: datetime
    training_metrics: Dict[str, float]
    evaluation_metrics: Dict[str, float]
    safety_metrics: Dict[str, Any]
    model_path: Path
    config: ExperimentConfig


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def start_experiment(self, config: ExperimentConfig) -> str:
        """
        Start tracking an experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        pass

    @abstractmethod
    def log_metrics(
        self, experiment_id: str, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics during experiment.

        Args:
            experiment_id: ID of the experiment
            metrics: Metrics to log
            step: Optional step number
        """
        pass

    @abstractmethod
    def log_artifact(self, experiment_id: str, artifact_path: Path, name: str) -> None:
        """
        Log an artifact.

        Args:
            experiment_id: ID of the experiment
            artifact_path: Path to the artifact
            name: Name of the artifact
        """
        pass

    @abstractmethod
    def end_experiment(self, experiment_id: str, status: str = "COMPLETED") -> None:
        """
        End tracking an experiment.

        Args:
            experiment_id: ID of the experiment
            status: Status of the experiment
        """
        pass


class Experiment(ABC):
    """Abstract base class for experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        model_adapter: ModelAdapter,
        fine_tuning_method: FineTuningMethod,
        training_dataset: Dataset,
        eval_dataset: Dataset,
        evaluator: Evaluator,
        safety_evaluator: SafetyEvaluator,
        tracker: Optional[ExperimentTracker] = None,
    ):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration
            model_adapter: Model adapter to use
            fine_tuning_method: Fine-tuning method to use
            training_dataset: Training dataset
            eval_dataset: Evaluation dataset
            evaluator: Model evaluator
            safety_evaluator: Safety evaluator
            tracker: Optional experiment tracker
        """
        self.config = config
        self.model_adapter = model_adapter
        self.fine_tuning_method = fine_tuning_method
        self.training_dataset = training_dataset
        self.eval_dataset = eval_dataset
        self.evaluator = evaluator
        self.safety_evaluator = safety_evaluator
        self.tracker = tracker

    def setup(self) -> None:
        """Set up the experiment."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if self.tracker and self.config.tracking_enabled:
            self.experiment_id = self.tracker.start_experiment(self.config)

    @abstractmethod
    def run(self) -> ExperimentResult:
        """
        Run the experiment.

        Returns:
            Experiment results
        """
        pass

    def teardown(self) -> None:
        """Tear down the experiment."""
        if self.tracker and self.config.tracking_enabled:
            self.tracker.end_experiment(self.experiment_id)
