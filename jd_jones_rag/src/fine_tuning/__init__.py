"""
Fine-Tuning Module
Provides capabilities to fine-tune models on domain-specific data.
Used when specialized terminology is not understood by base models.

Note: Fine-tuning is OPTIONAL - rely on RAG for factual accuracy.
Fine-tuning is primarily for:
- Domain-specific terminology understanding
- Response style and formatting
- Task-specific optimizations
"""

from src.fine_tuning.data_preparation import FineTuningDataPreparer
from src.fine_tuning.trainer import ModelTrainer
from src.fine_tuning.evaluation import ModelEvaluator

__all__ = [
    "FineTuningDataPreparer",
    "ModelTrainer",
    "ModelEvaluator"
]
