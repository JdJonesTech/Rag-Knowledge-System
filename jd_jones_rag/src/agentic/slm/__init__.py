"""
Specialized Small Language Models (SLMs) Module
Provides task-specific SLMs for privacy, cost, and latency optimization.

Use cases:
- SQL generation (runs locally)
- PII filtering
- Fast classification
- Document summarization
"""

from src.agentic.slm.sql_generator import SQLGeneratorSLM
from src.agentic.slm.pii_filter import PIIFilter
from src.agentic.slm.classifier import FastClassifier

__all__ = [
    "SQLGeneratorSLM",
    "PIIFilter",
    "FastClassifier"
]
