"""
SLM Configuration for JD Jones RAG
Defines which models to use for different tasks.

ARCHITECTURE:
=============
┌─────────────────────────────────────────────────────────┐
│                 MAIN BRAIN (LLM)                        │
│                   Llama 3.2                             │
│  • Complex reasoning & orchestration                    │
│  • Multi-step problem solving                           │
│  • Response synthesis                                   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   SKLEARN       │ │   TINYLLAMA     │ │   TINYLLAMA     │
│ Classification  │ │  Entity Extrac  │ │  Generation     │
│                 │ │                 │ │                 │
│ • Intent        │ │ • Product codes │ │ • Quick answers │
│ • Urgency       │ │ • Specs extract │ │ • Summaries     │
│ • Sentiment     │ │ • Standards     │ │ • Formatting    │
│ < 10ms          │ │ ~200ms          │ │ ~200ms          │
└─────────────────┘ └─────────────────┘ └─────────────────┘
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class SLMBackend(str, Enum):
    """Backend options for SLM tasks."""
    SKLEARN = "sklearn"       # Fast classification (< 10ms)
    TINYLLAMA = "tinyllama"   # Small Ollama model (~200ms)
    LLAMA = "llama3.2"        # Main brain (500ms+)


class TaskType(str, Enum):
    """Types of SLM tasks."""
    CLASSIFICATION = "classification"        # Use sklearn
    ENTITY_EXTRACTION = "entity_extraction"  # Use tinyllama  
    GENERATION = "generation"                # Use tinyllama or llama
    REASONING = "reasoning"                  # Use llama (main brain)


@dataclass
class SLMConfig:
    """Configuration for SLM system."""
    
    # Main brain configuration
    main_llm: str = "llama3.2"
    main_llm_base_url: str = "http://localhost:11434/v1"
    
    # SLM model for generation/extraction tasks
    slm_model: str = "tinyllama"
    slm_base_url: str = "http://localhost:11434/v1"
    
    # Task routing
    task_backends: Dict[TaskType, SLMBackend] = field(default_factory=lambda: {
        TaskType.CLASSIFICATION: SLMBackend.SKLEARN,
        TaskType.ENTITY_EXTRACTION: SLMBackend.TINYLLAMA,
        TaskType.GENERATION: SLMBackend.TINYLLAMA,
        TaskType.REASONING: SLMBackend.LLAMA
    })
    
    # Temperature settings
    classification_temperature: float = 0.0  # Deterministic for classification
    extraction_temperature: float = 0.1      # Low for accuracy
    generation_temperature: float = 0.3      # Slightly creative
    reasoning_temperature: float = 0.7       # Allow creativity for main brain
    
    # Timeouts (seconds)
    sklearn_timeout: float = 1.0
    tinyllama_timeout: float = 30.0
    llama_timeout: float = 120.0
    
    def get_backend(self, task: TaskType) -> SLMBackend:
        """Get the appropriate backend for a task."""
        return self.task_backends.get(task, SLMBackend.LLAMA)
    
    def get_model_for_task(self, task: TaskType) -> str:
        """Get the model name for a task type."""
        backend = self.get_backend(task)
        if backend == SLMBackend.SKLEARN:
            return "sklearn"
        elif backend == SLMBackend.TINYLLAMA:
            return self.slm_model
        else:
            return self.main_llm
    
    def get_temperature(self, task: TaskType) -> float:
        """Get temperature for a task type."""
        if task == TaskType.CLASSIFICATION:
            return self.classification_temperature
        elif task == TaskType.ENTITY_EXTRACTION:
            return self.extraction_temperature
        elif task == TaskType.GENERATION:
            return self.generation_temperature
        else:
            return self.reasoning_temperature


# Default configuration
DEFAULT_CONFIG = SLMConfig()


# Classification tasks that use sklearn
SKLEARN_TASKS = [
    "intent_classification",
    "urgency_detection", 
    "sentiment_analysis",
    "category_classification",
    "compliance_flag_detection"
]

# Extraction tasks that use TinyLlama
TINYLLAMA_EXTRACTION_TASKS = [
    "product_code_extraction",
    "specification_extraction",
    "standard_identification",
    "entity_extraction",
    "parameter_extraction"
]

# Generation tasks that use TinyLlama
TINYLLAMA_GENERATION_TASKS = [
    "quick_answer",
    "summary_generation",
    "format_conversion",
    "simple_qa"
]

# Complex tasks that use main Llama brain
LLAMA_BRAIN_TASKS = [
    "complex_reasoning",
    "multi_step_planning",
    "tool_orchestration",
    "response_synthesis",
    "cross_reference_analysis"
]


def get_task_type(task_name: str) -> TaskType:
    """Determine task type from task name."""
    if task_name in SKLEARN_TASKS:
        return TaskType.CLASSIFICATION
    elif task_name in TINYLLAMA_EXTRACTION_TASKS:
        return TaskType.ENTITY_EXTRACTION
    elif task_name in TINYLLAMA_GENERATION_TASKS:
        return TaskType.GENERATION
    else:
        return TaskType.REASONING


def get_model_for_task(task_name: str, config: Optional[SLMConfig] = None) -> Dict[str, Any]:
    """
    Get model configuration for a specific task.
    
    Returns dict with:
    - model: Model name
    - backend: Backend type (sklearn, ollama)
    - temperature: Suggested temperature
    - timeout: Timeout in seconds
    """
    config = config or DEFAULT_CONFIG
    task_type = get_task_type(task_name)
    backend = config.get_backend(task_type)
    
    return {
        "model": config.get_model_for_task(task_type),
        "backend": backend.value,
        "temperature": config.get_temperature(task_type),
        "timeout": {
            SLMBackend.SKLEARN: config.sklearn_timeout,
            SLMBackend.TINYLLAMA: config.tinyllama_timeout,
            SLMBackend.LLAMA: config.llama_timeout
        }.get(backend, config.llama_timeout),
        "task_type": task_type.value
    }


# Print architecture summary
def print_architecture():
    """Print the SLM architecture summary."""
    print("""
JD JONES RAG - AI MODEL ARCHITECTURE
====================================

MAIN BRAIN (LLM): Llama 3.2
---------------------------
  Role: Central orchestrator for complex reasoning
  Tasks:
    - Multi-step problem solving
    - Tool selection and orchestration  
    - Response synthesis
    - Cross-reference analysis
  Latency: 500ms - 2s

SLM LAYER 1: sklearn (Classification)
-------------------------------------
  Role: Fast classification tasks
  Tasks:
    - Intent classification
    - Urgency detection
    - Sentiment analysis
    - Category classification
  Latency: < 10ms

SLM LAYER 2: TinyLlama (Extraction & Generation)
-------------------------------------------------
  Role: Local LLM for extraction and quick generation
  Tasks:
    - Entity extraction (product codes, specs)
    - Standard identification
    - Quick answer generation
    - Summary generation
  Latency: ~200ms

WORKFLOW:
---------
1. Query arrives
2. sklearn classifies intent (< 10ms)
3. TinyLlama extracts entities (< 200ms)
4. IF simple query: TinyLlama handles directly
5. IF complex: Escalate to Llama 3.2 main brain
6. Response returned

BENEFITS:
---------
- 60-70% queries handled by SLMs (< 250ms total)
- Llama 3.2 reserved for complex reasoning
- sklearn trained on YOUR company data
- Privacy-preserving (all local)
""")


if __name__ == "__main__":
    print_architecture()
