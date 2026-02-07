"""
Fine-Tuning Data Preparation
Prepares domain-specific data for model fine-tuning.
Handles data cleaning, formatting, and validation.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
import uuid


class DataFormat(str, Enum):
    """Supported data formats for fine-tuning."""
    OPENAI = "openai"           # OpenAI fine-tuning format
    ANTHROPIC = "anthropic"     # Anthropic fine-tuning format
    ALPACA = "alpaca"           # Alpaca/Stanford format
    SHAREGPT = "sharegpt"       # ShareGPT format
    CUSTOM = "custom"           # Custom format


@dataclass
class FineTuningExample:
    """A single fine-tuning example."""
    id: str
    system_prompt: Optional[str]
    user_input: str
    assistant_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI fine-tuning format."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_input})
        messages.append({"role": "assistant", "content": self.assistant_output})
        return {"messages": messages}
    
    def to_alpaca_format(self) -> Dict[str, Any]:
        """Convert to Alpaca format."""
        return {
            "instruction": self.user_input,
            "input": "",
            "output": self.assistant_output
        }


@dataclass
class FineTuningDataset:
    """A dataset for fine-tuning."""
    name: str
    description: str
    examples: List[FineTuningExample]
    format: DataFormat
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "example_count": len(self.examples),
            "format": self.format.value,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }


class FineTuningDataPreparer:
    """
    Prepares data for fine-tuning language models.
    
    Workflow:
    1. Collect domain-specific Q&A pairs
    2. Clean and validate data
    3. Format for target platform
    4. Split into train/validation sets
    5. Export for fine-tuning
    
    Data Sources:
    - Historical chat logs
    - FAQ documents
    - Technical documentation
    - Expert-curated examples
    """
    
    # JD Jones domain-specific terminology
    DOMAIN_TERMS = {
        "gasket": "A mechanical seal that fills the space between two mating surfaces",
        "packing": "Sealing material used around rotating or reciprocating shafts",
        "fugitive emissions": "Unintentional release of gases or vapors from equipment",
        "API 622": "Standard for testing packing materials in rising stem valves",
        "API 624": "Standard for fugitive emission testing of rising stem valves",
        "PACMAAN": "JD Jones premium gasket product line",
        "FLEXSEAL": "JD Jones flexible sealing solutions product line",
        "spiral wound": "A gasket type made of alternating layers of metal and filler",
        "PTFE": "Polytetrafluoroethylene, a synthetic fluoropolymer",
        "graphite": "A carbon-based material used in high-temperature sealing"
    }
    
    # System prompt for JD Jones assistant
    DEFAULT_SYSTEM_PROMPT = """You are a technical assistant for JD Jones, a manufacturer of industrial sealing solutions including gaskets, packings, and expansion joints.

Key guidelines:
- Provide accurate technical information about sealing products
- Reference industry standards (API, ASME, ISO) when relevant
- Recommend appropriate products based on application requirements
- Prioritize safety and compliance in all recommendations
- Be helpful but never make claims outside your knowledge"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        min_input_length: int = 10,
        max_input_length: int = 2000,
        min_output_length: int = 20,
        max_output_length: int = 4000
    ):
        """
        Initialize data preparer.
        
        Args:
            system_prompt: System prompt to use
            min_input_length: Minimum input length
            max_input_length: Maximum input length
            min_output_length: Minimum output length
            max_output_length: Maximum output length
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.min_input_length = min_input_length
        self.max_input_length = max_input_length
        self.min_output_length = min_output_length
        self.max_output_length = max_output_length
        
        self.examples: List[FineTuningExample] = []
        self.validation_errors: List[Dict[str, Any]] = []
    
    def add_example(
        self,
        user_input: str,
        assistant_output: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[FineTuningExample]:
        """
        Add a training example.
        
        Args:
            user_input: User's input/question
            assistant_output: Expected assistant response
            system_prompt: Optional custom system prompt
            metadata: Optional metadata
            
        Returns:
            The example if valid, None otherwise
        """
        # Clean inputs
        user_input = self._clean_text(user_input)
        assistant_output = self._clean_text(assistant_output)
        
        # Validate
        validation = self._validate_example(user_input, assistant_output)
        if not validation["valid"]:
            self.validation_errors.append({
                "input_preview": user_input[:100],
                "errors": validation["errors"]
            })
            return None
        
        example = FineTuningExample(
            id=f"ex_{uuid.uuid4().hex[:12]}",
            system_prompt=system_prompt or self.system_prompt,
            user_input=user_input,
            assistant_output=assistant_output,
            metadata=metadata or {}
        )
        
        self.examples.append(example)
        return example
    
    def add_from_conversation(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add examples from a conversation history.
        
        Args:
            messages: List of messages with 'role' and 'content'
            metadata: Optional metadata
            
        Returns:
            Number of examples added
        """
        added = 0
        
        # Find user-assistant pairs
        i = 0
        while i < len(messages) - 1:
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                example = self.add_example(
                    user_input=messages[i]["content"],
                    assistant_output=messages[i+1]["content"],
                    metadata=metadata
                )
                if example:
                    added += 1
            i += 1
        
        return added
    
    def add_from_faq(
        self,
        faqs: List[Dict[str, str]],
        enhance: bool = True
    ) -> int:
        """
        Add examples from FAQ documents.
        
        Args:
            faqs: List of FAQ items with 'question' and 'answer'
            enhance: Whether to enhance with variations
            
        Returns:
            Number of examples added
        """
        added = 0
        
        for faq in faqs:
            question = faq.get("question", "")
            answer = faq.get("answer", "")
            
            # Add original
            example = self.add_example(question, answer)
            if example:
                added += 1
            
            # Add variations if enhancing
            if enhance:
                variations = self._generate_question_variations(question)
                for var in variations:
                    ex = self.add_example(var, answer)
                    if ex:
                        added += 1
        
        return added
    
    def add_domain_terminology_examples(self) -> int:
        """Add examples for domain-specific terminology."""
        added = 0
        
        for term, definition in self.DOMAIN_TERMS.items():
            # "What is X?" format
            question = f"What is {term}?"
            answer = f"{term.title()} refers to {definition}."
            if self.add_example(question, answer):
                added += 1
            
            # "Define X" format
            question2 = f"Can you define {term}?"
            if self.add_example(question2, answer):
                added += 1
            
            # "Explain X" format
            question3 = f"Please explain what {term} means."
            if self.add_example(question3, answer):
                added += 1
        
        return added
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    def _validate_example(
        self,
        user_input: str,
        assistant_output: str
    ) -> Dict[str, Any]:
        """Validate an example."""
        errors = []
        
        # Length checks
        if len(user_input) < self.min_input_length:
            errors.append(f"Input too short: {len(user_input)} < {self.min_input_length}")
        if len(user_input) > self.max_input_length:
            errors.append(f"Input too long: {len(user_input)} > {self.max_input_length}")
        if len(assistant_output) < self.min_output_length:
            errors.append(f"Output too short: {len(assistant_output)} < {self.min_output_length}")
        if len(assistant_output) > self.max_output_length:
            errors.append(f"Output too long: {len(assistant_output)} > {self.max_output_length}")
        
        # Content checks
        if not user_input.strip():
            errors.append("Empty input")
        if not assistant_output.strip():
            errors.append("Empty output")
        
        # Check for potential issues
        if "I don't know" in assistant_output and len(assistant_output) < 50:
            errors.append("Low-quality response (just 'I don't know')")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _generate_question_variations(self, question: str) -> List[str]:
        """Generate variations of a question."""
        variations = []
        
        # Simple variations
        if question.startswith("What is"):
            variations.append(question.replace("What is", "Can you explain"))
            variations.append(question.replace("What is", "Tell me about"))
        elif question.startswith("How"):
            variations.append(question.replace("How", "What is the process for"))
        
        return variations[:2]  # Limit variations
    
    def create_dataset(
        self,
        name: str,
        description: str,
        format: DataFormat = DataFormat.OPENAI
    ) -> FineTuningDataset:
        """
        Create a dataset from collected examples.
        
        Args:
            name: Dataset name
            description: Dataset description
            format: Output format
            
        Returns:
            FineTuningDataset
        """
        return FineTuningDataset(
            name=name,
            description=description,
            examples=self.examples.copy(),
            format=format
        )
    
    def split_dataset(
        self,
        dataset: FineTuningDataset,
        validation_ratio: float = 0.1
    ) -> Tuple[FineTuningDataset, FineTuningDataset]:
        """
        Split dataset into training and validation sets.
        
        Args:
            dataset: Dataset to split
            validation_ratio: Ratio for validation set
            
        Returns:
            Tuple of (training_dataset, validation_dataset)
        """
        import random
        
        examples = dataset.examples.copy()
        random.shuffle(examples)
        
        split_idx = int(len(examples) * (1 - validation_ratio))
        
        train_dataset = FineTuningDataset(
            name=f"{dataset.name}_train",
            description=f"Training split of {dataset.name}",
            examples=examples[:split_idx],
            format=dataset.format
        )
        
        val_dataset = FineTuningDataset(
            name=f"{dataset.name}_val",
            description=f"Validation split of {dataset.name}",
            examples=examples[split_idx:],
            format=dataset.format
        )
        
        return train_dataset, val_dataset
    
    def export_dataset(
        self,
        dataset: FineTuningDataset,
        output_path: str
    ) -> str:
        """
        Export dataset to file.
        
        Args:
            dataset: Dataset to export
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        if dataset.format == DataFormat.OPENAI:
            # JSONL format for OpenAI
            with open(output_path, 'w') as f:
                for example in dataset.examples:
                    f.write(json.dumps(example.to_openai_format()) + '\n')
        
        elif dataset.format == DataFormat.ALPACA:
            # JSON array for Alpaca
            data = [ex.to_alpaca_format() for ex in dataset.examples]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            # Generic JSON
            data = [ex.to_openai_format() for ex in dataset.examples]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return output_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preparation statistics."""
        if not self.examples:
            return {"total_examples": 0}
        
        input_lengths = [len(ex.user_input) for ex in self.examples]
        output_lengths = [len(ex.assistant_output) for ex in self.examples]
        
        return {
            "total_examples": len(self.examples),
            "validation_errors": len(self.validation_errors),
            "input_length": {
                "min": min(input_lengths),
                "max": max(input_lengths),
                "avg": sum(input_lengths) / len(input_lengths)
            },
            "output_length": {
                "min": min(output_lengths),
                "max": max(output_lengths),
                "avg": sum(output_lengths) / len(output_lengths)
            }
        }
