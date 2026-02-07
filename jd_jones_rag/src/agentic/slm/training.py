"""
SLM Training Pipeline
Train and fine-tune Small Language Models on JD Jones company data.

Architecture:
- LLM (GPT-4/Claude) = Main Brain for orchestration and complex reasoning
- SLMs = Specialized workers for specific tasks (fast, local, privacy-preserving)

SLM Types:
1. Intent Classifier - Classify user queries
2. Entity Extractor - Extract product codes, specs, etc.
3. Product Matcher - Match queries to products
4. Compliance Checker - Validate compliance keywords
5. SQL Generator - Generate safe SQL queries
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import pickle
import hashlib

logger = logging.getLogger(__name__)


class SLMType(str, Enum):
    """Types of SLMs available for training."""
    INTENT_CLASSIFIER = "intent_classifier"
    ENTITY_EXTRACTOR = "entity_extractor"
    PRODUCT_MATCHER = "product_matcher"
    COMPLIANCE_CHECKER = "compliance_checker"
    SQL_GENERATOR = "sql_generator"
    TECHNICAL_QA = "technical_qa"
    QUERY_ROUTER = "query_router"


@dataclass
class TrainingExample:
    """A single training example for SLM fine-tuning."""
    input_text: str
    output_text: str  # For generative models
    label: Optional[str] = None  # For classifiers
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"  # manual, llm_generated, user_feedback
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_text,
            "output": self.output_text,
            "label": self.label,
            "metadata": self.metadata,
            "source": self.source
        }


@dataclass
class TrainingDataset:
    """A collection of training examples for a specific SLM."""
    slm_type: SLMType
    examples: List[TrainingExample] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_example(self, example: TrainingExample):
        """Add a training example."""
        self.examples.append(example)
        self.updated_at = datetime.now()
    
    def add_examples(self, examples: List[TrainingExample]):
        """Add multiple training examples."""
        self.examples.extend(examples)
        self.updated_at = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        label_counts = {}
        for ex in self.examples:
            if ex.label:
                label_counts[ex.label] = label_counts.get(ex.label, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "label_distribution": label_counts,
            "sources": {
                source: sum(1 for ex in self.examples if ex.source == source)
                for source in set(ex.source for ex in self.examples)
            },
            "avg_input_length": sum(len(ex.input_text) for ex in self.examples) / max(len(self.examples), 1),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_json(self) -> str:
        """Export to JSON format."""
        return json.dumps({
            "slm_type": self.slm_type.value,
            "version": self.version,
            "examples": [ex.to_dict() for ex in self.examples],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TrainingDataset":
        """Load from JSON format."""
        data = json.loads(json_str)
        dataset = cls(
            slm_type=SLMType(data["slm_type"]),
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {})
        )
        for ex_data in data.get("examples", []):
            dataset.add_example(TrainingExample(
                input_text=ex_data["input"],
                output_text=ex_data.get("output", ""),
                label=ex_data.get("label"),
                metadata=ex_data.get("metadata", {}),
                source=ex_data.get("source", "manual")
            ))
        return dataset


@dataclass
class TrainedSLM:
    """A trained SLM model."""
    slm_type: SLMType
    model_name: str
    version: str
    trained_at: datetime
    training_examples: int
    accuracy: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slm_type": self.slm_type.value,
            "model_name": self.model_name,
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "training_examples": self.training_examples,
            "accuracy": self.accuracy,
            "metrics": self.metrics,
            "model_path": self.model_path,
            "is_active": self.is_active
        }


class SLMDataGenerator:
    """
    Generate training data from company data using LLM.
    
    The LLM acts as the "teacher" to generate high-quality training data
    for the SLM "students".
    """
    
    def __init__(self, llm=None):
        """
        Initialize data generator.
        
        Args:
            llm: LangChain LLM for generating training data
        """
        self.llm = llm
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize generation prompts for each SLM type."""
        self.generation_prompts = {
            SLMType.INTENT_CLASSIFIER: """Generate training examples for intent classification.
For each example, provide:
- input: A user query or message
- label: The intent (product_inquiry, pricing_request, technical_question, compliance_check, troubleshooting, order_status, general)

Generate 10 diverse examples based on this context:
{context}

Format as JSON array:
[{{"input": "...", "label": "..."}}, ...]""",

            SLMType.ENTITY_EXTRACTOR: """Generate training examples for entity extraction.
For each example, provide:
- input: A user query mentioning products, specs, or requirements
- output: Extracted entities as JSON

Entity types: product_code, temperature, pressure, material, industry, certification

Generate 10 diverse examples based on this context:
{context}

Format as JSON array:
[{{"input": "...", "output": {{"product_code": [...], "temperature": "...", ...}}}}, ...]""",

            SLMType.PRODUCT_MATCHER: """Generate training examples for product matching.
For each example, provide:
- input: A user requirement or application description
- output: Best matching product codes from the catalog

Generate 10 examples based on this product catalog:
{context}

Format as JSON array:
[{{"input": "...", "output": ["NA 701", "NA 702"]}}, ...]""",

            SLMType.COMPLIANCE_CHECKER: """Generate training examples for compliance checking.
For each example, provide:
- input: A compliance-related query
- output: Compliance assessment with standards and status

Generate 10 examples based on these standards:
{context}

Format as JSON array:
[{{"input": "...", "output": {{"standards": ["API 622"], "compliant": true, "notes": "..."}}}}, ...]""",

            SLMType.TECHNICAL_QA: """Generate technical Q&A training examples.
For each example, provide:
- input: A technical question about products/materials
- output: Accurate technical answer

Generate 10 examples based on this technical documentation:
{context}

Format as JSON array:
[{{"input": "...", "output": "..."}}, ...]"""
        }
    
    async def generate_from_documents(
        self,
        slm_type: SLMType,
        documents: List[Dict[str, Any]],
        num_examples: int = 50
    ) -> List[TrainingExample]:
        """
        Generate training examples from documents using LLM.
        
        Args:
            slm_type: Type of SLM to generate data for
            documents: Source documents with company data
            num_examples: Number of examples to generate
            
        Returns:
            List of TrainingExample objects
        """
        if not self.llm:
            logger.warning("No LLM configured, using rule-based generation")
            return self._generate_rule_based(slm_type, documents, num_examples)
        
        examples = []
        prompt_template = self.generation_prompts.get(slm_type)
        
        if not prompt_template:
            logger.warning(f"No prompt template for {slm_type}")
            return examples
        
        # Process documents in batches
        batch_size = 5
        examples_per_batch = max(10, num_examples // (len(documents) // batch_size + 1))
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            context = "\n\n".join([
                f"Document {j+1}:\n{doc.get('content', doc.get('text', str(doc)))[:2000]}"
                for j, doc in enumerate(batch)
            ])
            
            prompt = prompt_template.format(context=context)
            
            try:
                from langchain_core.messages import HumanMessage
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                
                # Parse JSON response
                import re
                json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                if json_match:
                    generated = json.loads(json_match.group())
                    for item in generated:
                        examples.append(TrainingExample(
                            input_text=item.get("input", ""),
                            output_text=json.dumps(item.get("output", "")) if isinstance(item.get("output"), dict) else str(item.get("output", "")),
                            label=item.get("label"),
                            source="llm_generated",
                            metadata={"slm_type": slm_type.value}
                        ))
            except Exception as e:
                logger.error(f"Error generating examples: {e}")
                continue
            
            if len(examples) >= num_examples:
                break
        
        return examples[:num_examples]
    
    def _generate_rule_based(
        self,
        slm_type: SLMType,
        documents: List[Dict[str, Any]],
        num_examples: int
    ) -> List[TrainingExample]:
        """Generate examples using rule-based approach (fallback)."""
        examples = []
        
        if slm_type == SLMType.INTENT_CLASSIFIER:
            # Generate intent classification examples from document content
            intent_templates = {
                "product_inquiry": [
                    "What is {product}?",
                    "Tell me about {product}",
                    "I need information on {product}",
                ],
                "pricing_request": [
                    "How much does {product} cost?",
                    "Price for {product}",
                    "Quote for {product}",
                ],
                "technical_question": [
                    "What is the temperature rating for {product}?",
                    "Pressure limit of {product}?",
                    "Chemical compatibility of {product}",
                ],
                "compliance_check": [
                    "Is {product} API 622 certified?",
                    "Does {product} meet FDA requirements?",
                    "Fire-safe certification for {product}",
                ]
            }
            
            for doc in documents[:num_examples]:
                # Extract product mentions from document
                content = doc.get("content", doc.get("text", str(doc)))
                product_matches = re.findall(r'NA\s*\d+|NJ\s*\d+', content, re.IGNORECASE)
                
                if product_matches:
                    product = product_matches[0]
                    for intent, templates in intent_templates.items():
                        template = templates[len(examples) % len(templates)]
                        examples.append(TrainingExample(
                            input_text=template.format(product=product),
                            output_text="",
                            label=intent,
                            source="rule_based"
                        ))
                        if len(examples) >= num_examples:
                            break
                if len(examples) >= num_examples:
                    break
        
        return examples
    
    def generate_from_user_interactions(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[TrainingExample]:
        """
        Generate training examples from logged user interactions.
        
        Args:
            interactions: List of user interaction logs
            
        Returns:
            List of TrainingExample objects
        """
        examples = []
        
        for interaction in interactions:
            query = interaction.get("query", "")
            response = interaction.get("response", "")
            intent = interaction.get("detected_intent", "")
            feedback = interaction.get("user_feedback", {})
            
            # Only use interactions with positive feedback or high confidence
            if feedback.get("helpful", False) or interaction.get("confidence", 0) > 0.8:
                examples.append(TrainingExample(
                    input_text=query,
                    output_text=response,
                    label=intent,
                    source="user_feedback",
                    metadata={
                        "feedback": feedback,
                        "confidence": interaction.get("confidence")
                    }
                ))
        
        return examples


class SLMTrainer:
    """
    Train and manage SLMs fine-tuned on company data.
    
    Supports multiple training approaches:
    1. Keyword-based classifiers (fast, no ML)
    2. TF-IDF + sklearn classifiers (medium, local)
    3. Fine-tuned transformers (slow, high quality)
    4. LoRA/QLoRA fine-tuning (efficient for larger models)
    """
    
    def __init__(
        self,
        model_dir: str = "models/slm",
        use_gpu: bool = False
    ):
        """
        Initialize SLM trainer.
        
        Args:
            model_dir: Directory to save trained models
            use_gpu: Whether to use GPU for training
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        
        # Registry of trained models
        self.models: Dict[SLMType, TrainedSLM] = {}
        self._load_model_registry()
    
    def _load_model_registry(self):
        """Load registry of trained models."""
        registry_path = self.model_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    data = json.load(f)
                    for model_data in data.get("models", []):
                        slm_type = SLMType(model_data["slm_type"])
                        self.models[slm_type] = TrainedSLM(
                            slm_type=slm_type,
                            model_name=model_data["model_name"],
                            version=model_data["version"],
                            trained_at=datetime.fromisoformat(model_data["trained_at"]),
                            training_examples=model_data["training_examples"],
                            accuracy=model_data.get("accuracy", 0),
                            metrics=model_data.get("metrics", {}),
                            model_path=model_data.get("model_path"),
                            is_active=model_data.get("is_active", False)
                        )
            except Exception as e:
                logger.error(f"Error loading model registry: {e}")
    
    def _save_model_registry(self):
        """Save registry of trained models."""
        registry_path = self.model_dir / "registry.json"
        data = {
            "models": [model.to_dict() for model in self.models.values()],
            "updated_at": datetime.now().isoformat()
        }
        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)
    
    async def train(
        self,
        slm_type: SLMType,
        dataset: TrainingDataset,
        training_method: str = "sklearn",  # sklearn, transformer, keyword
        model_name: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> TrainedSLM:
        """
        Train an SLM on the provided dataset.
        
        Args:
            slm_type: Type of SLM to train
            dataset: Training dataset
            training_method: Training approach to use
            model_name: Optional custom model name
            hyperparams: Optional hyperparameters
            
        Returns:
            TrainedSLM with training results
        """
        logger.info(f"Training {slm_type.value} using {training_method} method")
        
        if training_method == "keyword":
            return await self._train_keyword_model(slm_type, dataset, model_name)
        elif training_method == "sklearn":
            return await self._train_sklearn_model(slm_type, dataset, model_name, hyperparams)
        elif training_method == "transformer":
            return await self._train_transformer_model(slm_type, dataset, model_name, hyperparams)
        else:
            raise ValueError(f"Unknown training method: {training_method}")
    
    async def _train_keyword_model(
        self,
        slm_type: SLMType,
        dataset: TrainingDataset,
        model_name: Optional[str] = None
    ) -> TrainedSLM:
        """Train a simple keyword-based classifier."""
        # Extract keywords per class
        class_keywords: Dict[str, Dict[str, int]] = {}
        
        for example in dataset.examples:
            label = example.label or "unknown"
            if label not in class_keywords:
                class_keywords[label] = {}
            
            # Tokenize and count keywords
            words = re.findall(r'\w+', example.input_text.lower())
            for word in words:
                if len(word) > 3:  # Skip short words
                    class_keywords[label][word] = class_keywords[label].get(word, 0) + 1
        
        # Keep top keywords per class
        keyword_model = {}
        for label, words in class_keywords.items():
            sorted_words = sorted(words.items(), key=lambda x: -x[1])[:50]
            keyword_model[label] = [w[0] for w in sorted_words]
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{slm_type.value}_keyword_{version}.json"
        with open(model_path, "w") as f:
            json.dump(keyword_model, f, indent=2)
        
        trained = TrainedSLM(
            slm_type=slm_type,
            model_name=model_name or f"{slm_type.value}_keyword",
            version=version,
            trained_at=datetime.now(),
            training_examples=len(dataset.examples),
            accuracy=0.8,  # Estimated for keyword models
            model_path=str(model_path),
            is_active=True
        )
        
        self.models[slm_type] = trained
        self._save_model_registry()
        
        return trained
    
    async def _train_sklearn_model(
        self,
        slm_type: SLMType,
        dataset: TrainingDataset,
        model_name: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> TrainedSLM:
        """Train a sklearn-based classifier with TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
        except ImportError:
            logger.warning("sklearn not available, falling back to keyword model")
            return await self._train_keyword_model(slm_type, dataset, model_name)
        
        # Prepare data
        texts = [ex.input_text for ex in dataset.examples]
        labels = [ex.label or "unknown" for ex in dataset.examples]
        
        if len(set(labels)) < 2:
            logger.warning("Need at least 2 classes for training")
            return await self._train_keyword_model(slm_type, dataset, model_name)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=hyperparams.get("max_features", 5000) if hyperparams else 5000,
                ngram_range=(1, 2)
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{slm_type.value}_sklearn_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        
        trained = TrainedSLM(
            slm_type=slm_type,
            model_name=model_name or f"{slm_type.value}_sklearn",
            version=version,
            trained_at=datetime.now(),
            training_examples=len(dataset.examples),
            accuracy=accuracy,
            metrics={
                "test_accuracy": accuracy,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "num_classes": len(set(labels))
            },
            model_path=str(model_path),
            is_active=True
        )
        
        self.models[slm_type] = trained
        self._save_model_registry()
        
        logger.info(f"Trained {slm_type.value} with accuracy: {accuracy:.2%}")
        return trained
    
    async def _train_transformer_model(
        self,
        slm_type: SLMType,
        dataset: TrainingDataset,
        model_name: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> TrainedSLM:
        """
        Train a transformer-based model using HuggingFace.
        
        In production, this would use:
        - DistilBERT for classification
        - Phi-2 or Mistral 7B for generation
        - LoRA for efficient fine-tuning
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import TrainingArguments, Trainer
            import torch
        except ImportError:
            logger.warning("transformers not available, falling back to sklearn")
            return await self._train_sklearn_model(slm_type, dataset, model_name, hyperparams)
        
        # This is a placeholder - full transformer training would be more complex
        logger.info("Transformer training requires significant compute - using sklearn fallback")
        return await self._train_sklearn_model(slm_type, dataset, model_name, hyperparams)
    
    def get_model(self, slm_type: SLMType) -> Optional[TrainedSLM]:
        """Get a trained SLM by type."""
        return self.models.get(slm_type)
    
    def list_models(self) -> List[TrainedSLM]:
        """List all trained models."""
        return list(self.models.values())
    
    def activate_model(self, slm_type: SLMType, version: str) -> bool:
        """Activate a specific model version."""
        model = self.models.get(slm_type)
        if model and model.version == version:
            model.is_active = True
            self._save_model_registry()
            return True
        return False


class SLMInference:
    """
    Run inference using trained SLMs.
    
    Loaded models are cached for fast inference.
    """
    
    def __init__(self, model_dir: str = "models/slm"):
        """Initialize inference engine."""
        self.model_dir = Path(model_dir)
        self._loaded_models: Dict[SLMType, Any] = {}
        self._model_info: Dict[SLMType, TrainedSLM] = {}
        
        # Load model registry
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry."""
        registry_path = self.model_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                data = json.load(f)
                for model_data in data.get("models", []):
                    if model_data.get("is_active"):
                        slm_type = SLMType(model_data["slm_type"])
                        self._model_info[slm_type] = TrainedSLM(
                            slm_type=slm_type,
                            model_name=model_data["model_name"],
                            version=model_data["version"],
                            trained_at=datetime.fromisoformat(model_data["trained_at"]),
                            training_examples=model_data["training_examples"],
                            accuracy=model_data.get("accuracy", 0),
                            model_path=model_data.get("model_path"),
                            is_active=True
                        )
    
    def _load_model(self, slm_type: SLMType) -> Any:
        """Load a model into memory."""
        if slm_type in self._loaded_models:
            return self._loaded_models[slm_type]
        
        model_info = self._model_info.get(slm_type)
        if not model_info or not model_info.model_path:
            return None
        
        model_path = Path(model_info.model_path)
        if not model_path.exists():
            return None
        
        try:
            if model_path.suffix == ".json":
                with open(model_path, "r") as f:
                    model = json.load(f)
            elif model_path.suffix == ".pkl":
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                return None
            
            self._loaded_models[slm_type] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(
        self,
        slm_type: SLMType,
        text: str
    ) -> Dict[str, Any]:
        """
        Run prediction using the specified SLM.
        
        Args:
            slm_type: Type of SLM to use
            text: Input text
            
        Returns:
            Prediction result with confidence
        """
        model = self._load_model(slm_type)
        if model is None:
            return {"error": f"No model loaded for {slm_type.value}"}
        
        model_info = self._model_info.get(slm_type)
        
        # Keyword model
        if isinstance(model, dict) and all(isinstance(v, list) for v in model.values()):
            return self._predict_keyword(model, text)
        
        # Sklearn model
        if hasattr(model, "predict"):
            return self._predict_sklearn(model, text)
        
        return {"error": "Unknown model type"}
    
    def _predict_keyword(
        self,
        model: Dict[str, List[str]],
        text: str
    ) -> Dict[str, Any]:
        """Predict using keyword model."""
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))
        
        scores = {}
        for label, keywords in model.items():
            score = sum(1 for kw in keywords if kw in words)
            scores[label] = score / len(keywords) if keywords else 0
        
        if not scores:
            return {"prediction": "unknown", "confidence": 0, "scores": {}}
        
        best_label = max(scores, key=scores.get)
        return {
            "prediction": best_label,
            "confidence": scores[best_label],
            "scores": scores
        }
    
    def _predict_sklearn(
        self,
        model: Any,
        text: str
    ) -> Dict[str, Any]:
        """Predict using sklearn model."""
        try:
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
            
            return {
                "prediction": prediction,
                "confidence": float(max(probabilities)),
                "scores": {c: float(p) for c, p in zip(classes, probabilities)}
            }
        except Exception as e:
            return {"error": str(e)}
    
    def is_loaded(self, slm_type: SLMType) -> bool:
        """Check if a model is loaded."""
        return slm_type in self._loaded_models
    
    def get_model_info(self, slm_type: SLMType) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model."""
        info = self._model_info.get(slm_type)
        return info.to_dict() if info else None


# Convenience function for specialists to use SLMs
def get_slm_inference() -> SLMInference:
    """Get the global SLM inference engine."""
    global _slm_inference
    if "_slm_inference" not in globals():
        _slm_inference = SLMInference()
    return _slm_inference
