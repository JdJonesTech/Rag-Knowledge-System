"""
Model Evaluator
Evaluates fine-tuned models for quality and performance.
Compares against base models and validates domain understanding.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from src.fine_tuning.data_preparation import FineTuningDataset, FineTuningExample


class EvaluationMetric(str, Enum):
    """Evaluation metrics."""
    ACCURACY = "accuracy"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FACTUAL_ACCURACY = "factual_accuracy"
    DOMAIN_UNDERSTANDING = "domain_understanding"


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    example_id: str
    input_text: str
    expected_output: str
    model_output: str
    metrics: Dict[str, float]
    passed: bool
    notes: str = ""


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    model_id: str
    dataset_name: str
    timestamp: datetime
    results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float]
    summary: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "total_examples": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "aggregate_metrics": self.aggregate_metrics,
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class ModelEvaluator:
    """
    Evaluates fine-tuned models.
    
    Evaluation Types:
    1. Automated metrics (BLEU, ROUGE, perplexity)
    2. Semantic similarity with expected outputs
    3. Domain-specific validation
    4. A/B comparison with base model
    5. Factual accuracy checking
    
    Best Practices:
    - Always evaluate on held-out test set
    - Compare with base model performance
    - Check for regression on general tasks
    - Validate domain terminology understanding
    """
    
    # JD Jones domain-specific test cases
    DOMAIN_TEST_CASES = [
        {
            "input": "What is API 622?",
            "expected_keywords": ["packing", "valve", "fugitive", "emissions", "testing"],
            "category": "standards"
        },
        {
            "input": "What temperature can PACMAAN gaskets withstand?",
            "expected_keywords": ["temperature", "°C", "°F", "high", "thermal"],
            "category": "products"
        },
        {
            "input": "What is the difference between gaskets and packings?",
            "expected_keywords": ["static", "dynamic", "seal", "rotating", "reciprocating"],
            "category": "technical"
        },
        {
            "input": "Which certifications are required for food industry seals?",
            "expected_keywords": ["FDA", "food", "safe", "certified", "compliance"],
            "category": "compliance"
        }
    ]
    
    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.7,
        keyword_match_threshold: float = 0.6
    ):
        """
        Initialize evaluator.
        
        Args:
            embedding_model: Model for semantic similarity
            similarity_threshold: Threshold for semantic match
            keyword_match_threshold: Threshold for keyword presence
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.keyword_match_threshold = keyword_match_threshold
    
    async def evaluate_model(
        self,
        model_id: str,
        test_dataset: FineTuningDataset,
        model_callable=None
    ) -> EvaluationReport:
        """
        Evaluate a fine-tuned model.
        
        Args:
            model_id: ID of the fine-tuned model
            test_dataset: Test dataset
            model_callable: Function to call model
            
        Returns:
            EvaluationReport
        """
        results = []
        
        for example in test_dataset.examples:
            # Get model output
            if model_callable:
                model_output = await model_callable(example.user_input)
            else:
                model_output = self._simulate_model_output(example.user_input)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                expected=example.assistant_output,
                actual=model_output
            )
            
            # Determine if passed
            passed = self._check_passed(metrics)
            
            results.append(EvaluationResult(
                example_id=example.id,
                input_text=example.user_input,
                expected_output=example.assistant_output,
                model_output=model_output,
                metrics=metrics,
                passed=passed
            ))
        
        # Aggregate metrics
        aggregate = self._aggregate_metrics(results)
        
        # Generate summary and recommendations
        summary, recommendations = self._generate_summary(results, aggregate)
        
        return EvaluationReport(
            model_id=model_id,
            dataset_name=test_dataset.name,
            timestamp=datetime.now(),
            results=results,
            aggregate_metrics=aggregate,
            summary=summary,
            recommendations=recommendations
        )
    
    async def evaluate_domain_understanding(
        self,
        model_id: str,
        model_callable=None
    ) -> Dict[str, Any]:
        """
        Evaluate model's understanding of domain-specific concepts.
        
        Args:
            model_id: Model to evaluate
            model_callable: Function to call model
            
        Returns:
            Domain evaluation results
        """
        results = []
        
        for test_case in self.DOMAIN_TEST_CASES:
            # Get model output
            if model_callable:
                output = await model_callable(test_case["input"])
            else:
                output = self._simulate_model_output(test_case["input"])
            
            # Check keyword presence
            output_lower = output.lower()
            found_keywords = [
                kw for kw in test_case["expected_keywords"]
                if kw.lower() in output_lower
            ]
            
            keyword_ratio = len(found_keywords) / len(test_case["expected_keywords"])
            passed = keyword_ratio >= self.keyword_match_threshold
            
            results.append({
                "input": test_case["input"],
                "category": test_case["category"],
                "expected_keywords": test_case["expected_keywords"],
                "found_keywords": found_keywords,
                "keyword_ratio": keyword_ratio,
                "passed": passed,
                "output_preview": output[:200] + "..." if len(output) > 200 else output
            })
        
        # Calculate overall score
        passed_count = sum(1 for r in results if r["passed"])
        
        return {
            "model_id": model_id,
            "total_tests": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results) if results else 0,
            "by_category": self._group_by_category(results),
            "details": results
        }
    
    async def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
        test_dataset: FineTuningDataset,
        model_a_callable=None,
        model_b_callable=None
    ) -> Dict[str, Any]:
        """
        Compare two models (e.g., base vs fine-tuned).
        
        Args:
            model_a_id: First model ID
            model_b_id: Second model ID
            test_dataset: Test dataset
            model_a_callable: Callable for model A
            model_b_callable: Callable for model B
            
        Returns:
            Comparison results
        """
        results_a = await self.evaluate_model(model_a_id, test_dataset, model_a_callable)
        results_b = await self.evaluate_model(model_b_id, test_dataset, model_b_callable)
        
        comparison = {
            "model_a": {
                "id": model_a_id,
                "metrics": results_a.aggregate_metrics,
                "pass_rate": sum(1 for r in results_a.results if r.passed) / len(results_a.results)
            },
            "model_b": {
                "id": model_b_id,
                "metrics": results_b.aggregate_metrics,
                "pass_rate": sum(1 for r in results_b.results if r.passed) / len(results_b.results)
            },
            "winner": None,
            "improvement": {}
        }
        
        # Determine winner
        if comparison["model_a"]["pass_rate"] > comparison["model_b"]["pass_rate"]:
            comparison["winner"] = model_a_id
        else:
            comparison["winner"] = model_b_id
        
        # Calculate improvement
        for metric in results_a.aggregate_metrics:
            if metric in results_b.aggregate_metrics:
                diff = results_b.aggregate_metrics[metric] - results_a.aggregate_metrics[metric]
                comparison["improvement"][metric] = {
                    "absolute": diff,
                    "relative": diff / results_a.aggregate_metrics[metric] if results_a.aggregate_metrics[metric] else 0
                }
        
        return comparison
    
    def _calculate_metrics(
        self,
        expected: str,
        actual: str
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        # Exact match
        metrics["exact_match"] = 1.0 if expected.strip() == actual.strip() else 0.0
        
        # Word overlap (simple BLEU approximation)
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if expected_words:
            overlap = len(expected_words & actual_words) / len(expected_words)
            metrics["word_overlap"] = overlap
        else:
            metrics["word_overlap"] = 0.0
        
        # Length ratio
        if len(expected) > 0:
            metrics["length_ratio"] = len(actual) / len(expected)
        else:
            metrics["length_ratio"] = 1.0
        
        # Semantic similarity (if embedding model available)
        if self.embedding_model:
            try:
                emb_expected = self.embedding_model.embed_query(expected)
                emb_actual = self.embedding_model.embed_query(actual)
                metrics["semantic_similarity"] = self._cosine_similarity(emb_expected, emb_actual)
            except Exception:
                metrics["semantic_similarity"] = metrics["word_overlap"]  # Fallback
        else:
            metrics["semantic_similarity"] = metrics["word_overlap"]
        
        return metrics
    
    def _check_passed(self, metrics: Dict[str, float]) -> bool:
        """Check if an evaluation passed."""
        # Primary check: semantic similarity
        if metrics.get("semantic_similarity", 0) >= self.similarity_threshold:
            return True
        
        # Secondary check: word overlap
        if metrics.get("word_overlap", 0) >= self.keyword_match_threshold:
            return True
        
        return False
    
    def _aggregate_metrics(
        self,
        results: List[EvaluationResult]
    ) -> Dict[str, float]:
        """Aggregate metrics across all results."""
        if not results:
            return {}
        
        aggregated = {}
        metric_keys = results[0].metrics.keys()
        
        for key in metric_keys:
            values = [r.metrics.get(key, 0) for r in results]
            aggregated[f"avg_{key}"] = sum(values) / len(values)
            aggregated[f"min_{key}"] = min(values)
            aggregated[f"max_{key}"] = max(values)
        
        # Pass rate
        aggregated["pass_rate"] = sum(1 for r in results if r.passed) / len(results)
        
        return aggregated
    
    def _generate_summary(
        self,
        results: List[EvaluationResult],
        aggregate: Dict[str, float]
    ) -> Tuple[str, List[str]]:
        """Generate summary and recommendations."""
        pass_rate = aggregate.get("pass_rate", 0)
        avg_similarity = aggregate.get("avg_semantic_similarity", 0)
        
        # Summary
        if pass_rate >= 0.9:
            summary = f"Excellent performance. Pass rate: {pass_rate:.1%}. Model is ready for deployment."
        elif pass_rate >= 0.7:
            summary = f"Good performance. Pass rate: {pass_rate:.1%}. Minor improvements recommended."
        elif pass_rate >= 0.5:
            summary = f"Moderate performance. Pass rate: {pass_rate:.1%}. Consider additional training data."
        else:
            summary = f"Poor performance. Pass rate: {pass_rate:.1%}. Model needs significant improvement."
        
        # Recommendations
        recommendations = []
        
        if pass_rate < 0.7:
            recommendations.append("Increase training data quantity (aim for at least 500 examples)")
        
        if avg_similarity < 0.7:
            recommendations.append("Add more diverse examples covering edge cases")
        
        if aggregate.get("avg_length_ratio", 1) < 0.5:
            recommendations.append("Model responses are too short - add examples with longer responses")
        elif aggregate.get("avg_length_ratio", 1) > 2:
            recommendations.append("Model responses are too verbose - add concise examples")
        
        if not recommendations:
            recommendations.append("Model performs well. Monitor production performance.")
        
        return summary, recommendations
    
    def _group_by_category(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group results by category."""
        categories = {}
        
        for result in results:
            cat = result.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            
            if result["passed"]:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        return categories
    
    def _simulate_model_output(self, input_text: str) -> str:
        """Simulate model output for testing."""
        # Simple simulation based on input keywords
        input_lower = input_text.lower()
        
        if "api 622" in input_lower:
            return "API 622 is a standard that establishes testing requirements for packing materials used in rising stem valves. It focuses on fugitive emissions testing procedures."
        elif "pacmaan" in input_lower:
            return "PACMAAN gaskets are designed for high-temperature applications, withstanding temperatures up to 450°C (842°F) depending on the specific variant."
        elif "gasket" in input_lower and "packing" in input_lower:
            return "Gaskets are static seals used between non-moving surfaces, while packings are dynamic seals used around rotating or reciprocating shafts."
        elif "fda" in input_lower or "food" in input_lower:
            return "For food industry applications, seals must be FDA certified and compliant with food safety regulations."
        
        return "I can help with information about our sealing products and industry standards."
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
