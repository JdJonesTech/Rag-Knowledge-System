"""
Evaluation Hooks
Automated evaluation of agent responses for hallucination detection,
factual grounding, response relevance, and domain accuracy scoring.
Integrates with LangFuse for persistent scoring.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationHookType(str, Enum):
    """Types of evaluation hooks."""
    HALLUCINATION_CHECK = "hallucination_check"
    FACTUAL_GROUNDING = "factual_grounding"
    RESPONSE_RELEVANCE = "response_relevance"
    DOMAIN_ACCURACY = "domain_accuracy"
    SAFETY_CHECK = "safety_check"


@dataclass
class EvaluationScore:
    """A single evaluation score."""
    hook_type: EvaluationHookType
    score: float  # 0.0 to 1.0
    passed: bool
    details: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hook_type": self.hook_type.value,
            "score": self.score,
            "passed": self.passed,
            "details": self.details,
            "metadata": self.metadata,
        }


class EvaluationHooks:
    """
    Automated evaluation hooks for agent responses.
    Designed to integrate with LangFuse scoring API.
    """

    # JD Jones domain terms that must be used correctly
    CRITICAL_DOMAIN_TERMS = {
        "api 622": "packing testing standard valve",
        "api 624": "valve fugitive emission testing",
        "pacmaan": "jd jones gasket product line",
        "flexseal": "jd jones packing product",
        "expansoflex": "jd jones expansion joint",
        "fugitive emissions": "unintentional gas vapor release leakage",
        "spiral wound": "gasket type metallic winding",
        "ptfe": "polytetrafluoroethylene sealing material",
    }

    def __init__(self, langfuse_exporter=None):
        self.langfuse_exporter = langfuse_exporter
        self._hooks = {
            EvaluationHookType.HALLUCINATION_CHECK: self._check_hallucination,
            EvaluationHookType.FACTUAL_GROUNDING: self._check_factual_grounding,
            EvaluationHookType.RESPONSE_RELEVANCE: self._check_relevance,
            EvaluationHookType.DOMAIN_ACCURACY: self._check_domain_accuracy,
        }

    async def evaluate_response(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        trace_id: Optional[str] = None,
    ) -> List[EvaluationScore]:
        """
        Run all evaluation hooks on a response.

        Args:
            query: The user query
            response: The agent's response
            sources: Source documents used
            trace_id: Optional trace ID for LangFuse scoring

        Returns:
            List of evaluation scores
        """
        scores = []
        for hook_type, hook_fn in self._hooks.items():
            try:
                score = await hook_fn(query, response, sources)
                scores.append(score)

                if self.langfuse_exporter and trace_id:
                    self.langfuse_exporter.log_score(
                        trace_id=trace_id,
                        name=hook_type.value,
                        value=score.score,
                        comment=score.details,
                    )
            except Exception as e:
                logger.error(f"Evaluation hook {hook_type.value} failed: {e}")
                scores.append(EvaluationScore(
                    hook_type=hook_type,
                    score=0.0,
                    passed=False,
                    details=f"Hook error: {str(e)}",
                ))
        return scores

    async def _check_hallucination(
        self, query: str, response: str, sources: List[Dict[str, Any]]
    ) -> EvaluationScore:
        """Check if response contains claims not supported by sources."""
        if not sources:
            has_factual_claims = any(
                indicator in response.lower()
                for indicator in [
                    "degrees", "bar", "psi", "certified", "rated for",
                    "withstand", "temperature", "pressure", "standard"
                ]
            )
            return EvaluationScore(
                hook_type=EvaluationHookType.HALLUCINATION_CHECK,
                score=0.3 if has_factual_claims else 0.7,
                passed=not has_factual_claims,
                details=(
                    "Factual claims without source citations"
                    if has_factual_claims
                    else "No factual claims detected"
                ),
            )

        source_text = " ".join(
            str(s.get("content", s.get("text", ""))) for s in sources
        ).lower()

        response_words = set(response.lower().split())
        source_words = set(source_text.split())
        overlap = len(response_words & source_words) / max(len(response_words), 1)

        score = min(1.0, overlap * 2)
        return EvaluationScore(
            hook_type=EvaluationHookType.HALLUCINATION_CHECK,
            score=score,
            passed=score >= 0.4,
            details=f"Source overlap ratio: {overlap:.2f}",
        )

    async def _check_factual_grounding(
        self, query: str, response: str, sources: List[Dict[str, Any]]
    ) -> EvaluationScore:
        """Check if factual claims in response are grounded in sources."""
        if not sources:
            return EvaluationScore(
                hook_type=EvaluationHookType.FACTUAL_GROUNDING,
                score=0.5,
                passed=True,
                details="No sources to verify against",
            )

        source_refs = sum(
            1 for s in sources
            if any(
                word in response.lower()
                for word in str(s.get("title", s.get("source", ""))).lower().split()
                if len(word) > 3
            )
        )
        score = min(1.0, source_refs / max(len(sources), 1))
        return EvaluationScore(
            hook_type=EvaluationHookType.FACTUAL_GROUNDING,
            score=score,
            passed=score >= 0.3,
            details=f"Referenced {source_refs}/{len(sources)} sources",
        )

    async def _check_relevance(
        self, query: str, response: str, sources: List[Dict[str, Any]]
    ) -> EvaluationScore:
        """Check if the response is relevant to the query."""
        query_words = set(w for w in query.lower().split() if len(w) > 2)
        response_words = set(response.lower().split())

        if not query_words:
            return EvaluationScore(
                hook_type=EvaluationHookType.RESPONSE_RELEVANCE,
                score=0.5,
                passed=True,
                details="Query too short to assess relevance",
            )

        overlap = len(query_words & response_words) / len(query_words)
        score = min(1.0, overlap * 2)
        return EvaluationScore(
            hook_type=EvaluationHookType.RESPONSE_RELEVANCE,
            score=score,
            passed=score >= 0.3,
            details=f"Query-response keyword overlap: {overlap:.2f}",
        )

    async def _check_domain_accuracy(
        self, query: str, response: str, sources: List[Dict[str, Any]]
    ) -> EvaluationScore:
        """Check if domain terms are used correctly."""
        response_lower = response.lower()
        terms_found = 0
        terms_correct = 0

        for term, expected_context in self.CRITICAL_DOMAIN_TERMS.items():
            if term in response_lower:
                terms_found += 1
                if any(w in response_lower for w in expected_context.split()):
                    terms_correct += 1

        if terms_found == 0:
            return EvaluationScore(
                hook_type=EvaluationHookType.DOMAIN_ACCURACY,
                score=1.0,
                passed=True,
                details="No domain terms to verify",
            )

        score = terms_correct / terms_found
        return EvaluationScore(
            hook_type=EvaluationHookType.DOMAIN_ACCURACY,
            score=score,
            passed=score >= 0.5,
            details=f"Domain terms correct: {terms_correct}/{terms_found}",
        )
