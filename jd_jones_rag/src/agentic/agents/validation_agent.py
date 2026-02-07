"""
Validation Agent
Cross-references retrieved facts against trusted internal sources.
Minimizes hallucinations and ensures compliance.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings


class ValidationLevel(str, Enum):
    """Validation confidence levels."""
    VERIFIED = "verified"           # Confirmed against trusted source
    LIKELY_CORRECT = "likely_correct"  # High confidence but not verified
    UNCERTAIN = "uncertain"         # Cannot verify
    CONTRADICTED = "contradicted"   # Conflicts with trusted source
    OUTDATED = "outdated"          # Source may be out of date


class FactType(str, Enum):
    """Types of facts to validate."""
    PRODUCT_SPEC = "product_specification"
    PRICING = "pricing"
    AVAILABILITY = "availability"
    CERTIFICATION = "certification"
    TECHNICAL_CLAIM = "technical_claim"
    POLICY = "policy"
    CONTACT = "contact_information"


@dataclass
class ValidationResult:
    """Result of validating a single fact."""
    fact: str
    fact_type: FactType
    validation_level: ValidationLevel
    confidence: float
    supporting_sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    notes: str = ""
    corrected_value: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact,
            "fact_type": self.fact_type.value,
            "validation_level": self.validation_level.value,
            "confidence": self.confidence,
            "supporting_sources": self.supporting_sources,
            "contradicting_sources": self.contradicting_sources,
            "notes": self.notes,
            "corrected_value": self.corrected_value
        }


@dataclass
class ValidationReport:
    """Complete validation report for a response."""
    report_id: str
    original_response: str
    facts_extracted: int
    facts_verified: int
    facts_uncertain: int
    facts_contradicted: int
    overall_reliability: float
    validation_results: List[ValidationResult]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "original_response": self.original_response,
            "facts_extracted": self.facts_extracted,
            "facts_verified": self.facts_verified,
            "facts_uncertain": self.facts_uncertain,
            "facts_contradicted": self.facts_contradicted,
            "overall_reliability": self.overall_reliability,
            "validation_results": [v.to_dict() for v in self.validation_results],
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat()
        }


class ValidationAgent:
    """
    Validates AI-generated responses against trusted sources.
    
    Capabilities:
    - Extract factual claims from responses
    - Cross-reference against knowledge base
    - Identify contradictions or outdated info
    - Provide confidence scores
    - Suggest corrections
    """
    
    # Trusted source priority (higher = more trusted)
    SOURCE_TRUST_LEVELS = {
        "official_datasheet": 1.0,
        "product_database": 0.95,
        "erp_system": 0.9,
        "internal_documentation": 0.85,
        "knowledge_base": 0.8,
        "training_data": 0.5
    }
    
    FACT_EXTRACTION_PROMPT = """Extract all factual claims from this AI-generated response.
Focus on:
- Product specifications (temperature, pressure, dimensions)
- Pricing and availability
- Certifications and standards
- Technical capabilities
- Company policies
- Contact information

Response to analyze:
{response}

Return as JSON array:
[
    {{
        "fact": "The specific claim",
        "fact_type": "product_specification|pricing|availability|certification|technical_claim|policy|contact_information",
        "key_entities": ["entity1", "entity2"],
        "requires_verification": true/false
    }}
]
"""

    VALIDATION_PROMPT = """Validate this fact against the provided trusted sources.

FACT TO VALIDATE:
{fact}

FACT TYPE: {fact_type}

TRUSTED SOURCES:
{sources}

Respond in JSON:
{{
    "validation_level": "verified|likely_correct|uncertain|contradicted|outdated",
    "confidence": 0.0-1.0,
    "supporting_evidence": ["evidence1", "evidence2"],
    "contradicting_evidence": ["if any"],
    "notes": "explanation",
    "corrected_value": "if fact is incorrect, provide correct value"
}}
"""

    def __init__(self, retriever=None):
        """
        Initialize validation agent.
        
        Args:
            retriever: Knowledge base retriever for fact-checking
        """
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0)
        self.retriever = retriever
    
    async def validate_response(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """
        Validate an AI-generated response.
        
        Args:
            response: The response to validate
            context: Additional context
            
        Returns:
            ValidationReport with results
        """
        import uuid
        
        # Step 1: Extract facts from response
        facts = await self._extract_facts(response)
        
        # Step 2: Validate each fact
        validation_results = []
        for fact_data in facts:
            if fact_data.get("requires_verification", True):
                result = await self._validate_fact(
                    fact=fact_data["fact"],
                    fact_type=FactType(fact_data["fact_type"]),
                    entities=fact_data.get("key_entities", [])
                )
                validation_results.append(result)
        
        # Step 3: Calculate overall reliability
        verified = sum(1 for v in validation_results if v.validation_level == ValidationLevel.VERIFIED)
        likely = sum(1 for v in validation_results if v.validation_level == ValidationLevel.LIKELY_CORRECT)
        uncertain = sum(1 for v in validation_results if v.validation_level == ValidationLevel.UNCERTAIN)
        contradicted = sum(1 for v in validation_results if v.validation_level == ValidationLevel.CONTRADICTED)
        
        total = len(validation_results) or 1
        reliability = (verified * 1.0 + likely * 0.8 + uncertain * 0.5 + contradicted * 0.0) / total
        
        # Step 4: Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        
        return ValidationReport(
            report_id=f"val_{uuid.uuid4().hex[:8]}",
            original_response=response[:500] + "..." if len(response) > 500 else response,
            facts_extracted=len(facts),
            facts_verified=verified,
            facts_uncertain=uncertain,
            facts_contradicted=contradicted,
            overall_reliability=reliability,
            validation_results=validation_results,
            recommendations=recommendations
        )
    
    async def _extract_facts(self, response: str) -> List[Dict[str, Any]]:
        """Extract factual claims from response."""
        prompt = self.FACT_EXTRACTION_PROMPT.format(response=response)
        
        messages = [HumanMessage(content=prompt)]
        llm_response = await self.llm.ainvoke(messages)
        
        content = llm_response.content
        if "[" in content:
            start = content.index("[")
            end = content.rindex("]") + 1
            return json.loads(content[start:end])
        
        return []
    
    async def _validate_fact(
        self,
        fact: str,
        fact_type: FactType,
        entities: List[str]
    ) -> ValidationResult:
        """Validate a single fact against trusted sources."""
        # Retrieve relevant documents
        sources_text = "No additional sources available."
        supporting = []
        
        if self.retriever:
            # Search for relevant documents
            search_query = f"{fact} {' '.join(entities)}"
            try:
                from src.knowledge_base.retriever import UserRole
                results = self.retriever.retrieve(
                    query=search_query,
                    user_role=UserRole.ADMIN,  # Full access for validation
                    n_results=5
                )
                
                if results.all_results:
                    sources_text = "\n".join([
                        f"- [{r.source}]: {r.content[:200]}"
                        for r in results.all_results
                    ])
                    supporting = [r.source for r in results.all_results]
            except Exception as e:
                logger.warning(f"Failed to retrieve sources for fact validation: {e}")
        
        # Use LLM to validate
        prompt = self.VALIDATION_PROMPT.format(
            fact=fact,
            fact_type=fact_type.value,
            sources=sources_text
        )
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        content = response.content
        if "{" in content:
            start = content.index("{")
            end = content.rindex("}") + 1
            result_data = json.loads(content[start:end])
            
            return ValidationResult(
                fact=fact,
                fact_type=fact_type,
                validation_level=ValidationLevel(result_data.get("validation_level", "uncertain")),
                confidence=result_data.get("confidence", 0.5),
                supporting_sources=result_data.get("supporting_evidence", supporting),
                contradicting_sources=result_data.get("contradicting_evidence", []),
                notes=result_data.get("notes", ""),
                corrected_value=result_data.get("corrected_value")
            )
        
        return ValidationResult(
            fact=fact,
            fact_type=fact_type,
            validation_level=ValidationLevel.UNCERTAIN,
            confidence=0.5,
            notes="Unable to parse validation result"
        )
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        contradicted = [r for r in results if r.validation_level == ValidationLevel.CONTRADICTED]
        uncertain = [r for r in results if r.validation_level == ValidationLevel.UNCERTAIN]
        outdated = [r for r in results if r.validation_level == ValidationLevel.OUTDATED]
        
        if contradicted:
            recommendations.append(
                f"CRITICAL: {len(contradicted)} fact(s) contradict trusted sources. "
                "Review and correct before sharing with customer."
            )
            for r in contradicted[:2]:
                if r.corrected_value:
                    recommendations.append(f"  - '{r.fact[:50]}...' should be '{r.corrected_value}'")
        
        if outdated:
            recommendations.append(
                f"WARNING: {len(outdated)} fact(s) may be outdated. Verify current information."
            )
        
        if uncertain:
            recommendations.append(
                f"NOTE: {len(uncertain)} fact(s) could not be verified. "
                "Consider adding source citations."
            )
        
        if not recommendations:
            recommendations.append("All facts verified against trusted sources.")
        
        return recommendations
    
    async def quick_check(
        self,
        claim: str,
        claim_type: str = "technical_claim"
    ) -> Dict[str, Any]:
        """
        Quick validation of a single claim.
        
        Args:
            claim: The claim to check
            claim_type: Type of claim
            
        Returns:
            Quick validation result
        """
        result = await self._validate_fact(
            fact=claim,
            fact_type=FactType(claim_type),
            entities=[]
        )
        
        return {
            "claim": claim,
            "is_valid": result.validation_level in [ValidationLevel.VERIFIED, ValidationLevel.LIKELY_CORRECT],
            "confidence": result.confidence,
            "level": result.validation_level.value,
            "notes": result.notes
        }
