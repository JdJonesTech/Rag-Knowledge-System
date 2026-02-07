"""
SLM-Enhanced Specialized Agents
Specialized agents that use locally-trained SLMs for fast, domain-specific processing.

Architecture:
- LLM (GPT-4/Claude) = Main Brain for complex reasoning and orchestration
- SLMs = Fast local workers for classification, extraction, and pre-filtering
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from src.agentic.agents.react_agent import ReActAgent, ReActResult
from src.agentic.slm.training import SLMInference, SLMType, get_slm_inference
from src.agentic.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class SLMEnhancedResult:
    """Result from an SLM-enhanced agent."""
    success: bool
    final_answer: str
    slm_predictions: Dict[str, Any] = field(default_factory=dict)
    llm_used: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    processing_time_ms: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "slm_predictions": self.slm_predictions,
            "llm_used": self.llm_used,
            "trace": self.trace,
            "sources": self.sources,
            "processing_time_ms": self.processing_time_ms
        }


class SLMEnhancedTechnicalAgent:
    """
    Technical specifications agent enhanced with SLMs.
    
    Uses:
    - SLM for intent classification (fast)
    - SLM for entity extraction (product codes, specs)
    - LLM for complex reasoning (when needed)
    
    Flow:
    1. SLM classifies intent (< 10ms)
    2. SLM extracts entities (< 10ms)
    3. If simple lookup: use indexed data directly
    4. If complex: escalate to LLM for reasoning
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, BaseTool]] = None,
        llm_agent: Optional[ReActAgent] = None,
        slm_inference: Optional[SLMInference] = None
    ):
        self.tools = tools or {}
        self.llm_agent = llm_agent
        self.slm = slm_inference or get_slm_inference()
        self.agent_name = "SLM_TechnicalAgent"
        
        # Quick lookup cache for common queries
        self._spec_cache: Dict[str, Dict[str, Any]] = {}
    
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SLMEnhancedResult:
        """
        Execute query using SLM-first approach.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            SLMEnhancedResult with answer and processing info
        """
        import time
        start_time = time.time()
        
        trace = []
        slm_predictions = {}
        
        # Step 1: SLM Intent Classification
        trace.append({
            "step": "slm_intent_classification",
            "agent": self.agent_name,
            "description": "Using SLM to classify query intent"
        })
        
        intent_result = self.slm.predict(SLMType.INTENT_CLASSIFIER, query)
        slm_predictions["intent"] = intent_result
        
        # Step 2: SLM Entity Extraction
        trace.append({
            "step": "slm_entity_extraction",
            "agent": self.agent_name,
            "description": "Using SLM to extract product codes and specifications"
        })
        
        # Simple regex-based entity extraction for product codes
        import re
        product_codes = re.findall(r'NA\s*\d+|NJ\s*\d+', query, re.IGNORECASE)
        temperatures = re.findall(r'(\d+)\s*°?[CF]', query)
        pressures = re.findall(r'(\d+)\s*(psi|bar|mpa)', query, re.IGNORECASE)
        
        entities = {
            "product_codes": product_codes,
            "temperatures": temperatures,
            "pressures": pressures
        }
        slm_predictions["entities"] = entities
        
        # Step 3: Determine if simple lookup suffices
        is_simple = (
            intent_result.get("confidence", 0) > 0.8 and
            product_codes and
            intent_result.get("prediction") in ["product_inquiry", "technical_question"]
        )
        
        if is_simple and product_codes:
            # Try direct specification lookup
            trace.append({
                "step": "direct_lookup",
                "agent": self.agent_name,
                "description": f"Performing direct lookup for: {product_codes}"
            })
            
            spec_data = await self._lookup_specifications(product_codes)
            
            if spec_data:
                # Format response without LLM
                answer = self._format_spec_response(product_codes, spec_data, query)
                
                return SLMEnhancedResult(
                    success=True,
                    final_answer=answer,
                    slm_predictions=slm_predictions,
                    llm_used=False,
                    trace=trace,
                    sources=[{"type": "specifications", "products": product_codes}],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        
        # Step 4: Escalate to LLM for complex reasoning
        trace.append({
            "step": "llm_escalation",
            "agent": self.agent_name,
            "description": "Query requires LLM reasoning - escalating to main brain"
        })
        
        if self.llm_agent:
            # Enhance context with SLM predictions
            enhanced_context = {
                **(context or {}),
                "slm_intent": intent_result.get("prediction"),
                "slm_entities": entities,
                "slm_confidence": intent_result.get("confidence", 0)
            }
            
            llm_result = await self.llm_agent.execute(query, enhanced_context)
            
            for step in llm_result.trace:
                trace.append({
                    "step": f"llm_{step.step_type.value}",
                    "agent": "LLM_ReActAgent",
                    "content": step.content[:200] if step.content else None
                })
            
            return SLMEnhancedResult(
                success=llm_result.success,
                final_answer=llm_result.final_answer,
                slm_predictions=slm_predictions,
                llm_used=True,
                trace=trace,
                sources=llm_result.sources,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        return SLMEnhancedResult(
            success=False,
            final_answer="Unable to process query - LLM not available",
            slm_predictions=slm_predictions,
            llm_used=False,
            trace=trace,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _lookup_specifications(
        self,
        product_codes: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Look up specifications for product codes."""
        # Check cache first
        cached = {}
        for code in product_codes:
            if code.upper() in self._spec_cache:
                cached[code] = self._spec_cache[code.upper()]
        
        if len(cached) == len(product_codes):
            return cached
        
        # Use vector search tool if available
        if "vector_search" in self.tools:
            try:
                tool = self.tools["vector_search"]
                query = f"specifications for {' '.join(product_codes)}"
                result = await tool.execute(query, {"product_codes": product_codes})
                if result.success:
                    return result.data
            except Exception as e:
                logger.error(f"Error in spec lookup: {e}")
        
        return None
    
    def _format_spec_response(
        self,
        product_codes: List[str],
        spec_data: Dict[str, Any],
        original_query: str
    ) -> str:
        """Format specification data into a readable response."""
        response_parts = []
        
        for code in product_codes:
            code_upper = code.upper()
            if code_upper in spec_data:
                specs = spec_data[code_upper]
                response_parts.append(f"**{code_upper}**:")
                for key, value in specs.items():
                    response_parts.append(f"  - {key}: {value}")
        
        if response_parts:
            return "\n".join(response_parts)
        
        return f"Specifications for {', '.join(product_codes)} not found in quick lookup. Please refine your query."


class SLMEnhancedComplianceAgent:
    """
    Compliance checking agent enhanced with SLMs.
    
    Uses:
    - SLM for standard identification (API 622, FDA, etc.)
    - SLM for compliance status classification
    - LLM for nuanced compliance interpretation
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, BaseTool]] = None,
        llm_agent: Optional[ReActAgent] = None,
        slm_inference: Optional[SLMInference] = None
    ):
        self.tools = tools or {}
        self.llm_agent = llm_agent
        self.slm = slm_inference or get_slm_inference()
        self.agent_name = "SLM_ComplianceAgent"
        
        # Known standards database
        self.standards_db = {
            "API 622": {
                "name": "API 622",
                "full_name": "Type Testing of Process Valve Packing",
                "categories": ["valve packing", "fugitive emissions"],
                "requirements": ["fugitive emissions < 100 ppm", "10,000+ cycles"]
            },
            "API 624": {
                "name": "API 624",
                "full_name": "Type Testing of Rising Stem Valves",
                "categories": ["rising stem valves", "fugitive emissions"],
                "requirements": ["thermal cycling", "310°C test temperature"]
            },
            "FDA 21 CFR 177": {
                "name": "FDA 21 CFR 177",
                "full_name": "Food Contact Materials",
                "categories": ["food contact", "pharmaceutical"],
                "requirements": ["non-toxic", "clean room compatible"]
            }
        }
    
    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SLMEnhancedResult:
        """Execute compliance check using SLM-first approach."""
        import time
        start_time = time.time()
        
        trace = []
        slm_predictions = {}
        
        # Step 1: Extract standards mentioned
        import re
        standards_found = []
        
        for standard_key in self.standards_db.keys():
            if standard_key.lower() in query.lower():
                standards_found.append(standard_key)
        
        # Also check for common patterns
        api_matches = re.findall(r'API\s*\d+', query, re.IGNORECASE)
        standards_found.extend(api_matches)
        
        slm_predictions["standards_detected"] = list(set(standards_found))
        
        trace.append({
            "step": "standard_extraction",
            "agent": self.agent_name,
            "standards_found": standards_found
        })
        
        # Step 2: SLM compliance classification
        compliance_result = self.slm.predict(SLMType.COMPLIANCE_CHECKER, query)
        slm_predictions["compliance_check"] = compliance_result
        
        # Step 3: If specific standard + product mentioned, do quick lookup
        product_codes = re.findall(r'NA\s*\d+|NJ\s*\d+', query, re.IGNORECASE)
        
        if standards_found and product_codes:
            trace.append({
                "step": "quick_compliance_lookup",
                "agent": self.agent_name,
                "products": product_codes,
                "standards": standards_found
            })
            
            # Try quick lookup
            quick_answer = self._quick_compliance_check(product_codes, standards_found)
            
            if quick_answer:
                return SLMEnhancedResult(
                    success=True,
                    final_answer=quick_answer,
                    slm_predictions=slm_predictions,
                    llm_used=False,
                    trace=trace,
                    sources=[{"type": "compliance_database"}],
                    processing_time_ms=(time.time() - start_time) * 1000
                )
        
        # Step 4: Escalate to LLM for complex compliance questions
        trace.append({
            "step": "llm_escalation",
            "agent": self.agent_name,
            "reason": "Complex compliance question requires LLM reasoning"
        })
        
        if self.llm_agent:
            enhanced_context = {
                **(context or {}),
                "standards_detected": standards_found,
                "slm_compliance": compliance_result
            }
            
            llm_result = await self.llm_agent.execute(query, enhanced_context)
            
            return SLMEnhancedResult(
                success=llm_result.success,
                final_answer=llm_result.final_answer,
                slm_predictions=slm_predictions,
                llm_used=True,
                trace=trace,
                sources=llm_result.sources,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        return SLMEnhancedResult(
            success=False,
            final_answer="Complex compliance check requires LLM - not available",
            slm_predictions=slm_predictions,
            llm_used=False,
            trace=trace,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _quick_compliance_check(
        self,
        product_codes: List[str],
        standards: List[str]
    ) -> Optional[str]:
        """Quick compliance check from database."""
        # This would query a compliance database in production
        # For now, return None to escalate to LLM
        return None


class SLMEnhancedQueryRouter:
    """
    Query router enhanced with SLMs.
    
    Uses SLMs to:
    1. Classify query intent (< 10ms)
    2. Extract key entities (< 10ms)
    3. Route to appropriate specialist
    
    Only escalates to LLM for ambiguous queries.
    """
    
    # Routing rules based on SLM predictions
    ROUTING_RULES = {
        "product_inquiry": "technical_specifications",
        "technical_question": "technical_specifications",
        "pricing_request": "pricing_quotes",
        "compliance_check": "compliance_and_standards",
        "troubleshooting": "troubleshooting",
        "order_status": "order_support",
        "complaint": "escalate_to_human",
        "general": "general_agent"
    }
    
    def __init__(self, slm_inference: Optional[SLMInference] = None):
        self.slm = slm_inference or get_slm_inference()
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Route query to appropriate specialist.
        
        Args:
            query: User query
            
        Returns:
            Routing decision with confidence
        """
        # SLM-based intent classification
        intent_result = self.slm.predict(SLMType.INTENT_CLASSIFIER, query)
        
        predicted_intent = intent_result.get("prediction", "general")
        confidence = intent_result.get("confidence", 0)
        
        # Get routing destination
        destination = self.ROUTING_RULES.get(predicted_intent, "general_agent")
        
        # If low confidence, route to general agent (which uses LLM)
        if confidence < 0.6:
            destination = "general_agent"
            needs_llm = True
        else:
            needs_llm = predicted_intent == "general"
        
        return {
            "destination": destination,
            "intent": predicted_intent,
            "confidence": confidence,
            "needs_llm": needs_llm,
            "all_scores": intent_result.get("scores", {})
        }


# Factory functions
def create_slm_enhanced_technical_agent(
    tools: Optional[Dict[str, BaseTool]] = None
) -> SLMEnhancedTechnicalAgent:
    """Create an SLM-enhanced technical agent."""
    from src.agentic.agents.specialized_agents import TechnicalSpecsAgent
    
    llm_agent = TechnicalSpecsAgent(tools=tools)
    return SLMEnhancedTechnicalAgent(
        tools=tools,
        llm_agent=llm_agent
    )


def create_slm_enhanced_compliance_agent(
    tools: Optional[Dict[str, BaseTool]] = None
) -> SLMEnhancedComplianceAgent:
    """Create an SLM-enhanced compliance agent."""
    from src.agentic.agents.specialized_agents import ComplianceAgent
    
    llm_agent = ComplianceAgent(tools=tools)
    return SLMEnhancedComplianceAgent(
        tools=tools,
        llm_agent=llm_agent
    )
