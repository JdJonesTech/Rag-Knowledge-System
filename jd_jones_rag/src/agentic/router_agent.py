"""
Router Agent
Analyzes queries, determines intent, identifies missing parameters,
and decides which tools/agents to invoke.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings


class QueryIntent(str, Enum):
    """Intent categories for queries."""
    PRODUCT_SELECTION = "product_selection"
    TECHNICAL_SUPPORT = "technical_support"
    PRICING_QUOTE = "pricing_quote"
    ORDER_STATUS = "order_status"
    DOCUMENTATION = "documentation"
    SPECIFICATION_LOOKUP = "specification_lookup"
    COMPLIANCE_CHECK = "compliance_check"
    STOCK_AVAILABILITY = "stock_availability"
    GENERAL_INQUIRY = "general_inquiry"
    ENQUIRY_ROUTING = "enquiry_routing"


class QueryComplexity(str, Enum):
    """Complexity level of the query."""
    SIMPLE = "simple"          # Direct lookup, single source
    MODERATE = "moderate"      # Multiple sources or some reasoning
    COMPLEX = "complex"        # Multi-step reasoning, validation needed
    EXPERT = "expert"          # Requires human expert involvement


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    confidence: float
    
    # Parameter extraction
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    missing_parameters: List[str] = field(default_factory=list)
    suggested_values: Dict[str, List[str]] = field(default_factory=dict)
    
    # Tool routing
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    
    # Follow-up
    follow_up_questions: List[str] = field(default_factory=list)
    requires_clarification: bool = False
    
    # Metadata
    entities: Dict[str, List[str]] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "extracted_parameters": self.extracted_parameters,
            "missing_parameters": self.missing_parameters,
            "suggested_values": self.suggested_values,
            "required_tools": self.required_tools,
            "optional_tools": self.optional_tools,
            "follow_up_questions": self.follow_up_questions,
            "requires_clarification": self.requires_clarification,
            "entities": self.entities,
            "keywords": self.keywords
        }


class RouterAgent:
    """
    Analyzes queries to determine intent and routing.
    Uses an LLM for sophisticated understanding with rule-based fallbacks.
    """
    
    # Critical parameters for product selection
    PRODUCT_SELECTION_PARAMS = {
        "required": ["industry", "application_type"],
        "recommended": ["temperature", "pressure", "media", "equipment_type"],
        "optional": ["size", "certification", "ph_level", "chemical_resistance"]
    }
    
    # Parameter suggestions based on industry
    INDUSTRY_SUGGESTIONS = {
        "oil_gas": {
            "certifications": ["API 622", "API 624", "API 6A", "API 6D"],
            "media": ["crude oil", "natural gas", "H2S", "CO2", "drilling mud"],
            "equipment": ["valve", "pump", "compressor", "flange", "heat exchanger"]
        },
        "chemical": {
            "certifications": ["FDA", "USP Class VI", "3A Sanitary"],
            "media": ["acids", "alkalis", "solvents", "corrosive fluids"],
            "equipment": ["reactor", "tank", "pipe", "valve", "agitator"]
        },
        "pharmaceutical": {
            "certifications": ["FDA", "USP Class VI", "cGMP", "ISO 13485"],
            "media": ["WFI", "CIP/SIP", "sterile fluids", "active ingredients"],
            "equipment": ["bioreactor", "filling line", "clean room equipment"]
        },
        "power": {
            "certifications": ["ASME", "IEEE", "NEMA"],
            "media": ["steam", "water", "fuel oil", "flue gas"],
            "equipment": ["turbine", "boiler", "condenser", "pump"]
        }
    }
    
    # Intent to tool mapping
    INTENT_TOOLS = {
        QueryIntent.PRODUCT_SELECTION: ["vector_search", "product_database", "compliance_checker"],
        QueryIntent.TECHNICAL_SUPPORT: ["vector_search", "knowledge_base"],
        QueryIntent.PRICING_QUOTE: ["product_database", "pricing_engine", "erp_query"],
        QueryIntent.ORDER_STATUS: ["erp_query", "crm_lookup"],
        QueryIntent.DOCUMENTATION: ["document_generator", "vector_search", "compliance_checker"],
        QueryIntent.SPECIFICATION_LOOKUP: ["vector_search", "product_database"],
        QueryIntent.COMPLIANCE_CHECK: ["compliance_checker", "vector_search"],
        QueryIntent.STOCK_AVAILABILITY: ["erp_query", "inventory_check"],
        QueryIntent.ENQUIRY_ROUTING: ["classifier", "crm_update", "email_router"],
        QueryIntent.GENERAL_INQUIRY: ["vector_search", "knowledge_base"]
    }
    
    ANALYSIS_PROMPT = """You are a query analyzer for JD Jones Manufacturing, a company that makes industrial sealing solutions.

Analyze the following query and extract:
1. Intent (product_selection, technical_support, pricing_quote, order_status, documentation, specification_lookup, compliance_check, stock_availability, general_inquiry, enquiry_routing)
2. Complexity (simple, moderate, complex, expert)
3. Any parameters mentioned (temperature, pressure, media, industry, equipment, size, certifications)
4. Missing critical parameters for the intent
5. Named entities (product names, company names, standards mentioned)

IMPORTANT PARAMETER DETECTION:
- Temperature: Look for degrees (°C, °F, K), ranges like "high temperature", "-40 to 200°C"
- Pressure: Look for bar, psi, MPa, "high pressure", "450 bar"
- Media: Look for fluid names, chemicals, "crude oil", "steam", "acid"
- Industry: Look for "oil and gas", "chemical", "pharmaceutical", "food", "power"
- Standards: Look for API 622, API 624, Shell SPE, FDA, ASME, etc.
- Equipment: valve, pump, flange, gasket, expansion joint, etc.

For PRODUCT_SELECTION queries, these parameters are critical:
- Required: industry, application_type
- Recommended: temperature, pressure, media, equipment_type

Respond in JSON format:
{
    "intent": "string",
    "complexity": "string",
    "confidence": 0.0-1.0,
    "extracted_parameters": {"param": "value"},
    "missing_parameters": ["param1", "param2"],
    "entities": {"type": ["entity1", "entity2"]},
    "keywords": ["keyword1", "keyword2"],
    "reasoning": "Brief explanation of analysis"
}

Query: """

    def __init__(self, use_llm: bool = True):
        """
        Initialize router agent.
        
        Args:
            use_llm: Whether to use LLM for analysis
        """
        self.use_llm = use_llm
        
        if use_llm:
            from src.config.settings import get_llm
            self.llm = get_llm(temperature=0)
    
    async def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> QueryAnalysis:
        """
        Analyze a query to determine intent and extract parameters.
        
        Args:
            query: User's query
            context: Previously collected parameters
            conversation_history: Previous conversation turns
            
        Returns:
            QueryAnalysis with intent, parameters, and routing info
        """
        context = context or {}
        
        if self.use_llm:
            try:
                return await self._analyze_with_llm(query, context, conversation_history)
            except Exception as e:
                logger.error(f"LLM analysis failed: {e}, falling back to rules")
                return self._analyze_with_rules(query, context)
        else:
            return self._analyze_with_rules(query, context)
    
    async def _analyze_with_llm(
        self,
        query: str,
        context: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> QueryAnalysis:
        """Analyze using LLM."""
        # Build prompt with context
        prompt = self.ANALYSIS_PROMPT + query
        
        if context:
            prompt += f"\n\nPreviously collected parameters: {json.dumps(context)}"
        
        if conversation_history:
            recent = conversation_history[-4:]  # Last 2 turns
            conv_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
            prompt += f"\n\nRecent conversation:\n{conv_text}"
        
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        
        # Parse response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content.strip())
        
        # Convert to QueryAnalysis
        intent = QueryIntent(result.get("intent", "general_inquiry"))
        complexity = QueryComplexity(result.get("complexity", "moderate"))
        
        extracted = result.get("extracted_parameters", {})
        missing = result.get("missing_parameters", [])
        
        # Merge with existing context
        merged_params = {**context, **extracted}
        
        # Determine actual missing params based on intent
        actual_missing = self._determine_missing_params(intent, merged_params, missing)
        
        # Get suggested values for missing params
        suggested = self._get_suggested_values(actual_missing, merged_params)
        
        # Get required tools
        tools = self.INTENT_TOOLS.get(intent, ["vector_search"])
        
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            complexity=complexity,
            confidence=result.get("confidence", 0.8),
            extracted_parameters=extracted,
            missing_parameters=actual_missing,
            suggested_values=suggested,
            required_tools=tools,
            optional_tools=[],
            follow_up_questions=self._generate_follow_ups(intent, merged_params),
            requires_clarification=len(actual_missing) > 0,
            entities=result.get("entities", {}),
            keywords=result.get("keywords", [])
        )
    
    def _analyze_with_rules(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> QueryAnalysis:
        """Rule-based fallback analysis."""
        query_lower = query.lower()
        
        # Intent detection
        intent = self._detect_intent_rules(query_lower)
        
        # Parameter extraction
        extracted = self._extract_params_rules(query_lower)
        merged = {**context, **extracted}
        
        # Missing parameters
        missing = self._determine_missing_params(intent, merged, [])
        
        # Complexity
        complexity = self._estimate_complexity(intent, merged, missing)
        
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            complexity=complexity,
            confidence=0.7,  # Lower confidence for rule-based
            extracted_parameters=extracted,
            missing_parameters=missing,
            suggested_values=self._get_suggested_values(missing, merged),
            required_tools=self.INTENT_TOOLS.get(intent, ["vector_search"]),
            requires_clarification=len(missing) > 0
        )
    
    def _detect_intent_rules(self, query: str) -> QueryIntent:
        """Detect intent using keyword rules."""
        intent_keywords = {
            QueryIntent.PRODUCT_SELECTION: [
                "which", "recommend", "select", "best", "suitable", "what gasket",
                "what seal", "need a", "looking for"
            ],
            QueryIntent.PRICING_QUOTE: [
                "price", "cost", "quote", "pricing", "how much", "estimate"
            ],
            QueryIntent.ORDER_STATUS: [
                "order", "track", "shipment", "delivery", "status", "where is"
            ],
            QueryIntent.TECHNICAL_SUPPORT: [
                "problem", "issue", "help", "support", "not working", "failed"
            ],
            QueryIntent.DOCUMENTATION: [
                "datasheet", "document", "specification sheet", "certificate",
                "tds", "msds", "generate"
            ],
            QueryIntent.COMPLIANCE_CHECK: [
                "api 622", "api 624", "certified", "comply", "standard", "approval"
            ],
            QueryIntent.STOCK_AVAILABILITY: [
                "stock", "available", "inventory", "in stock", "lead time"
            ]
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in query for kw in keywords):
                return intent
        
        return QueryIntent.GENERAL_INQUIRY
    
    def _extract_params_rules(self, query: str) -> Dict[str, Any]:
        """Extract parameters using regex patterns."""
        import re
        params = {}
        
        # Temperature
        temp_match = re.search(r'(-?\d+)\s*(?:to|-)?\s*(-?\d+)?\s*°?([CFK])', query, re.I)
        if temp_match:
            params["temperature"] = temp_match.group(0)
        
        # Pressure
        pressure_match = re.search(r'(\d+)\s*(bar|psi|mpa|kpa)', query, re.I)
        if pressure_match:
            params["pressure"] = pressure_match.group(0)
        
        # Industry
        industries = ["oil and gas", "oil & gas", "chemical", "pharmaceutical", 
                     "food", "power", "petrochemical", "refinery"]
        for ind in industries:
            if ind in query:
                params["industry"] = ind
                break
        
        # Standards
        standards = re.findall(r'api\s*\d+[a-z]*|shell\s*spe|fda|asme|astm', query, re.I)
        if standards:
            params["certifications"] = standards
        
        return params
    
    def _determine_missing_params(
        self,
        intent: QueryIntent,
        current_params: Dict[str, Any],
        llm_suggested: List[str]
    ) -> List[str]:
        """Determine which critical parameters are missing."""
        if intent != QueryIntent.PRODUCT_SELECTION:
            return llm_suggested[:2]  # Limit for non-product queries
        
        missing = []
        
        # Check required params
        for param in self.PRODUCT_SELECTION_PARAMS["required"]:
            if param not in current_params:
                missing.append(param)
        
        # Check recommended params (limit to prevent too many questions)
        if not missing:
            for param in self.PRODUCT_SELECTION_PARAMS["recommended"][:2]:
                if param not in current_params:
                    missing.append(param)
        
        return missing[:2]  # Max 2 missing params at a time
    
    def _get_suggested_values(
        self,
        missing_params: List[str],
        current_params: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Get suggested values for missing parameters."""
        suggestions = {}
        industry = current_params.get("industry", "").lower().replace(" ", "_")
        
        industry_data = self.INDUSTRY_SUGGESTIONS.get(
            industry,
            self.INDUSTRY_SUGGESTIONS.get("oil_gas", {})
        )
        
        for param in missing_params:
            if param == "industry":
                suggestions[param] = [
                    "Oil & Gas", "Chemical", "Pharmaceutical", 
                    "Food & Beverage", "Power Generation"
                ]
            elif param == "certification":
                suggestions[param] = industry_data.get("certifications", [])
            elif param == "media":
                suggestions[param] = industry_data.get("media", [])
            elif param == "equipment_type":
                suggestions[param] = industry_data.get("equipment", [])
            elif param == "temperature":
                suggestions[param] = [
                    "Below 0°C (Cryogenic)",
                    "0-100°C (Ambient)",
                    "100-300°C (Medium)",
                    "300-500°C (High)",
                    "Above 500°C (Extreme)"
                ]
            elif param == "pressure":
                suggestions[param] = [
                    "Low (<10 bar)",
                    "Medium (10-100 bar)",
                    "High (100-400 bar)",
                    "Very High (>400 bar)"
                ]
        
        return suggestions
    
    def _estimate_complexity(
        self,
        intent: QueryIntent,
        params: Dict[str, Any],
        missing: List[str]
    ) -> QueryComplexity:
        """Estimate query complexity."""
        if intent == QueryIntent.ORDER_STATUS:
            return QueryComplexity.SIMPLE
        
        if intent == QueryIntent.PRODUCT_SELECTION:
            if len(missing) > 2:
                return QueryComplexity.COMPLEX
            if any(k in params for k in ["certification", "compliance"]):
                return QueryComplexity.COMPLEX
            return QueryComplexity.MODERATE
        
        if intent == QueryIntent.COMPLIANCE_CHECK:
            return QueryComplexity.COMPLEX
        
        return QueryComplexity.MODERATE
    
    def _generate_follow_ups(
        self,
        intent: QueryIntent,
        params: Dict[str, Any]
    ) -> List[str]:
        """Generate relevant follow-up questions."""
        follow_ups = []
        
        if intent == QueryIntent.PRODUCT_SELECTION:
            if "certification" not in params:
                follow_ups.append("Would you like to see certified options for specific standards?")
            if "quantity" not in params:
                follow_ups.append("How many units do you need? (for pricing)")
        
        elif intent == QueryIntent.PRICING_QUOTE:
            follow_ups.append("Would you like a formal quote document?")
            follow_ups.append("Are there any specific delivery requirements?")
        
        return follow_ups[:2]
