"""
Specialized Domain Agents
These agents are experts in specific domains and provide more precise responses
than the general-purpose agents. They are delegated to by the Orchestrator.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings, get_llm
from src.agentic.tools.base_tool import BaseTool
from src.agentic.agents.react_agent import ReActAgent, ReActResult

logger = logging.getLogger(__name__)


# ============================================================================
# DYNAMIC DATA ACCESS
# All product/industry/company data is loaded dynamically from JSON files
# via the JDJonesDataLoader. This eliminates hardcoded data and ensures
# the system always uses the latest information without code changes.
# ============================================================================

def get_industry_recommendations(industry: str) -> Optional[Dict[str, Any]]:
    """
    Get industry-specific product recommendations dynamically from JSON data.
    
    Args:
        industry: Industry identifier (e.g., 'oil_refinery', 'petrochemical_plant')
        
    Returns:
        Dictionary with industry name, description, applications, and notes
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        industry_data = loader.get_industry_recommendations(industry)
        if industry_data:
            return industry_data.to_dict()
        return None
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return None
    except Exception as e:
        logger.error(f"Error loading industry recommendations: {e}")
        return None


def get_all_industry_recommendations() -> Dict[str, Dict[str, Any]]:
    """
    Get all industry product recommendations dynamically from JSON data.
    
    Returns:
        Dictionary mapping industry_id -> industry data
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        industries = loader.get_all_industries()
        return {ind_id: ind_data.to_dict() for ind_id, ind_data in industries.items()}
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return {}
    except Exception as e:
        logger.error(f"Error loading industry recommendations: {e}")
        return {}


def get_company_information() -> Dict[str, Any]:
    """
    Get company information dynamically from JSON data.
    
    Returns:
        Dictionary with company info (name, vision, mission, history, etc.)
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        return loader.get_company_information()
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return {}
    except Exception as e:
        logger.error(f"Error loading company information: {e}")
        return {}


def get_product_data(code: str) -> Optional[Dict[str, Any]]:
    """
    Get product data dynamically from JSON data.
    
    Args:
        code: Product code (e.g., 'NA 715')
        
    Returns:
        Dictionary with product details or None if not found
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        product = loader.get_product(code)
        if product:
            return product.to_dict()
        return None
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return None
    except Exception as e:
        logger.error(f"Error loading product data: {e}")
        return None


def get_all_products() -> Dict[str, Dict[str, Any]]:
    """
    Get all products dynamically from JSON data.
    
    Returns:
        Dictionary mapping product_code -> product data
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        products = loader.get_all_products()
        return {code: prod.to_dict() for code, prod in products.items()}
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return {}
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        return {}


def get_product_certifications(code: str) -> List[str]:
    """
    Get certifications for a product dynamically from JSON data.
    
    Args:
        code: Product code (e.g., 'NA 715')
        
    Returns:
        List of certification strings
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        return loader.get_product_certifications(code)
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return []
    except Exception as e:
        logger.error(f"Error loading certifications: {e}")
        return []


def search_products(
    query: Optional[str] = None,
    category: Optional[str] = None,
    industry: Optional[str] = None,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    application: Optional[str] = None,
    certification: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search products with multiple criteria dynamically from JSON data.
    
    Returns:
        List of matching product dictionaries
    """
    try:
        from src.data_ingestion.jd_jones_data_loader import get_data_loader
        loader = get_data_loader()
        products = loader.search_products(
            query=query,
            category=category,
            industry=industry,
            min_temp=min_temp,
            max_temp=max_temp,
            application=application,
            certification=certification,
        )
        return [p.to_dict() for p in products]
    except ImportError:
        logger.warning("JDJonesDataLoader not available, falling back to empty data")
        return []
    except Exception as e:
        logger.error(f"Error searching products: {e}")
        return []



class SpecialistDomain(str, Enum):
    """Domains for specialized agents."""
    TECHNICAL_SPECS = "technical_specifications"
    COMPLIANCE = "compliance_and_standards"
    PRODUCT_SELECTION = "product_selection"
    TROUBLESHOOTING = "troubleshooting"
    PRICING_QUOTES = "pricing_quotes"
    ORDER_SUPPORT = "order_support"


@dataclass
class ThoughtStep:
    """A single step in the agent's reasoning process."""
    step_number: int
    step_type: str  # thought, action, observation, decision, delegation
    content: str
    agent_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "content": self.content,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentThoughtTrace:
    """Complete thought trace from an agent execution."""
    trace_id: str
    agent_name: str
    query: str
    steps: List[ThoughtStep] = field(default_factory=list)
    final_answer: str = ""
    delegated_to: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_time_ms: float = 0
    success: bool = True
    
    def add_thought(self, content: str, metadata: Dict[str, Any] = None):
        """Add a thought step."""
        self.steps.append(ThoughtStep(
            step_number=len(self.steps) + 1,
            step_type="thought",
            content=content,
            agent_name=self.agent_name,
            metadata=metadata or {}
        ))
    
    def add_action(self, tool_name: str, parameters: Dict[str, Any]):
        """Add an action step."""
        self.steps.append(ThoughtStep(
            step_number=len(self.steps) + 1,
            step_type="action",
            content=f"Executing tool: {tool_name}",
            agent_name=self.agent_name,
            metadata={"tool": tool_name, "parameters": parameters}
        ))
        self.tools_used.append(tool_name)
    
    def add_observation(self, content: str, source: str = None):
        """Add an observation step."""
        self.steps.append(ThoughtStep(
            step_number=len(self.steps) + 1,
            step_type="observation",
            content=content[:500] + "..." if len(content) > 500 else content,
            agent_name=self.agent_name,
            metadata={"source": source} if source else {}
        ))
    
    def add_delegation(self, specialist_name: str, reason: str):
        """Record a delegation to a specialist agent."""
        self.steps.append(ThoughtStep(
            step_number=len(self.steps) + 1,
            step_type="delegation",
            content=f"Delegating to {specialist_name}: {reason}",
            agent_name=self.agent_name,
            metadata={"delegated_to": specialist_name}
        ))
        self.delegated_to.append(specialist_name)
    
    def add_decision(self, content: str):
        """Add a decision step."""
        self.steps.append(ThoughtStep(
            step_number=len(self.steps) + 1,
            step_type="decision",
            content=content,
            agent_name=self.agent_name
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "delegated_to": self.delegated_to,
            "tools_used": self.tools_used,
            "total_time_ms": self.total_time_ms,
            "success": self.success,
            "step_count": len(self.steps)
        }


class TechnicalSpecsAgent(ReActAgent):
    """
    Specialized agent for technical specifications.
    More precise than general agent for:
    - Product specifications lookup
    - Material properties
    - Operating temperature/pressure limits
    - Chemical compatibility
    """
    
    SPECIALIST_PROMPT = """You are a TECHNICAL SPECIFICATIONS SPECIALIST for JD Jones Manufacturing.

YOUR EXPERTISE:
- Industrial sealing products (gaskets, packings, expansion joints, sheet materials)
- Material properties (PTFE, graphite, aramid, rubber compounds)
- Operating limits (temperature, pressure, chemical resistance)
- Industry specifications and material datasheets

CRITICAL RULES:
1. ALWAYS provide exact numerical values from specifications (not approximations)
2. ALWAYS cite the specific product code/material for each specification
3. If a specification is not in the data, say "Specification not found in documentation"
4. For chemical compatibility, specify resistance levels (Excellent/Good/Fair/Poor)
5. Include safety warnings for extreme operating conditions

RESPONSE FORMAT:
- Lead with the most critical specifications
- Use tables for comparing multiple products
- Include units for all measurements
- Note any limitations or caveats

You have superior precision compared to general agents for technical data.

AVAILABLE TOOLS:
{tools_description}

Current Query: {query}
Context: {context}
"""
    
    def __init__(self, tools: Optional[Dict[str, BaseTool]] = None):
        super().__init__(tools=tools, max_iterations=4)
        self.agent_name = "TechnicalSpecsAgent"
    
    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ReActResult:
        """Execute with specialized technical prompt."""
        # Override system prompt for precision
        original_prompt = self.REACT_SYSTEM_PROMPT
        self.REACT_SYSTEM_PROMPT = self.SPECIALIST_PROMPT
        
        try:
            result = await super().execute(query, context)
            return result
        finally:
            self.REACT_SYSTEM_PROMPT = original_prompt


class ComplianceAgent(ReActAgent):
    """
    Specialized agent for compliance and standards verification.
    More precise than general agent for:
    - API 622/624, Shell SPE, Saudi Aramco standards
    - FDA, ATEX, ASME certifications
    - Fire-safe testing requirements
    - Regulatory compliance verification
    """
    
    SPECIALIST_PROMPT = """You are a COMPLIANCE AND STANDARDS SPECIALIST for JD Jones Manufacturing.

YOUR EXPERTISE:
- Industry standards: API 622, API 624, Shell SPE 77/312, Saudi Aramco specs
- Certifications: FDA, ATEX, ASME, CE, TA-Luft, ISO standards
- Fire-safe testing: API 607, API 6FA, ISO 15848
- Material compliance and traceability requirements

CRITICAL RULES:
1. VERIFY exact certification numbers/versions when citing compliance
2. DISTINGUISH between "certified" (tested & approved) vs "suitable for" (meets requirements)
3. Note expiration dates or revision levels of certifications if available
4. Identify if third-party testing documentation is required
5. Flag any compliance gaps or additional requirements needed

RESPONSE FORMAT:
- State compliance status clearly (Compliant/Non-Compliant/Partial)
- List specific standards met with version numbers
- Note any caveats or conditions for compliance
- Recommend additional certifications if needed

You have superior precision compared to general agents for compliance verification.

AVAILABLE TOOLS:
{tools_description}

Current Query: {query}
Context: {context}
"""
    
    def __init__(self, tools: Optional[Dict[str, BaseTool]] = None):
        super().__init__(tools=tools, max_iterations=4)
        self.agent_name = "ComplianceAgent"
    
    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ReActResult:
        """Execute with specialized compliance prompt."""
        original_prompt = self.REACT_SYSTEM_PROMPT
        self.REACT_SYSTEM_PROMPT = self.SPECIALIST_PROMPT
        
        try:
            result = await super().execute(query, context)
            return result
        finally:
            self.REACT_SYSTEM_PROMPT = original_prompt


class TroubleshootingAgent(ReActAgent):
    """
    Specialized agent for technical troubleshooting.
    More precise than general agent for:
    - Seal failure diagnosis
    - Installation issues
    - Performance problems
    - Root cause analysis
    """
    
    SPECIALIST_PROMPT = """You are a TROUBLESHOOTING SPECIALIST for JD Jones Manufacturing.

YOUR EXPERTISE:
- Seal failure modes (extrusion, chemical attack, thermal degradation, mechanical damage)
- Installation best practices and common mistakes
- Performance optimization and adjustment
- Root cause analysis methodology

CRITICAL RULES:
1. ALWAYS ask clarifying questions if failure mode is unclear
2. Consider ALL potential causes before recommending solutions
3. Prioritize safety-critical issues first
4. Recommend diagnostic steps in logical order
5. Distinguish between temporary fixes and permanent solutions

TROUBLESHOOTING APPROACH:
1. Symptom identification
2. Failure mode analysis
3. Root cause determination
4. Solution recommendation
5. Prevention measures

RESPONSE FORMAT:
- Start with most likely cause based on symptoms
- Provide step-by-step diagnostic process
- Include visual inspection checkpoints
- Recommend replacement products if needed
- Note safety precautions

You have superior diagnostic precision compared to general agents.

AVAILABLE TOOLS:
{tools_description}

Current Query: {query}
Context: {context}
"""
    
    def __init__(self, tools: Optional[Dict[str, BaseTool]] = None):
        super().__init__(tools=tools, max_iterations=5)
        self.agent_name = "TroubleshootingAgent"
    
    async def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> ReActResult:
        """Execute with specialized troubleshooting prompt."""
        original_prompt = self.REACT_SYSTEM_PROMPT
        self.REACT_SYSTEM_PROMPT = self.SPECIALIST_PROMPT
        
        try:
            result = await super().execute(query, context)
            return result
        finally:
            self.REACT_SYSTEM_PROMPT = original_prompt


class PricingQuoteAgent(ReActAgent):
    """
    Specialized agent for pricing and quotes.
    More precise than general agent for:
    - Price calculations
    - Volume discounts
    - Lead times
    - Quote generation
    """
    
    SPECIALIST_PROMPT = """You are a PRICING AND QUOTATION SPECIALIST for JD Jones Manufacturing.

YOUR EXPERTISE:
- Product pricing and cost calculation
- Volume discount tiers and special pricing
- Lead times and delivery schedules
- Custom product quotes and MOQ requirements

CRITICAL RULES:
1. ALWAYS verify current pricing from the database (prices may change)
2. Apply appropriate discount tiers based on quantity
3. Include lead times for standard vs custom products
4. Note any minimum order quantities (MOQ)
5. Flag products requiring custom manufacturing quotes

PRICING CONSIDERATIONS:
- Standard products: Immediate pricing available
- Custom sizes: May require engineering review
- Large quantities: Volume discounts apply (10%, 15%, 20%+)
- Rush orders: Expedite fees may apply

RESPONSE FORMAT:
- Provide unit price and extended price
- Show discount tier applied
- Include estimated lead time
- Note any additional fees (cutting, shipping)
- Validity period for quote

You have superior precision for pricing compared to general agents.

AVAILABLE TOOLS:
{tools_description}

Current Query: {query}
Context: {context}
"""
    
    def __init__(self, tools: Optional[Dict[str, BaseTool]] = None):
        super().__init__(tools=tools, max_iterations=3)
        self.agent_name = "PricingQuoteAgent"


class SpecialistAgentRegistry:
    """Registry for specialized agents with delegation logic."""
    
    def __init__(self):
        self.specialists: Dict[SpecialistDomain, ReActAgent] = {}
        self._init_specialists()
    
    def _init_specialists(self):
        """Initialize all specialized agents."""
        self.specialists[SpecialistDomain.TECHNICAL_SPECS] = TechnicalSpecsAgent()
        self.specialists[SpecialistDomain.COMPLIANCE] = ComplianceAgent()
        self.specialists[SpecialistDomain.TROUBLESHOOTING] = TroubleshootingAgent()
        self.specialists[SpecialistDomain.PRICING_QUOTES] = PricingQuoteAgent()
    
    def get_specialist(self, domain: SpecialistDomain) -> Optional[ReActAgent]:
        """Get a specialist agent by domain."""
        return self.specialists.get(domain)
    
    def register_tools(self, tools: Dict[str, BaseTool]):
        """Register tools with all specialists."""
        for specialist in self.specialists.values():
            specialist.tools = tools
    
    def determine_specialist(self, query: str, intent: str) -> Optional[SpecialistDomain]:
        """
        Determine which specialist should handle the query.
        Returns None if general agent should handle it.
        """
        query_lower = query.lower()
        
        # Technical specifications indicators
        tech_keywords = [
            "specification", "specs", "temperature", "pressure", 
            "material", "properties", "datasheet", "dimensions",
            "chemical resistance", "compatibility", "limits"
        ]
        
        # Compliance indicators
        compliance_keywords = [
            "api 622", "api 624", "certification", "certified",
            "compliant", "compliance", "standard", "fda approved",
            "atex", "fire safe", "shell spe", "saudi aramco", "asme"
        ]
        
        # Troubleshooting indicators
        trouble_keywords = [
            "problem", "issue", "failure", "failed", "leaking",
            "troubleshoot", "diagnose", "not working", "broken",
            "why is", "cause", "fix", "repair"
        ]
        
        # Pricing indicators
        pricing_keywords = [
            "price", "pricing", "quote", "cost", "how much",
            "discount", "lead time", "delivery", "moq", "order"
        ]
        
        # Count matches for each domain
        scores = {
            SpecialistDomain.TECHNICAL_SPECS: sum(1 for k in tech_keywords if k in query_lower),
            SpecialistDomain.COMPLIANCE: sum(1 for k in compliance_keywords if k in query_lower),
            SpecialistDomain.TROUBLESHOOTING: sum(1 for k in trouble_keywords if k in query_lower),
            SpecialistDomain.PRICING_QUOTES: sum(1 for k in pricing_keywords if k in query_lower),
        }
        
        # Also consider intent
        if intent:
            intent_lower = intent.lower()
            if "technic" in intent_lower or "specification" in intent_lower:
                scores[SpecialistDomain.TECHNICAL_SPECS] += 2
            if "compliance" in intent_lower or "certification" in intent_lower:
                scores[SpecialistDomain.COMPLIANCE] += 2
            if "trouble" in intent_lower or "support" in intent_lower:
                scores[SpecialistDomain.TROUBLESHOOTING] += 2
            if "pricing" in intent_lower or "quote" in intent_lower:
                scores[SpecialistDomain.PRICING_QUOTES] += 2
        
        # Return domain with highest score if above threshold
        max_score = max(scores.values())
        if max_score >= 2:
            for domain, score in scores.items():
                if score == max_score:
                    return domain
        
        return None
    
    def list_specialists(self) -> List[Dict[str, Any]]:
        """List all available specialists."""
        return [
            {
                "domain": domain.value,
                "agent_name": agent.agent_name if hasattr(agent, 'agent_name') else domain.value,
                "description": agent.__doc__.strip().split('\n')[0] if agent.__doc__ else ""
            }
            for domain, agent in self.specialists.items()
        ]
