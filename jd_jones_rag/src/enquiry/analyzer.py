"""
Enquiry Analyzer - Multi-Agent AI-powered enquiry analysis.

Uses specialized sub-agents for different aspects of enquiry analysis,
then synthesizes results into a comprehensive structured analysis.
"""

import logging
import re
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings, get_llm
from src.enquiry.models import (
    Enquiry, EnquiryType, EnquiryPriority, AIAnalysis, 
    SuggestedResponse, StructuredRequirements
)

logger = logging.getLogger(__name__)


class ClassifierAgent:
    """
    Specialized agent for classifying enquiry type and priority.
    Uses structured prompts for accurate classification.
    """
    
    def __init__(self):
        self.llm = get_llm(temperature=0.1)  # Low temperature for classification
    
    async def classify(self, message: str, subject: str = "") -> Dict[str, Any]:
        """Classify enquiry type and priority."""
        system_prompt = """You are a classification expert for industrial product enquiries.
Classify the enquiry into EXACTLY ONE type and priority level.

TYPES (choose one):
- product_selection: Customer wants help choosing products
- technical_assistance: Customer needs technical support/specifications
- pricing: Customer asking about prices/quotes
- order_status: Customer asking about existing order
- quotation: Customer requesting formal quotation
- complaint: Customer has an issue/complaint
- general: General information request
- other: Doesn't fit other categories

PRIORITY (choose one):
- urgent: Contains words like "urgent", "ASAP", "immediately", "emergency", deadline within 1-2 days
- high: Important project, short deadline (within 1-2 weeks), key customer indicators
- medium: Standard business enquiry, reasonable timeline
- low: General information, no time pressure

URGENCY_SCORE (1-5):
1 = No urgency, informational
2 = Low urgency, flexible timeline
3 = Normal business urgency
4 = High urgency, short deadline
5 = Critical/Emergency

Respond in EXACTLY this format:
TYPE: <type>
PRIORITY: <priority>
URGENCY_SCORE: <1-5>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""
        
        user_message = f"Subject: {subject}\n\nMessage:\n{message}"
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ])
            
            content = response.content
            
            type_match = re.search(r'TYPE:\s*(\w+)', content, re.IGNORECASE)
            priority_match = re.search(r'PRIORITY:\s*(\w+)', content, re.IGNORECASE)
            urgency_match = re.search(r'URGENCY_SCORE:\s*(\d)', content)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', content)
            
            return {
                "type": type_match.group(1).lower() if type_match else "general",
                "priority": priority_match.group(1).lower() if priority_match else "medium",
                "urgency_score": int(urgency_match.group(1)) if urgency_match else 3,
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5
            }
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"type": "general", "priority": "medium", "urgency_score": 3, "confidence": 0.3}


class RequirementsExtractor:
    """
    Specialized agent for extracting structured requirements from enquiries.
    """
    
    def __init__(self):
        self.llm = get_llm(temperature=0.2)
    
    async def extract(self, message: str) -> StructuredRequirements:
        """Extract structured requirements from message."""
        system_prompt = """You are an expert at extracting technical requirements from industrial enquiries.
Extract ALL relevant information into structured fields. If information is not present, leave blank.

Extract the following (use EXACTLY these field names):
- INDUSTRY: e.g., mining, oil & gas, power plant, chemical, pharmaceutical, food
- APPLICATION: e.g., valve, pump, agitator, flange, boiler
- EQUIPMENT_TYPE: specific equipment mentioned
- TEMPERATURE: operating temperature range (include units)
- PRESSURE: operating pressure (include units)
- MEDIA: what fluids/materials are being handled
- CERTIFICATIONS: any certifications mentioned (FDA, API, etc.)
- DIMENSIONS: size requirements (OD x ID x thickness, shaft size, etc.)
- QUANTITY: how many units needed
- DELIVERY_URGENCY: timeline mentioned (ASAP, 2 weeks, flexible, etc.)
- PROJECT_DEADLINE: specific deadline date if mentioned
- BUDGET_MENTIONED: YES or NO
- BUDGET_RANGE: approximate budget if mentioned

Format response as:
INDUSTRY: <value or NONE>
APPLICATION: <value or NONE>
EQUIPMENT_TYPE: <value or NONE>
TEMPERATURE: <value or NONE>
PRESSURE: <value or NONE>
MEDIA: <value or NONE>
CERTIFICATIONS: <value or NONE>
DIMENSIONS: <value or NONE>
QUANTITY: <value or NONE>
DELIVERY_URGENCY: <value or NONE>
PROJECT_DEADLINE: <value or NONE>
BUDGET_MENTIONED: <YES or NO>
BUDGET_RANGE: <value or NONE>
"""
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Extract requirements from:\n\n{message}")
            ])
            
            content = response.content
            
            def get_field(name: str) -> Optional[str]:
                match = re.search(rf'{name}:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
                if match:
                    val = match.group(1).strip()
                    return None if val.upper() == "NONE" else val
                return None
            
            def get_list(name: str) -> List[str]:
                val = get_field(name)
                if val:
                    return [v.strip() for v in val.split(",") if v.strip()]
                return []
            
            return StructuredRequirements(
                industry=get_field("INDUSTRY"),
                application=get_field("APPLICATION"),
                equipment_type=get_field("EQUIPMENT_TYPE"),
                operating_temperature=get_field("TEMPERATURE"),
                operating_pressure=get_field("PRESSURE"),
                media_handled=get_field("MEDIA"),
                certifications_needed=get_list("CERTIFICATIONS"),
                dimensions=get_field("DIMENSIONS"),
                quantity=get_field("QUANTITY"),
                delivery_urgency=get_field("DELIVERY_URGENCY"),
                project_deadline=get_field("PROJECT_DEADLINE"),
                budget_mentioned=get_field("BUDGET_MENTIONED") == "YES",
                budget_range=get_field("BUDGET_RANGE")
            )
            
        except Exception as e:
            logger.error(f"Requirements extraction error: {e}")
            return StructuredRequirements()


class ProductMatchAgent:
    """
    Specialized agent for matching enquiries to products.
    Uses the product catalog for grounded recommendations.
    """
    
    # Known product code patterns for JD Jones
    PRODUCT_PATTERNS = [
        r'NA[\s\-]?(\d{3}[A-Z]*)',  # NA 701, NA-707, NA 750
        r'product[\s:]*(NA[\s\-]?\d{3}[A-Z]*)',  # Product: NA 701
        r'code[\s:]*(NA[\s\-]?\d{3}[A-Z]*)',  # Code: NA 701
    ]
    
    def __init__(self):
        self.llm = get_llm(temperature=0.3)
    
    def _extract_product_codes(self, message: str) -> List[str]:
        """Extract explicitly mentioned product codes from message."""
        codes = set()
        message_upper = message.upper()
        
        # Look for NA XXX patterns
        for pattern in self.PRODUCT_PATTERNS:
            matches = re.findall(pattern, message_upper, re.IGNORECASE)
            for match in matches:
                # Normalize code format
                if isinstance(match, tuple):
                    match = match[0]
                code = match.strip()
                if not code.startswith('NA'):
                    code = f"NA {code}"
                else:
                    # Ensure proper spacing: NA701 -> NA 701
                    code = re.sub(r'NA[\s\-]*(\d+)', r'NA \1', code)
                codes.add(code.upper())
        
        # Also check for direct mentions like "701" after "NA" or "product"
        simple_matches = re.findall(r'\b(\d{3})\b', message)
        for num in simple_matches:
            if f"NA {num}" in codes or f"NA{num}" in message_upper or int(num) >= 700 and int(num) <= 799:
                codes.add(f"NA {num}")
        
        return list(codes)
    
    async def find_products(self, requirements: StructuredRequirements, message: str) -> Dict[str, Any]:
        """Find matching products based on requirements and explicit mentions."""
        mentioned_products = []
        matched_products = []
        
        # FIRST: Extract explicitly mentioned product codes
        mentioned_products = self._extract_product_codes(message)
        logger.info(f"Explicitly mentioned products: {mentioned_products}")
        
        try:
            from src.data_ingestion.product_catalog_retriever import get_product_retriever
            retriever = get_product_retriever()
            
            # If products are explicitly mentioned, prioritize them
            if mentioned_products:
                for code in mentioned_products:
                    # Verify product exists in catalog
                    try:
                        # Query specifically for this product
                        matches = retriever.find_products(
                            query=code,
                            limit=1
                        )
                        if matches:
                            matched_products.append(code)
                    except Exception:
                        matched_products.append(code)  # Include even if not verified
                
                return {
                    "products": matched_products if matched_products else mentioned_products,
                    "confidence": "high",  # High because explicitly mentioned
                    "match_count": len(matched_products or mentioned_products),
                    "explicitly_mentioned": True
                }
            
            # SECOND: Semantic matching based on requirements
            temp = None
            if requirements.operating_temperature:
                temp_match = re.search(r'(\d+)', requirements.operating_temperature)
                if temp_match:
                    temp = float(temp_match.group(1))
            
            pressure = None
            if requirements.operating_pressure:
                pressure_match = re.search(r'(\d+)', requirements.operating_pressure)
                if pressure_match:
                    pressure = float(pressure_match.group(1))
            
            # Search products semantically
            matches = retriever.find_products(
                industry=requirements.industry,
                application=requirements.application,
                operating_temp=temp,
                operating_pressure=pressure,
                limit=5
            )
            
            products = []
            confidence = "low"
            
            for match in matches:
                products.append(match.product.code)
                if match.confidence.value == "high":
                    confidence = "high"
                elif match.confidence.value == "medium" and confidence != "high":
                    confidence = "medium"
            
            return {
                "products": products,
                "confidence": confidence,
                "match_count": len(matches),
                "explicitly_mentioned": False
            }
            
        except Exception as e:
            logger.warning(f"Product matching error: {e}")
            # Return explicitly mentioned products even if retriever fails
            if mentioned_products:
                return {
                    "products": mentioned_products, 
                    "confidence": "medium", 
                    "match_count": len(mentioned_products),
                    "explicitly_mentioned": True
                }
            return {"products": [], "confidence": "low", "match_count": 0, "explicitly_mentioned": False}


class SummarizerAgent:
    """
    Specialized agent for creating structured summaries.
    """
    
    def __init__(self):
        self.llm = get_llm(temperature=0.4)
    
    async def summarize(
        self, 
        message: str, 
        classification: Dict[str, Any],
        requirements: StructuredRequirements
    ) -> Dict[str, Any]:
        """Create structured summary of the enquiry."""
        system_prompt = """You are an expert at summarizing customer enquiries for internal team review.
Create a structured summary that allows the team to understand the enquiry at a glance.

Provide:
1. ONE_LINER: A single sentence (max 15 words) capturing the core request
2. SUMMARY: 2-3 sentences providing full context
3. CUSTOMER_INTENT: What does the customer ultimately want to achieve?
4. MAIN_ASK: The primary request/question (one clear statement)
5. SECONDARY_ASKS: Any additional requests (bullet points, or "None")
6. KEY_POINTS: 3-5 bullet points of important information
7. RECOMMENDED_ACTIONS: What should the team do? (bullet points)
8. REQUIRES_TECHNICAL: YES or NO - does this need technical team review?
9. REQUIRES_PRICING: YES or NO - does customer want pricing?
10. REQUIRES_SAMPLES: YES or NO - are samples requested?
11. SENTIMENT: positive, neutral, or negative

Format EXACTLY as shown with each field on its own line.
"""
        
        context = f"""
Customer Enquiry:
{message}

Classification: {classification.get('type', 'general')} (Priority: {classification.get('priority', 'medium')})

Extracted Requirements:
- Industry: {requirements.industry or 'Not specified'}
- Application: {requirements.application or 'Not specified'}
- Temperature: {requirements.operating_temperature or 'Not specified'}
- Pressure: {requirements.operating_pressure or 'Not specified'}
- Quantity: {requirements.quantity or 'Not specified'}
- Delivery: {requirements.delivery_urgency or 'Not specified'}
"""
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ])
            
            content = response.content
            
            def get_field(name: str) -> str:
                match = re.search(rf'{name}:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
                return match.group(1).strip() if match else ""
            
            def get_list(name: str) -> List[str]:
                # Find the field and parse bullet points
                pattern = rf'{name}:\s*(.*?)(?=\n[A-Z_]+:|$)'
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    text = match.group(1).strip()
                    if text.lower() == "none":
                        return []
                    # Split by bullet points or newlines
                    items = re.split(r'[\n•\-\*]', text)
                    return [i.strip() for i in items if i.strip() and i.strip().lower() != "none"]
                return []
            
            return {
                "one_liner": get_field("ONE_LINER"),
                "summary": get_field("SUMMARY"),
                "customer_intent": get_field("CUSTOMER_INTENT"),
                "main_ask": get_field("MAIN_ASK"),
                "secondary_asks": get_list("SECONDARY_ASKS"),
                "key_points": get_list("KEY_POINTS"),
                "recommended_actions": get_list("RECOMMENDED_ACTIONS"),
                "requires_technical": get_field("REQUIRES_TECHNICAL").upper() == "YES",
                "requires_pricing": get_field("REQUIRES_PRICING").upper() == "YES",
                "requires_samples": get_field("REQUIRES_SAMPLES").upper() == "YES",
                "sentiment": get_field("SENTIMENT").lower() or "neutral"
            }
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                "one_liner": "Customer enquiry received",
                "summary": message[:200],
                "customer_intent": "To be determined",
                "main_ask": "Review required",
                "secondary_asks": [],
                "key_points": ["Enquiry received - manual review needed"],
                "recommended_actions": ["Review enquiry manually"],
                "requires_technical": False,
                "requires_pricing": False,
                "requires_samples": False,
                "sentiment": "neutral"
            }


class EnquiryAnalyzer:
    """
    Multi-Agent Enquiry Analyzer.
    
    Orchestrates specialized sub-agents to analyze enquiries:
    1. ClassifierAgent - Classifies type and priority
    2. RequirementsExtractor - Extracts structured requirements
    3. ProductMatchAgent - Finds matching products
    4. SummarizerAgent - Creates structured summary
    
    Then synthesizes all results into a comprehensive AIAnalysis.
    """
    
    def __init__(self):
        """Initialize analyzer with sub-agents."""
        self.classifier = ClassifierAgent()
        self.extractor = RequirementsExtractor()
        self.product_matcher = ProductMatchAgent()
        self.summarizer = SummarizerAgent()
        self.response_llm = get_llm(temperature=0.7)
    
    async def analyze_enquiry(self, enquiry: Enquiry) -> AIAnalysis:
        """
        Analyze an enquiry using multi-agent approach.
        
        Runs specialized sub-agents in parallel, then synthesizes results.
        
        Args:
            enquiry: The enquiry to analyze
            
        Returns:
            AIAnalysis with comprehensive structured analysis
        """
        message = enquiry.raw_message
        subject = enquiry.subject or ""
        
        # Run classification and requirements extraction in parallel
        classification_task = self.classifier.classify(message, subject)
        extraction_task = self.extractor.extract(message)
        
        classification, requirements = await asyncio.gather(
            classification_task,
            extraction_task
        )
        
        # Run product matching and summarization with context
        product_task = self.product_matcher.find_products(requirements, message)
        summary_task = self.summarizer.summarize(message, classification, requirements)
        
        products, summary = await asyncio.gather(
            product_task,
            summary_task
        )
        
        # Map to enums
        try:
            detected_type = EnquiryType(classification["type"])
        except ValueError:
            detected_type = EnquiryType.GENERAL
        
        try:
            detected_priority = EnquiryPriority(classification["priority"])
        except ValueError:
            detected_priority = EnquiryPriority.MEDIUM
        
        # Build urgency indicators from various sources
        urgency_indicators = []
        if requirements.delivery_urgency:
            urgency_indicators.append(requirements.delivery_urgency)
        if requirements.project_deadline:
            urgency_indicators.append(f"Deadline: {requirements.project_deadline}")
        
        # Synthesize into AIAnalysis
        analysis = AIAnalysis(
            # Overview
            summary=summary.get("summary", ""),
            one_liner=summary.get("one_liner", ""),
            
            # Classification
            detected_type=detected_type,
            detected_priority=detected_priority,
            confidence_score=classification.get("confidence", 0.5),
            
            # Structured breakdown
            customer_intent=summary.get("customer_intent", ""),
            main_ask=summary.get("main_ask", ""),
            secondary_asks=summary.get("secondary_asks", []),
            
            # Requirements
            requirements=requirements,
            
            # Key info
            key_points=summary.get("key_points", []),
            
            # Products
            suggested_products=products.get("products", []),
            product_match_confidence=products.get("confidence", "low"),
            
            # Sentiment & urgency
            sentiment=summary.get("sentiment", "neutral"),
            urgency_indicators=urgency_indicators,
            urgency_score=classification.get("urgency_score", 3),
            
            # Actions
            recommended_actions=summary.get("recommended_actions", []),
            requires_technical_review=summary.get("requires_technical", False),
            requires_pricing=summary.get("requires_pricing", False),
            requires_samples=summary.get("requires_samples", False),
            
            # Store sub-agent results for transparency
            sub_agent_analyses={
                "classifier": classification,
                "products": products
            }
        )
        
        logger.info(f"Multi-agent analysis complete for enquiry {enquiry.id}")
        return analysis
    
    async def generate_suggested_response(
        self,
        enquiry: Enquiry,
        tone: str = "professional",
        include_products: bool = True
    ) -> SuggestedResponse:
        """
        Generate a suggested response for an enquiry.
        
        Uses the analysis results AND RAG knowledge base to craft a contextual response.
        
        Args:
            enquiry: The enquiry to respond to
            tone: Response tone (professional, friendly, formal)
            include_products: Whether to include product recommendations
            
        Returns:
            SuggestedResponse with the generated text
        """
        # Get product details if needed
        product_context = ""
        if include_products and enquiry.ai_analysis and enquiry.ai_analysis.suggested_products:
            product_context = await self._get_product_context(enquiry.ai_analysis.suggested_products)
        
        # Get RAG context from knowledge base based on enquiry
        rag_context = await self._get_rag_context(enquiry.raw_message)
        
        system_prompt = f"""You are a customer service representative for JD Jones (Pacmaan), 
a leading manufacturer of industrial sealing products. Generate a {tone} response.

RESPONSE GUIDELINES:
1. Address the customer by name
2. Acknowledge their specific request (use the analysis summary)
3. Provide relevant SPECIFIC information from the knowledge base context provided
4. For product enquiries, include key specifications (temperature range, pressure, materials)
5. For pricing enquiries, inform that a formal quotation will be sent
6. For technical questions, provide detailed technical guidance from the knowledge base
7. Include clear next steps
8. Sign off as "JD Jones Technical Support Team"

IMPORTANT: Use the knowledge base information below to provide SPECIFIC, ACCURATE answers.
NEVER make up specifications - only use what's in the context.
NEVER include actual pricing - always refer to quotation process.

{product_context}

{rag_context}

The response should be ready to send via email.
"""
        
        # Use the structured analysis for context
        analysis = enquiry.ai_analysis
        analysis_context = ""
        if analysis:
            analysis_context = f"""
## Analysis Summary:
- One-liner: {analysis.one_liner}
- Customer Intent: {analysis.customer_intent}
- Main Request: {analysis.main_ask}
- Priority: {analysis.detected_priority.value}
- Suggested Products: {', '.join(analysis.suggested_products) if analysis.suggested_products else 'None identified'}

## Requirements:
{self._format_requirements(analysis.requirements) if analysis.requirements else "Not specified"}

## Recommended Actions:
{', '.join(analysis.recommended_actions) if analysis.recommended_actions else "Standard response"}
"""
        
        user_message = f"""Generate a {tone} response to:

From: {enquiry.customer.name}
Company: {enquiry.customer.company or 'N/A'}
Subject: {enquiry.subject or 'General Enquiry'}

Original Message:
{enquiry.raw_message}

{analysis_context}
"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = await self.response_llm.ainvoke(messages)
            
            return SuggestedResponse(
                response_text=response.content,
                tone=tone,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return SuggestedResponse(
                response_text=f"Dear {enquiry.customer.name},\n\nThank you for your enquiry. Our team is reviewing your request and will respond shortly.\n\nBest regards,\nJD Jones Technical Support Team",
                tone=tone,
                generated_at=datetime.now()
            )
    
    def _format_requirements(self, req: StructuredRequirements) -> str:
        """Format requirements for prompt context."""
        lines = []
        if req.industry: lines.append(f"- Industry: {req.industry}")
        if req.application: lines.append(f"- Application: {req.application}")
        if req.operating_temperature: lines.append(f"- Temperature: {req.operating_temperature}")
        if req.operating_pressure: lines.append(f"- Pressure: {req.operating_pressure}")
        if req.media_handled: lines.append(f"- Media: {req.media_handled}")
        if req.quantity: lines.append(f"- Quantity: {req.quantity}")
        if req.delivery_urgency: lines.append(f"- Timeline: {req.delivery_urgency}")
        return "\n".join(lines) if lines else "No specific requirements extracted"
    
    async def _get_product_context(self, product_codes: List[str]) -> str:
        """Get product context for response generation using RAG knowledge base."""
        context_parts = ["## PRODUCT INFORMATION FROM KNOWLEDGE BASE:"]
        
        try:
            # Try vector search first for detailed product info
            from src.agentic.tools.vector_search_tool import VectorSearchTool
            vector_tool = VectorSearchTool()
            
            for code in product_codes[:3]:
                # Normalize code for search
                search_code = code.upper().replace(" ", " ").strip()
                
                # Search knowledge base for this specific product
                try:
                    results = await vector_tool.search(
                        query=f"{search_code} specifications technical data applications",
                        top_k=3,
                        filter_type="product"
                    )
                    
                    if results and results.get("results"):
                        product_info = []
                        for result in results["results"][:2]:
                            content = result.get("content", "")[:500]
                            if content:
                                product_info.append(content)
                        
                        if product_info:
                            context_parts.append(f"""
**{search_code}**:
{chr(10).join(product_info)}
""")
                            continue
                except Exception as e:
                    logger.debug(f"Vector search for {code} failed: {e}")
                
                # Fallback: Try product catalog retriever
                try:
                    from src.data_ingestion.product_catalog_retriever import get_product_retriever
                    retriever = get_product_retriever()
                    matches = retriever.find_products(query=search_code, limit=1)
                    
                    for match in matches:
                        product = match.product
                        context_parts.append(f"""
**{product.code}** - {product.name}
- Material: {product.material}
- Temperature Range: {product.specs.temperature_min}°C to {product.specs.temperature_max}°C
- Pressure Rating: {product.specs.pressure_max} bar
- Applications: {', '.join(product.applications[:3])}
- Industries: {', '.join(product.industries[:3])}
""")
                        break
                except Exception as e:
                    logger.debug(f"Product catalog lookup for {code} failed: {e}")
            
            return "\n".join(context_parts) if len(context_parts) > 1 else ""
            
        except Exception as e:
            logger.warning(f"Could not get product context: {e}")
            return ""
    
    async def _get_rag_context(self, message: str) -> str:
        """Get relevant context from RAG knowledge base for the enquiry."""
        try:
            from src.agentic.tools.vector_search_tool import VectorSearchTool
            vector_tool = VectorSearchTool()
            
            # Search for relevant context from knowledge base
            results = await vector_tool.search(
                query=message,
                top_k=5
            )
            
            if not results or not results.get("results"):
                return ""
            
            context_parts = ["## RELEVANT KNOWLEDGE BASE INFORMATION:"]
            
            for result in results["results"][:4]:
                content = result.get("content", "")[:600]
                source = result.get("source", "Knowledge Base")
                if content:
                    context_parts.append(f"""
Source: {source}
{content}
---""")
            
            return "\n".join(context_parts) if len(context_parts) > 1 else ""
            
        except Exception as e:
            logger.debug(f"RAG context retrieval failed: {e}")
            return ""


# Singleton instance
_analyzer: Optional[EnquiryAnalyzer] = None


def get_enquiry_analyzer() -> EnquiryAnalyzer:
    """Get singleton analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EnquiryAnalyzer()
    return _analyzer
