"""
Quotation Analyzer - Multi-agent AI system for analyzing quotation requests.

Similar architecture to the enquiry analyzer, this uses specialized sub-agents:
1. RequirementsAnalyzer - Extracts and validates technical requirements
2. ProductMatcher - Matches requirements to products with specifications
3. PricingEstimator - Estimates pricing based on product, quantity, and complexity
4. DeliveryEstimator - Estimates delivery timeline based on inventory and production
5. SummaryGenerator - Creates structured summary for internal review
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import settings
from src.quotation.models import QuotationRequest, QuotationLineItem

logger = logging.getLogger(__name__)


@dataclass
class QuotationRequirements:
    """Extracted technical requirements from quotation request."""
    industry: str = ""
    application: str = ""
    operating_temperature: Optional[str] = None
    operating_pressure: Optional[str] = None
    media_handled: Optional[str] = None
    shaft_speed: Optional[str] = None
    certifications_needed: List[str] = field(default_factory=list)
    special_requirements: List[str] = field(default_factory=list)
    fire_safe_required: bool = False
    fugitive_emission_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "industry": self.industry,
            "application": self.application,
            "operating_temperature": self.operating_temperature,
            "operating_pressure": self.operating_pressure,
            "media_handled": self.media_handled,
            "shaft_speed": self.shaft_speed,
            "certifications_needed": self.certifications_needed,
            "special_requirements": self.special_requirements,
            "fire_safe_required": self.fire_safe_required,
            "fugitive_emission_required": self.fugitive_emission_required
        }


@dataclass
class ProductMatch:
    """A matched product with specifications."""
    product_code: str
    product_name: str
    match_confidence: float  # 0.0 to 1.0
    match_reasons: List[str]
    suggested_specifications: Dict[str, Any]
    alternatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_code": self.product_code,
            "product_name": self.product_name,
            "match_confidence": self.match_confidence,
            "match_reasons": self.match_reasons,
            "suggested_specifications": self.suggested_specifications,
            "alternatives": self.alternatives
        }


@dataclass
class PricingEstimate:
    """Pricing estimate for a quotation."""
    estimated_unit_price: Optional[float] = None
    estimated_total: Optional[float] = None
    price_confidence: str = "low"  # low, medium, high
    pricing_factors: List[str] = field(default_factory=list)
    discount_applicable: bool = False
    volume_discount_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_unit_price": self.estimated_unit_price,
            "estimated_total": self.estimated_total,
            "price_confidence": self.price_confidence,
            "pricing_factors": self.pricing_factors,
            "discount_applicable": self.discount_applicable,
            "volume_discount_percent": self.volume_discount_percent
        }


@dataclass
class DeliveryEstimate:
    """Delivery timeline estimate."""
    estimated_days: int = 14
    delivery_confidence: str = "medium"
    factors: List[str] = field(default_factory=list)
    expedite_possible: bool = True
    expedite_premium_percent: float = 15.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_days": self.estimated_days,
            "delivery_confidence": self.delivery_confidence,
            "factors": self.factors,
            "expedite_possible": self.expedite_possible,
            "expedite_premium_percent": self.expedite_premium_percent
        }


@dataclass
class QuotationAnalysis:
    """Complete AI analysis of a quotation request."""
    # Quick overview
    one_liner: str = ""
    priority: str = "medium"  # low, medium, high, urgent
    complexity: str = "standard"  # simple, standard, complex, custom
    
    # Requirements
    requirements: Optional[QuotationRequirements] = None
    
    # Product matching
    product_matches: List[ProductMatch] = field(default_factory=list)
    
    # Estimates
    pricing_estimate: Optional[PricingEstimate] = None
    delivery_estimate: Optional[DeliveryEstimate] = None
    
    # Summary
    key_points: List[str] = field(default_factory=list)
    technical_notes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Flags for workflow
    requires_engineering_review: bool = False
    requires_custom_pricing: bool = False
    requires_sample: bool = False
    is_repeat_customer: bool = False
    
    # Confidence and metadata
    analysis_confidence: float = 0.5
    sub_agent_results: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "one_liner": self.one_liner,
            "priority": self.priority,
            "complexity": self.complexity,
            "requirements": self.requirements.to_dict() if self.requirements else None,
            "product_matches": [pm.to_dict() for pm in self.product_matches],
            "pricing_estimate": self.pricing_estimate.to_dict() if self.pricing_estimate else None,
            "delivery_estimate": self.delivery_estimate.to_dict() if self.delivery_estimate else None,
            "key_points": self.key_points,
            "technical_notes": self.technical_notes,
            "recommended_actions": self.recommended_actions,
            "requires_engineering_review": self.requires_engineering_review,
            "requires_custom_pricing": self.requires_custom_pricing,
            "requires_sample": self.requires_sample,
            "is_repeat_customer": self.is_repeat_customer,
            "analysis_confidence": self.analysis_confidence,
            "sub_agent_results": self.sub_agent_results,
            "analyzed_at": self.analyzed_at.isoformat()
        }
    
    def get_quick_view(self) -> Dict[str, Any]:
        """Get a quick-scan view for internal team dashboard."""
        return {
            "one_liner": self.one_liner,
            "priority": self.priority,
            "complexity": self.complexity,
            "products": [pm.product_code for pm in self.product_matches[:3]],
            "estimated_value": self.pricing_estimate.estimated_total if self.pricing_estimate else None,
            "delivery_days": self.delivery_estimate.estimated_days if self.delivery_estimate else None,
            "actions_needed": {
                "engineering_review": self.requires_engineering_review,
                "custom_pricing": self.requires_custom_pricing,
                "sample": self.requires_sample
            }
        }


class RequirementsAnalyzerAgent:
    """Sub-agent for extracting and validating technical requirements."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical requirements extraction specialist for industrial sealing products.
            
Analyze the quotation request and extract:
1. Industry (petrochemical, power, pharmaceutical, etc.)
2. Application (valve packing, pump sealing, gasket, etc.)
3. Operating conditions (temperature, pressure, media)
4. Certification requirements (API, ISO, FDA, etc.)
5. Special requirements (fire-safe, fugitive emission control, etc.)

Return a JSON object with the extracted information."""),
            ("human", """Quotation Request:
- Industry: {industry}
- Application: {application}
- Operating Conditions: {operating_conditions}
- Special Requirements: {special_requirements}
- Line Items: {line_items}

Extract detailed technical requirements as JSON with fields:
industry, application, operating_temperature, operating_pressure, media_handled,
shaft_speed, certifications_needed (array), special_requirements (array),
fire_safe_required (boolean), fugitive_emission_required (boolean)""")
        ])
    
    async def analyze(self, request: QuotationRequest) -> QuotationRequirements:
        """Extract requirements from quotation request."""
        try:
            line_items_str = ", ".join([
                f"{item.product_code} ({item.quantity} {item.unit})" 
                for item in request.line_items
            ])
            
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "industry": request.industry or "Not specified",
                "application": request.application or "Not specified",
                "operating_conditions": request.operating_conditions or "Not specified",
                "special_requirements": request.special_requirements or "None",
                "line_items": line_items_str or "None specified"
            })
            
            return QuotationRequirements(
                industry=result.get("industry", ""),
                application=result.get("application", ""),
                operating_temperature=result.get("operating_temperature"),
                operating_pressure=result.get("operating_pressure"),
                media_handled=result.get("media_handled"),
                shaft_speed=result.get("shaft_speed"),
                certifications_needed=result.get("certifications_needed", []),
                special_requirements=result.get("special_requirements", []),
                fire_safe_required=result.get("fire_safe_required", False),
                fugitive_emission_required=result.get("fugitive_emission_required", False)
            )
        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return QuotationRequirements(
                industry=request.industry or "",
                application=request.application or ""
            )


class ProductMatcherAgent:
    """Sub-agent for matching requirements to products."""
    
    # Product knowledge base
    PRODUCT_CATALOG = {
        "NA 701": {"name": "Pure Graphite Packing", "temp_max": 650, "applications": ["valves", "pumps"], "materials": ["graphite"]},
        "NA 702": {"name": "Graphite with PTFE Corners", "temp_max": 280, "applications": ["valves"], "materials": ["graphite", "ptfe"]},
        "NA 707": {"name": "PTFE Filament Packing", "temp_max": 280, "applications": ["pumps", "valves"], "materials": ["ptfe"]},
        "NA 710": {"name": "PTFE & Graphite Combination", "temp_max": 280, "applications": ["valves", "pumps"], "materials": ["ptfe", "graphite"]},
        "NA 715": {"name": "Pure PTFE Packing", "temp_max": 260, "applications": ["pumps", "rotary"], "materials": ["ptfe"]},
        "NA 750": {"name": "Aramid Fiber Packing", "temp_max": 290, "applications": ["valves", "pumps"], "certifications": ["API 622", "ISO 15848"]},
        "NA 752": {"name": "Aramid Braided Packing", "temp_max": 290, "applications": ["rotary", "pumps"], "materials": ["aramid"]},
    }
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a product matching specialist for industrial sealing products.

Available products: {product_catalog}

Match the customer requirements to the most suitable products."""),
            ("human", """Requirements:
- Industry: {industry}
- Application: {application}
- Temperature: {temperature}
- Pressure: {pressure}
- Media: {media}
- Certifications: {certifications}
- Special: {special}

Requested products: {requested_products}

Return JSON with 'matches' array, each with:
product_code, product_name, match_confidence (0-1), match_reasons (array), 
suggested_specifications (object with size recommendations), alternatives (array)""")
        ])
    
    async def match(self, requirements: QuotationRequirements, 
                   requested_products: List[str]) -> List[ProductMatch]:
        """Match requirements to products."""
        try:
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "product_catalog": str(self.PRODUCT_CATALOG),
                "industry": requirements.industry,
                "application": requirements.application,
                "temperature": requirements.operating_temperature or "Not specified",
                "pressure": requirements.operating_pressure or "Not specified",
                "media": requirements.media_handled or "Not specified",
                "certifications": ", ".join(requirements.certifications_needed) or "None",
                "special": ", ".join(requirements.special_requirements) or "None",
                "requested_products": ", ".join(requested_products) or "None specified"
            })
            
            matches = []
            for m in result.get("matches", []):
                matches.append(ProductMatch(
                    product_code=m.get("product_code", ""),
                    product_name=m.get("product_name", ""),
                    match_confidence=m.get("match_confidence", 0.5),
                    match_reasons=m.get("match_reasons", []),
                    suggested_specifications=m.get("suggested_specifications", {}),
                    alternatives=m.get("alternatives", [])
                ))
            return matches
            
        except Exception as e:
            logger.error(f"Product matching failed: {e}")
            # Fallback: return requested products with low confidence
            return [ProductMatch(
                product_code=p,
                product_name=self.PRODUCT_CATALOG.get(p, {}).get("name", "Unknown"),
                match_confidence=0.3,
                match_reasons=["Customer requested"],
                suggested_specifications={}
            ) for p in requested_products]


# The 4 valid material grades in the system
VALID_MATERIAL_GRADES = ["Standard", "High Purity", "Food Grade", "Nuclear Grade"]

# Mapping of common aliases/synonyms to valid grades
MATERIAL_GRADE_ALIASES = {
    "standard": "Standard",
    "regular": "Standard",
    "normal": "Standard",
    "basic": "Standard",
    "general": "Standard",
    "high purity": "High Purity",
    "high-purity": "High Purity",
    "highpurity": "High Purity",
    "hp": "High Purity",
    "premium": "High Purity",
    "top notch": "High Purity",
    "top-notch": "High Purity",
    "topnotch": "High Purity",
    "superior": "High Purity",
    "ultra": "High Purity",
    "food grade": "Food Grade",
    "food-grade": "Food Grade",
    "foodgrade": "Food Grade",
    "fda": "Food Grade",
    "food safe": "Food Grade",
    "food": "Food Grade",
    "pharmaceutical": "Food Grade",
    "nuclear grade": "Nuclear Grade",
    "nuclear-grade": "Nuclear Grade",
    "nucleargrade": "Nuclear Grade",
    "nuclear": "Nuclear Grade",
    "reactor": "Nuclear Grade",
    "radiation": "Nuclear Grade",
}


def normalize_material_grade(grade_text: str) -> str:
    """Normalize a free-text material grade to one of the 4 valid grades."""
    if not grade_text:
        return "Standard"
    
    text = grade_text.strip().lower()
    
    # Direct match
    if text in MATERIAL_GRADE_ALIASES:
        return MATERIAL_GRADE_ALIASES[text]
    
    # Check if any alias is contained in the text
    for alias, grade in sorted(MATERIAL_GRADE_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias in text:
            return grade
    
    # Check if any valid grade name is contained
    for grade in VALID_MATERIAL_GRADES:
        if grade.lower() in text:
            return grade
    
    return "Standard"  # Default


class SpecificationsRecommenderAgent:
    """
    Sub-agent for recommending additional specifications for requested products.
    
    Uses RAG to lookup product details from the knowledge base and suggests:
    - Style options (braided, die-formed, twisted, etc.)
    - Table dimensions (standard sizes available)
    - Quantity recommendations (minimum, suggested, bulk discounts)
    - Specific requirements (colour options, material grade options)
    
    NOTE: Does NOT suggest codes, certifications, or operating conditions as these
    are already collected in the Product Selection Wizard.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical specifications expert for industrial sealing products (valve packing, pump sealing, gaskets).

Based on the product information from the knowledge base and customer requirements, recommend additional specifications to complete the order.

NOTE: The customer has ALREADY specified the product code, certifications, and operating conditions in the wizard. Focus on:
1. Style options available for this product
2. Standard table dimensions (sizes) available
3. Quantity recommendations
4. Optional customizations (colour, material grade variations)

IMPORTANT RULES:
- Return ONLY valid JSON. Do NOT include any comments (no // or /* */ comments).
- material_grade_options MUST only contain values from this exact list: ["Standard", "High Purity", "Food Grade", "Nuclear Grade"]
- If the customer mentions grades like "top notch", "premium", "superior", map them to "High Purity"
- If the customer mentions "food safe", "FDA", "pharmaceutical", map them to "Food Grade"
- If the customer mentions "nuclear", "reactor", "radiation", map them to "Nuclear Grade"
- For any other or unspecified grade, use "Standard"
- The recommended_material_grade field must be one of the 4 valid grades above

You must return a JSON object with the following structure:
{{
  "products": [
    {{
      "product_code": "NA XXX",
      "product_name": "Product Name",
      "style_options": {{
        "available_styles": ["Braided", "Die-formed", "Twisted", "Interlock"],
        "recommended_style": "Braided",
        "style_notes": "Explanation of why this style is recommended"
      }},
      "table_dimensions": {{
        "standard_sizes": [
          {{"size": "6mm x 6mm", "suitable_for": "Small valves"}},
          {{"size": "10mm x 10mm", "suitable_for": "Medium valves"}},
          {{"size": "12mm x 12mm", "suitable_for": "Standard industrial"}},
          {{"size": "16mm x 16mm", "suitable_for": "Large valves"}},
          {{"size": "25mm x 25mm", "suitable_for": "Heavy duty"}}
        ],
        "custom_sizes_available": true,
        "size_recommendation": "Based on application, suggest 12mm x 12mm"
      }},
      "quantity_recommendations": {{
        "minimum_order_qty": 10,
        "suggested_quantity": 50,
        "bulk_discount_threshold": 100,
        "lead_time_standard": "5-7 days",
        "lead_time_bulk": "10-14 days"
      }},
      "customization_options": {{
        "colour_options": ["Natural/Grey", "Black", "White", "Custom"],
        "material_grade_options": ["Standard", "High Purity", "Food Grade", "Nuclear Grade"],
        "recommended_material_grade": "Standard",
        "surface_treatments": ["None", "Graphite coated", "PTFE impregnated"],
        "special_requirements_available": ["Cut rings", "Molded sets", "Custom lengths"]
      }},
      "suggestions": ["Tip 1 for this product", "Tip 2 based on application"]
    }}
  ],
  "general_notes": ["Overall recommendation 1", "Overall recommendation 2"]
}}"""),
            ("human", """Product Knowledge (from database):
{product_knowledge}

Customer Context:
- Industry: {industry}
- Application: {application}
- Customer Message/Notes: {customer_message}

Products Requested: {requested_products}

Provide style, dimensions, quantity, and customization recommendations for each product as JSON.""")
        ])
    
    async def recommend(self, 
                       request: QuotationRequest, 
                       requirements: QuotationRequirements) -> Dict[str, Any]:
        """Recommend specifications for the requested products."""
        try:
            # Get product codes from request
            product_codes = [item.product_code for item in request.line_items]
            if not product_codes:
                return {"recommendations": [], "error": "No products specified"}
            
            # Use RAG to get product knowledge from database
            product_knowledge = await self._get_product_knowledge(product_codes)
            
            # Get customer message for context
            customer_message = getattr(request, 'original_message', '') or ""
            if not customer_message:
                # Build context from line item notes
                customer_message = " ".join([
                    item.notes or "" for item in request.line_items if item.notes
                ])
            
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "product_knowledge": product_knowledge,
                "industry": requirements.industry or "Not specified",
                "application": requirements.application or "Not specified",
                "customer_message": customer_message or "No additional details provided",
                "requested_products": ", ".join(product_codes)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Specifications recommendation failed: {e}")
            # Return fallback specs in the SAME format the analyzer expects (with 'products' key)
            # so that AI-suggested line items can still be created
            fallback = self._get_fallback_specs(request)
            logger.info(f"Using fallback specs for {len(fallback.get('products', []))} products")
            return fallback
    
    async def _get_product_knowledge(self, product_codes: List[str]) -> str:
        """Retrieve product knowledge from ProductCatalogRetriever."""
        try:
            from src.data_ingestion.product_catalog_retriever import get_product_retriever
            retriever = get_product_retriever()
            
            knowledge_parts = []
            for code in product_codes[:5]:  # Limit to 5 products
                # Get detailed product information
                product_details = retriever.get_product_details(code)
                
                if product_details:
                    product_info = f"=== {code} ===\n"
                    product_info += f"Name: {product_details.get('name', 'Unknown')}\n"
                    product_info += f"Description: {product_details.get('description', 'N/A')}\n"
                    product_info += f"Category: {product_details.get('category', 'N/A')}\n"
                    product_info += f"Material: {product_details.get('material', 'N/A')}\n"
                    
                    if product_details.get('features'):
                        product_info += f"Features: {', '.join(product_details['features'][:5])}\n"
                    
                    if product_details.get('applications'):
                        product_info += f"Applications: {', '.join(product_details['applications'][:5])}\n"
                    
                    if product_details.get('certifications'):
                        product_info += f"Certifications: {', '.join(product_details['certifications'])}\n"
                    
                    if product_details.get('specifications'):
                        specs = product_details['specifications']
                        if specs.get('temperature_max'):
                            product_info += f"Max Temperature: {specs['temperature_max']}°C\n"
                        if specs.get('pressure_static'):
                            product_info += f"Max Pressure: {specs['pressure_static']} bar\n"
                        if specs.get('ph_min') and specs.get('ph_max'):
                            product_info += f"pH Range: {specs['ph_min']} - {specs['ph_max']}\n"
                    
                    if product_details.get('available_forms'):
                        product_info += f"Available Forms: {', '.join(product_details['available_forms'][:5])}\n"
                    
                    knowledge_parts.append(product_info)
                else:
                    logger.warning(f"Product {code} not found in catalog")
            
            return "\n\n".join(knowledge_parts) if knowledge_parts else "No product information found in catalog"
            
        except Exception as e:
            logger.warning(f"ProductCatalogRetriever for product knowledge failed: {e}")
            return "Product knowledge retrieval unavailable - using default recommendations"
    
    def _get_fallback_specs(self, request: QuotationRequest) -> Dict[str, Any]:
        """Provide fallback specifications when RAG/LLM fails."""
        products = []
        for item in request.line_items:
            products.append({
                "product_code": item.product_code,
                "product_name": item.product_name or "Unknown Product",
                "style_options": {
                    "available_styles": ["Braided", "Die-formed"],
                    "recommended_style": "Braided",
                    "style_notes": "Standard braided style recommended for general applications"
                },
                "table_dimensions": {
                    "standard_sizes": [
                        {"size": "6mm × 6mm", "suitable_for": "Small valves"},
                        {"size": "10mm × 10mm", "suitable_for": "Medium valves"},
                        {"size": "12mm × 12mm", "suitable_for": "Standard industrial"},
                        {"size": "16mm × 16mm", "suitable_for": "Large valves"},
                        {"size": "25mm × 25mm", "suitable_for": "Heavy duty"}
                    ],
                    "custom_sizes_available": True,
                    "size_recommendation": "Standard industrial size: 12mm × 12mm recommended"
                },
                "quantity_recommendations": {
                    "minimum_order_qty": 10,
                    "suggested_quantity": item.quantity * 2 if item.quantity else 50,
                    "bulk_discount_threshold": 100,
                    "lead_time_standard": "5-7 days",
                    "lead_time_bulk": "10-14 days"
                },
                "customization_options": {
                    "colour_options": ["Natural/Grey", "Black"],
                    "material_grade_options": ["Standard", "High Purity", "Food Grade", "Nuclear Grade"],
                    "recommended_material_grade": "Standard",
                    "surface_treatments": ["None", "Graphite coated"],
                    "special_requirements_available": ["Cut rings", "Molded sets"]
                },
                "suggestions": ["Contact technical team for detailed specifications"]
            })
        return {
            "products": products,
            "general_notes": ["Fallback recommendations - RAG retrieval was unavailable"]
        }


class PricingEstimatorAgent:
    """Sub-agent for estimating pricing."""
    
    # Base pricing (internal reference only - not exposed)
    BASE_PRICES = {
        "NA 701": 850,
        "NA 702": 750,
        "NA 707": 650,
        "NA 710": 700,
        "NA 715": 600,
        "NA 750": 950,
        "NA 752": 900,
    }
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
    
    async def estimate(self, line_items: List[QuotationLineItem], 
                       requirements: QuotationRequirements) -> PricingEstimate:
        """Estimate pricing for the quotation."""
        try:
            total_estimate = 0.0
            pricing_factors = []
            
            for item in line_items:
                base_price = self.BASE_PRICES.get(item.product_code, 500)
                
                # Adjust for size (larger = more expensive)
                size_multiplier = 1.0
                if item.size_od and item.size_od > 50:
                    size_multiplier = 1.0 + (item.size_od - 50) / 100
                    pricing_factors.append(f"Size adjustment for {item.product_code}")
                
                # Adjust for special requirements
                special_multiplier = 1.0
                if requirements.fire_safe_required:
                    special_multiplier += 0.15
                    pricing_factors.append("Fire-safe premium")
                if requirements.fugitive_emission_required:
                    special_multiplier += 0.10
                    pricing_factors.append("Fugitive emission certification")
                
                unit_price = base_price * size_multiplier * special_multiplier
                total_estimate += unit_price * item.quantity
            
            # Volume discount
            discount_percent = 0.0
            if total_estimate > 100000:
                discount_percent = 10.0
                pricing_factors.append("Volume discount (10%)")
            elif total_estimate > 50000:
                discount_percent = 5.0
                pricing_factors.append("Volume discount (5%)")
            
            confidence = "medium"
            if len(line_items) <= 3 and not requirements.special_requirements:
                confidence = "high"
            elif requirements.special_requirements or requirements.certifications_needed:
                confidence = "low"
            
            return PricingEstimate(
                estimated_unit_price=total_estimate / sum(i.quantity for i in line_items) if line_items else None,
                estimated_total=total_estimate * (1 - discount_percent/100),
                price_confidence=confidence,
                pricing_factors=pricing_factors,
                discount_applicable=discount_percent > 0,
                volume_discount_percent=discount_percent
            )
            
        except Exception as e:
            logger.error(f"Pricing estimation failed: {e}")
            return PricingEstimate(price_confidence="low")


class DeliveryEstimatorAgent:
    """Sub-agent for estimating delivery timeline."""
    
    # Standard lead times
    STANDARD_LEAD_TIMES = {
        "NA 701": 7,
        "NA 702": 10,
        "NA 707": 7,
        "NA 710": 10,
        "NA 715": 7,
        "NA 750": 14,
        "NA 752": 14,
    }
    
    async def estimate(self, line_items: List[QuotationLineItem], 
                       requirements: QuotationRequirements) -> DeliveryEstimate:
        """Estimate delivery timeline."""
        try:
            # Get max lead time from products
            max_lead_time = 14
            factors = []
            
            for item in line_items:
                lead_time = self.STANDARD_LEAD_TIMES.get(item.product_code, 14)
                
                # Adjust for quantity
                if item.quantity > 100:
                    lead_time += 7
                    factors.append(f"High quantity for {item.product_code}")
                
                # Adjust for custom sizes
                if item.size_od and (item.size_od < 10 or item.size_od > 100):
                    lead_time += 5
                    factors.append(f"Non-standard size for {item.product_code}")
                
                max_lead_time = max(max_lead_time, lead_time)
            
            # Adjust for certifications
            if requirements.certifications_needed:
                max_lead_time += 3
                factors.append("Certification documentation")
            
            confidence = "high" if max_lead_time <= 14 else "medium"
            if max_lead_time > 21:
                confidence = "low"
            
            return DeliveryEstimate(
                estimated_days=max_lead_time,
                delivery_confidence=confidence,
                factors=factors,
                expedite_possible=max_lead_time > 7,
                expedite_premium_percent=15.0 if max_lead_time > 14 else 10.0
            )
            
        except Exception as e:
            logger.error(f"Delivery estimation failed: {e}")
            return DeliveryEstimate()


class SummaryGeneratorAgent:
    """Sub-agent for generating structured summary."""
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quotation summary specialist. Create concise, actionable summaries for the sales team.

Generate:
1. A one-liner summary (max 100 chars)
2. Priority (low/medium/high/urgent)
3. Complexity (simple/standard/complex/custom)
4. Key points for quick review
5. Technical notes for engineering
6. Recommended actions"""),
            ("human", """Quotation for: {customer}
Industry: {industry}
Application: {application}
Products: {products}
Quantity: {total_quantity}
Estimated Value: {estimated_value}
Delivery: {delivery_days} days
Special Requirements: {special}

Return JSON with: one_liner, priority, complexity, key_points (array), 
technical_notes (array), recommended_actions (array),
requires_engineering_review (boolean), requires_custom_pricing (boolean),
requires_sample (boolean)""")
        ])
    
    async def summarize(self, request: QuotationRequest, 
                        requirements: QuotationRequirements,
                        pricing: PricingEstimate,
                        delivery: DeliveryEstimate) -> Dict[str, Any]:
        """Generate structured summary."""
        try:
            products_str = ", ".join([item.product_code for item in request.line_items])
            total_qty = sum(item.quantity for item in request.line_items)
            
            chain = self.prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "customer": f"{request.customer.name} ({request.customer.company})" if request.customer else "Unknown",
                "industry": requirements.industry,
                "application": requirements.application,
                "products": products_str,
                "total_quantity": total_qty,
                "estimated_value": f"₹{pricing.estimated_total:,.0f}" if pricing.estimated_total else "TBD",
                "delivery_days": delivery.estimated_days,
                "special": ", ".join(requirements.special_requirements) if requirements.special_requirements else "None"
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                "one_liner": f"Quotation request for {len(request.line_items)} items",
                "priority": "medium",
                "complexity": "standard",
                "key_points": [f"Customer: {request.customer.name if request.customer else 'Unknown'}"],
                "technical_notes": [],
                "recommended_actions": ["Review and prepare quotation"],
                "requires_engineering_review": False,
                "requires_custom_pricing": False,
                "requires_sample": False
            }


class QuotationAnalyzer:
    """
    Multi-agent quotation analyzer.
    
    Orchestrates specialized sub-agents:
    1. RequirementsAnalyzerAgent - Technical requirements extraction
    2. ProductMatcherAgent - Product matching and recommendations
    3. PricingEstimatorAgent - Price estimation
    4. DeliveryEstimatorAgent - Delivery timeline estimation
    5. SummaryGeneratorAgent - Structured summary generation
    6. SpecificationsRecommenderAgent - Product specification recommendations (uses RAG)
    """
    
    def __init__(self):
        from src.config.settings import get_llm
        self.llm = get_llm(temperature=0.3)
        
        # Initialize sub-agents
        self.requirements_agent = RequirementsAnalyzerAgent(self.llm)
        self.product_agent = ProductMatcherAgent(self.llm)
        self.pricing_agent = PricingEstimatorAgent(self.llm)
        self.delivery_agent = DeliveryEstimatorAgent()
        self.summary_agent = SummaryGeneratorAgent(self.llm)
        self.specs_agent = SpecificationsRecommenderAgent(self.llm)  # NEW: RAG-based spec recommendations
    
    async def analyze(self, request: QuotationRequest) -> QuotationAnalysis:
        """
        Perform comprehensive multi-agent analysis of a quotation request.
        
        Execution flow:
        1. For generic requests: Use RAG to find relevant products
        2. Extract requirements (parallel)
        3. Match products based on requirements
        4. Get specification recommendations (RAG-based)
        5. Estimate pricing and delivery (parallel)
        6. Generate summary
        """
        logger.info(f"Starting multi-agent analysis for quotation {request.id}")
        
        try:
            # For generic quotations with original_message, use RAG to find products
            rag_suggested_products = []
            original_msg = getattr(request, 'original_message', None) or ""
            
            if original_msg and not request.line_items:
                logger.info(f"Generic quotation detected - using ProductCatalogRetriever to find relevant products")
                try:
                    # Use ProductCatalogRetriever - a proper RAG-like system with product catalog
                    from src.data_ingestion.product_catalog_retriever import get_product_retriever
                    retriever = get_product_retriever()
                    
                    # Parse keywords from the message to find relevant products
                    msg_lower = original_msg.lower()
                    
                    # Determine industry from message
                    industry = None
                    industry_keywords = {
                        'power': 'power_plant', 'power plant': 'power_plant',
                        'refinery': 'refinery', 'oil': 'refinery', 'gas': 'refinery',
                        'petrochemical': 'petrochemical', 'petroleum': 'petrochemical',
                        'chemical': 'chemical', 'acid': 'chemical',
                        'pulp': 'pulp_paper', 'paper': 'pulp_paper',
                        'sugar': 'sugar', 'mining': 'mining',
                        'food': 'food', 'pharmaceutical': 'pharmaceutical',
                        'nuclear': 'power_plant',
                    }
                    for kw, ind in industry_keywords.items():
                        if kw in msg_lower:
                            industry = ind
                            break
                    
                    # Determine application type from message
                    application = None
                    app_keywords = {
                        'pump': 'pump', 'centrifugal': 'centrifugal_pump', 'plunger': 'plunger_pump',
                        'valve': 'valve', 'control valve': 'control_valve', 'block valve': 'block_valve',
                        'agitator': 'agitator', 'mixer': 'mixer',
                        'compressor': 'compressor', 'blower': 'blower',
                        'reactor': 'reactor', 'flange': 'flange',
                        'boiler': 'boiler', 'turbine': 'turbine',
                        'gasket': 'flange', 'seal': 'pump', 'packing': 'valve',
                    }
                    for kw, app in app_keywords.items():
                        if kw in msg_lower:
                            application = app
                            break
                    
                    # Extract temperature if mentioned
                    import re
                    temp_match = re.search(r'(\d+)\s*(?:°?C|celsius|degree)', msg_lower)
                    temperature = float(temp_match.group(1)) if temp_match else None
                    
                    # Extract pressure if mentioned
                    pressure_match = re.search(r'(\d+)\s*(?:bar|psi|mpa)', msg_lower)
                    pressure = float(pressure_match.group(1)) if pressure_match else None
                    
                    # Determine material preference
                    material = None
                    if 'graphite' in msg_lower:
                        material = 'graphite'
                    elif 'ptfe' in msg_lower or 'teflon' in msg_lower:
                        material = 'ptfe'
                    elif 'aramid' in msg_lower:
                        material = 'aramid'
                    elif 'carbon' in msg_lower:
                        material = 'carbon'
                    
                    # Use ProductCatalogRetriever to find matching products
                    matches = retriever.find_products(
                        industry=industry,
                        application=application,
                        operating_temp=temperature,
                        operating_pressure=pressure,
                        material_preference=material,
                        limit=5
                    )
                    
                    if matches:
                        rag_suggested_products = [m.product.code for m in matches]
                        logger.info(f"ProductCatalogRetriever found products: {rag_suggested_products}")
                    else:
                        # If no matches found with parameters, do a general search
                        # Just get some popular products based on any keywords
                        all_matches = []
                        if industry:
                            all_matches = retriever.find_products(industry=industry, limit=3)
                        if not all_matches and application:
                            all_matches = retriever.find_products(application=application, limit=3)
                        
                        if all_matches:
                            rag_suggested_products = [m.product.code for m in all_matches]
                            logger.info(f"ProductCatalogRetriever fallback found products: {rag_suggested_products}")
                    
                except Exception as rag_error:
                    logger.warning(f"ProductCatalogRetriever failed: {rag_error}")
                    # Fallback: Extract products from original message using regex
                    import re
                    codes = re.findall(r'NA\s*\d{3}', original_msg)
                    rag_suggested_products = list(set(codes))
            
            # Phase 1: Extract requirements
            requirements = await self.requirements_agent.analyze(request)
            
            # Phase 2: Match products
            requested_products = [item.product_code for item in request.line_items]
            
            # Include RAG-suggested products for generic quotations
            if rag_suggested_products and not requested_products:
                requested_products = rag_suggested_products
                
            product_matches = await self.product_agent.match(requirements, requested_products)
            
            # Phase 3: Get specification recommendations AND pricing/delivery estimation (parallel)
            specs_task = self.specs_agent.recommend(request, requirements)
            pricing_task = self.pricing_agent.estimate(request.line_items, requirements)
            delivery_task = self.delivery_agent.estimate(request.line_items, requirements)
            
            specs_recommendations, pricing, delivery = await asyncio.gather(
                specs_task, pricing_task, delivery_task
            )
            
            # Create AI-suggested line items from specs_recommendations
            # This works for both:
            # 1. Generic quotations with RAG-suggested products
            # 2. Generic quotations with product_code extracted from request
            # HYBRID APPROACH: Use AI specs where available, defaults for missing fields
            ai_suggested_line_items = []
            if specs_recommendations and getattr(request, 'requires_ai_processing', False):
                products_specs = specs_recommendations.get('products', [])
                
                # Also build a fallback for each product in case some specs are missing
                fallback = self.specs_agent._get_fallback_specs(request)
                fallback_by_code = {p['product_code']: p for p in fallback.get('products', [])}
                
                for product_spec in products_specs:
                    product_code = product_spec.get('product_code', '')
                    if not product_code:
                        continue
                    
                    # Get the fallback for this product (if available)
                    fb = fallback_by_code.get(product_code, {})
                    fb_style = fb.get('style_options', {})
                    fb_dims = fb.get('table_dimensions', {})
                    fb_qty = fb.get('quantity_recommendations', {})
                    fb_custom = fb.get('customization_options', {})
                    
                    # Get recommended values from spec suggestions, 
                    # falling back to defaults for any missing field
                    style_opts = product_spec.get('style_options', fb_style)
                    table_dims = product_spec.get('table_dimensions', fb_dims)
                    qty_recs = product_spec.get('quantity_recommendations', fb_qty)
                    custom_opts = product_spec.get('customization_options', fb_custom)
                    
                    # Extract recommended size from size_recommendation string
                    size_rec = table_dims.get('size_recommendation', fb_dims.get('size_recommendation', ''))
                    import re
                    size_match = re.search(r'(\d+mm\s*[×x]\s*\d+mm)', size_rec)
                    if size_match:
                        size_value = size_match.group(1)
                    else:
                        # Fallback: use first standard size or default
                        std_sizes = table_dims.get('standard_sizes', fb_dims.get('standard_sizes', []))
                        size_value = std_sizes[0].get('size', '12mm x 12mm') if std_sizes else '12mm x 12mm'
                    
                    # Determine material grade: use recommended_material_grade if available,
                    # then first from options, then normalize any free-text grade
                    raw_grade = (
                        custom_opts.get('recommended_material_grade') or
                        (custom_opts.get('material_grade_options', [])[0] if custom_opts.get('material_grade_options') else None) or
                        'Standard'
                    )
                    material_grade = normalize_material_grade(raw_grade)
                    
                    # Get style with fallback
                    style = style_opts.get('recommended_style') or fb_style.get('recommended_style', 'Braided')
                    
                    # Get colour with fallback
                    colour = (
                        custom_opts.get('colour_options', fb_custom.get('colour_options', ['Natural/Grey']))[0]
                    )
                    
                    # Get quantity with fallback
                    quantity = qty_recs.get('suggested_quantity') or fb_qty.get('suggested_quantity', 50)
                    
                    # Derive material_code from product catalogue
                    catalog_entry = ProductMatcherAgent.PRODUCT_CATALOG.get(product_code, {})
                    catalog_materials = catalog_entry.get('materials', [])
                    material_code = ', '.join(m.upper() for m in catalog_materials) if catalog_materials else ''
                    
                    # Parse size_value into OD/ID/TH numeric fields
                    size_od_val = None
                    size_id_val = None
                    size_th_val = None
                    if size_value:
                        # Parse sizes like "12mm x 12mm", "12mm × 12mm × 6mm", "12 x 12 x 6"
                        size_parts = re.findall(r'(\d+(?:\.\d+)?)', size_value)
                        if len(size_parts) >= 1:
                            size_od_val = float(size_parts[0])
                        if len(size_parts) >= 2:
                            size_id_val = float(size_parts[1])
                        if len(size_parts) >= 3:
                            size_th_val = float(size_parts[2])
                    
                    # Create AI-suggested line item with filled specifications
                    from src.quotation.models import QuotationLineItem
                    ai_item = QuotationLineItem(
                        product_code=product_code,
                        product_name=product_spec.get('product_name', product_code),
                        size=size_value,
                        size_od=size_od_val,
                        size_id=size_id_val,
                        size_th=size_th_val,
                        dimension_unit='mm',
                        style=style,
                        material_grade=material_grade,
                        material_code=material_code,
                        colour=colour,
                        quantity=quantity,
                        unit='Nos.',
                        dimensions={"standard_sizes": table_dims.get('standard_sizes', fb_dims.get('standard_sizes', []))},
                        specific_requirements="; ".join(product_spec.get('suggestions', fb.get('suggestions', []))),
                        is_ai_suggested=True,
                        ai_confidence=0.85,
                        notes=f"AI Suggested: Style={style}, Size={size_value}, Grade={material_grade}"
                    )
                    ai_suggested_line_items.append(ai_item)
                    logger.info(f"Created AI-suggested line item for {product_code}: size={size_value}, style={style}, grade={material_grade}")
            
            # Phase 4: Generate summary
            summary = await self.summary_agent.summarize(request, requirements, pricing, delivery)
            
            # For generic quotations with AI suggestions, enhance the one-liner
            one_liner = summary.get("one_liner", "")
            if ai_suggested_line_items:
                # Build a more informative one-liner with AI suggestions
                ai_item = ai_suggested_line_items[0]  # Primary product
                customer_name = request.customer.name if request.customer else "Customer"
                total_qty = sum(item.quantity for item in ai_suggested_line_items)
                
                one_liner = f"{ai_item.product_code} quotation for {customer_name}. AI suggests: {ai_item.style} style, {ai_item.size}, {total_qty} units. Est. value ₹{pricing.estimated_total:,.0f}, delivery {delivery.estimated_days} days."
            elif rag_suggested_products and not request.line_items:
                if one_liner:
                    one_liner = f"{one_liner}. Suggested products: {', '.join(rag_suggested_products)}"
                else:
                    one_liner = f"Customer inquiry for: {original_msg[:50]}... Suggested: {', '.join(rag_suggested_products)}"
            
            # Build final analysis
            analysis = QuotationAnalysis(
                one_liner=one_liner,
                priority=summary.get("priority", "medium"),
                complexity=summary.get("complexity", "standard"),
                requirements=requirements,
                product_matches=product_matches,
                pricing_estimate=pricing,
                delivery_estimate=delivery,
                key_points=summary.get("key_points", []),
                technical_notes=summary.get("technical_notes", []),
                recommended_actions=summary.get("recommended_actions", []),
                requires_engineering_review=summary.get("requires_engineering_review", False),
                requires_custom_pricing=summary.get("requires_custom_pricing", False),
                requires_sample=summary.get("requires_sample", False),
                analysis_confidence=self._calculate_confidence(pricing, delivery, product_matches),
                sub_agent_results={
                    "requirements": requirements.to_dict(),
                    "products": [pm.to_dict() for pm in product_matches],
                    "pricing": pricing.to_dict(),
                    "delivery": delivery.to_dict(),
                    "rag_suggested_products": rag_suggested_products,
                    "specifications_recommendations": specs_recommendations,
                    "ai_suggested_line_items": [item.to_dict() for item in ai_suggested_line_items]  # AI-filled line items for generic quotations
                }
            )
            
            logger.info(f"Quotation analysis complete: {analysis.one_liner}")
            return analysis
            
        except Exception as e:
            logger.error(f"Quotation analysis failed: {e}")
            return QuotationAnalysis(
                one_liner="Analysis failed - manual review required",
                priority="high",
                complexity="unknown",
                recommended_actions=["Manual review required", f"Error: {str(e)}"]
            )
    
    def _calculate_confidence(self, pricing: PricingEstimate, 
                             delivery: DeliveryEstimate,
                             matches: List[ProductMatch]) -> float:
        """Calculate overall analysis confidence."""
        confidence = 0.5
        
        if pricing.price_confidence == "high":
            confidence += 0.15
        elif pricing.price_confidence == "low":
            confidence -= 0.1
        
        if delivery.delivery_confidence == "high":
            confidence += 0.15
        elif delivery.delivery_confidence == "low":
            confidence -= 0.1
        
        if matches:
            avg_match_confidence = sum(m.match_confidence for m in matches) / len(matches)
            confidence += (avg_match_confidence - 0.5) * 0.4
        
        return max(0.1, min(0.95, confidence))


# Singleton instance
_analyzer: Optional[QuotationAnalyzer] = None


def get_quotation_analyzer() -> QuotationAnalyzer:
    """Get or create the quotation analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = QuotationAnalyzer()
    return _analyzer
