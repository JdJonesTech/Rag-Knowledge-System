"""
Grounding Validator - Programmatic Anti-Hallucination Layer

Instead of relying on system prompt instructions (which LLMs routinely ignore),
this module provides architectural enforcement against hallucination:

1. PRE-GENERATION: Context Sufficiency Check
   - If no relevant product data exists in context, short-circuit the LLM
     and return a structured "I don't have this info" response
   
2. POST-GENERATION: Response Grounding Verification
   - Extract any numerical specs (temperature, pressure, pH, speed) from the
     LLM's response
   - Cross-check every extracted spec against the provided context and product
     catalog data
   - Flag/replace any ungrounded specifications with warnings

3. PRODUCT CODE VERIFICATION
   - Verify that any product codes mentioned in the response actually exist
     in the catalog
   - Prevent the LLM from inventing fake product codes

This is NOT a prompt-based solution — it's an output filter that runs after
the LLM generates its response, ensuring factual accuracy through validation.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding validation."""
    is_grounded: bool
    original_response: str
    validated_response: str
    ungrounded_claims: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0 or len(self.ungrounded_claims) > 0


class GroundingValidator:
    """
    Programmatic grounding enforcement for RAG responses.
    
    This catches LLM hallucination by:
    1. Checking if the context actually contains relevant product data
    2. Extracting specs from the LLM response and cross-checking against context
    3. Verifying product codes against the real catalog
    """
    
    # Patterns to extract specs from LLM responses
    SPEC_PATTERNS = {
        "temperature": [
            re.compile(r'(-?\d+)\s*°?\s*C?\s*to\s*(\+?\d+)\s*°?\s*C', re.IGNORECASE),
            re.compile(r'temperature[:\s]+(-?\d+)\s*°?\s*C?\s*to\s*(\+?\d+)', re.IGNORECASE),
            re.compile(r'temp(?:erature)?\s+range[:\s]+(-?\d+)\s*to\s*(\+?\d+)', re.IGNORECASE),
        ],
        "pressure": [
            re.compile(r'(\d+(?:\.\d+)?)\s*bar', re.IGNORECASE),
            re.compile(r'pressure[:\s]+(\d+(?:\.\d+)?)\s*bar', re.IGNORECASE),
        ],
        "ph": [
            re.compile(r'pH[:\s]+(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)', re.IGNORECASE),
        ],
        "shaft_speed": [
            re.compile(r'(\d+(?:\.\d+)?)\s*m/sec', re.IGNORECASE),
        ],
    }
    
    # Pattern to extract product codes from text
    PRODUCT_CODE_PATTERN = re.compile(r'\bNA\s*\d+[A-Z]*\b', re.IGNORECASE)
    
    def __init__(self):
        self._catalog = None
    
    @property
    def catalog(self):
        """Lazy load product catalog."""
        if self._catalog is None:
            try:
                from src.data_ingestion.product_catalog_loader import get_product_catalog
                self._catalog = get_product_catalog()
            except Exception as e:
                logger.warning(f"Could not load product catalog for grounding: {e}")
        return self._catalog
    
    def check_context_sufficiency(
        self,
        context: str,
        product_matches: list,
        query: str
    ) -> Tuple[bool, Optional[str]]:
        """
        PRE-GENERATION CHECK: Is there enough context to answer this query?
        
        Returns:
            Tuple of (is_sufficient, fallback_response_if_not)
        """
        # Check if query is about a specific product
        code_match = re.search(r'\bNA\s*[-]?\s*(\d+[A-Z]*)\b', query, re.IGNORECASE)
        
        if code_match:
            product_code = f"NA {code_match.group(1).upper()}"
            
            # Check if we have product data from catalog
            has_catalog_data = any(
                hasattr(m, 'product') and m.product.code == product_code
                for m in (product_matches or [])
            )
            
            # Check if context mentions this product code
            code_in_context = product_code.lower() in context.lower()
            
            if not has_catalog_data and not code_in_context:
                return False, (
                    f"I don't have detailed specifications for {product_code} in my current "
                    f"knowledge base. To get accurate product specifications, please:\n\n"
                    f"- Contact our sales team at sales@jdjones.com\n"
                    f"- Visit our product page at www.jdjones.com\n"
                    f"- Request a product data sheet\n\n"
                    f"I want to make sure you get the correct specifications rather than "
                    f"providing potentially inaccurate information."
                )
        
        # For general queries, check minimum context quality
        no_info_phrases = [
            "no specific information available",
            "no relevant information found",
            "no highly relevant information found",
        ]
        context_is_empty = any(phrase in context.lower() for phrase in no_info_phrases)
        
        if context_is_empty and not product_matches:
            return False, None  # Insufficient but no specific product asked; let LLM handle
        
        return True, None
    
    def validate_response(
        self,
        response: str,
        context: str,
        product_matches: list = None,
        query: str = ""
    ) -> GroundingResult:
        """
        POST-GENERATION CHECK: Verify the LLM's response is grounded in context.
        
        Extracts specs from the response and checks if they appear in the
        provided context or product catalog data.
        """
        result = GroundingResult(
            is_grounded=True,
            original_response=response,
            validated_response=response,
        )
        
        # Step 1: Extract product codes from response
        response_codes = set()
        for match in self.PRODUCT_CODE_PATTERN.finditer(response):
            code = re.sub(r'\s+', ' ', match.group().upper()).strip()
            code = re.sub(r'^NA\s*', 'NA ', code)
            response_codes.add(code)
        
        # Step 2: Verify product codes exist in catalog
        if self.catalog and response_codes:
            for code in response_codes:
                product = self.catalog.get_product_by_code(code)
                if not product:
                    result.warnings.append(
                        f"Product code {code} not found in catalog"
                    )
                    result.ungrounded_claims.append({
                        "type": "unknown_product_code",
                        "code": code,
                    })
        
        # Step 3: Extract specs from response and verify against context
        response_specs = self._extract_specs(response)
        context_specs = self._extract_specs(context)
        
        # Build catalog spec reference for mentioned products
        catalog_specs = {}
        if self.catalog and product_matches:
            for match in product_matches:
                if hasattr(match, 'product') and match.product.specs:
                    p = match.product
                    s = p.specs
                    catalog_specs[p.code] = {
                        "temperature": (s.temperature_min, s.temperature_max),
                        "pressure_static": s.pressure_static,
                        "pressure_rotary": s.pressure_rotary,
                        "ph": (s.ph_min, s.ph_max),
                    }
        
        # Step 4: Check if response specs are grounded
        for spec_type, response_values in response_specs.items():
            context_values = context_specs.get(spec_type, [])
            
            for resp_val in response_values:
                is_in_context = self._spec_value_in_context(
                    resp_val, context_values
                )
                is_in_catalog = self._spec_value_in_catalog(
                    resp_val, spec_type, catalog_specs
                )
                
                if not is_in_context and not is_in_catalog:
                    result.is_grounded = False
                    result.ungrounded_claims.append({
                        "type": f"ungrounded_{spec_type}",
                        "value": resp_val,
                        "in_context": False,
                        "in_catalog": False,
                    })
                    result.warnings.append(
                        f"Specification '{spec_type}: {resp_val}' not found in "
                        f"provided context or product catalog"
                    )
        
        # Step 5: If ungrounded claims found, add disclaimer to response
        if result.ungrounded_claims:
            disclaimer = (
                "\n\n---\n"
                "*Note: Some specifications in this response could not be verified "
                "against our product database. Please contact our technical team at "
                "sales@jdjones.com for confirmed specifications.*"
            )
            
            # For severe cases (multiple ungrounded specs), replace the response
            ungrounded_spec_count = sum(
                1 for c in result.ungrounded_claims 
                if c["type"].startswith("ungrounded_")
            )
            
            if ungrounded_spec_count >= 3:
                # Too many ungrounded specs — likely hallucinated entirely
                code_str = ", ".join(response_codes) if response_codes else "the requested product"
                result.validated_response = (
                    f"I have limited information about {code_str} in my knowledge base. "
                    f"To ensure you receive accurate specifications, I recommend:\n\n"
                    f"- Contacting our sales team at sales@jdjones.com\n"
                    f"- Visiting www.jdjones.com for detailed product data sheets\n"
                    f"- Requesting a technical consultation\n\n"
                    f"I'd rather connect you with the right team than provide "
                    f"potentially incorrect specifications."
                )
                logger.warning(
                    f"Grounding check: replaced response with {ungrounded_spec_count} "
                    f"ungrounded specs for query: {query[:100]}"
                )
            else:
                result.validated_response = response + disclaimer
                logger.info(
                    f"Grounding check: added disclaimer for {ungrounded_spec_count} "
                    f"ungrounded specs"
                )
        
        return result
    
    def _extract_specs(self, text: str) -> Dict[str, List[str]]:
        """Extract all specification values from text."""
        specs = {}
        for spec_type, patterns in self.SPEC_PATTERNS.items():
            values = []
            for pattern in patterns:
                for match in pattern.finditer(text):
                    values.append(match.group(0))
            if values:
                specs[spec_type] = values
        return specs
    
    def _spec_value_in_context(
        self, value: str, context_values: List[str]
    ) -> bool:
        """Check if a spec value appears in the context values."""
        if not context_values:
            return False
        
        # Extract numbers from the value
        value_numbers = set(re.findall(r'-?\d+(?:\.\d+)?', value))
        
        for ctx_val in context_values:
            ctx_numbers = set(re.findall(r'-?\d+(?:\.\d+)?', ctx_val))
            # If all numbers in the response value appear in a context value
            if value_numbers and value_numbers.issubset(ctx_numbers):
                return True
        
        return False
    
    def _spec_value_in_catalog(
        self, value: str, spec_type: str, catalog_specs: Dict
    ) -> bool:
        """Check if a spec value matches catalog data."""
        if not catalog_specs:
            return False
        
        value_numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', value)]
        if not value_numbers:
            return False
        
        for code, specs in catalog_specs.items():
            if spec_type == "temperature":
                temp_range = specs.get("temperature")
                if temp_range and temp_range[0] is not None:
                    for num in value_numbers:
                        if abs(num - temp_range[0]) < 1 or abs(num - temp_range[1]) < 1:
                            return True
            
            elif spec_type == "pressure":
                for key in ["pressure_static", "pressure_rotary"]:
                    catalog_pressure = specs.get(key)
                    if catalog_pressure:
                        for num in value_numbers:
                            if abs(num - catalog_pressure) < 1:
                                return True
            
            elif spec_type == "ph":
                ph_range = specs.get("ph")
                if ph_range and ph_range[0] is not None:
                    for num in value_numbers:
                        if abs(num - ph_range[0]) < 0.5 or abs(num - ph_range[1]) < 0.5:
                            return True
        
        return False


# Singleton instance
_validator_instance: Optional[GroundingValidator] = None


def get_grounding_validator() -> GroundingValidator:
    """Get or create the grounding validator singleton."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = GroundingValidator()
    return _validator_instance
