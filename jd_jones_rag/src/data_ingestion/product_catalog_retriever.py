"""
Product Catalog Retriever
Provides intelligent product search for the RAG system.
Bridges Product Selection Agent with actual product data.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.data_ingestion.product_catalog_loader import (
    get_product_catalog, Product, ProductCatalogLoader
)

logger = logging.getLogger(__name__)


class MatchConfidence(Enum):
    """Confidence level of product match."""
    EXACT = "exact"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ProductMatch:
    """A matched product with relevance score."""
    product: Product
    confidence: MatchConfidence
    score: float
    match_reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_code": self.product.code,
            "product_name": self.product.name,
            "category": self.product.category,
            "description": self.product.description,
            "material": self.product.material,
            "confidence": self.confidence.value,
            "score": self.score,
            "match_reasons": self.match_reasons,
            "specifications": self.product.specs.to_dict() if self.product.specs else {},
            "applications": self.product.applications,
            "certifications": self.product.certifications,
            "service_media": self.product.service_media,
            "source_url": self.product.source_url,
        }


class ProductCatalogRetriever:
    """
    Intelligent product retriever for the RAG system.
    
    Matches customer requirements to products based on:
    - Operating conditions (temperature, pressure, pH)
    - Application type (pump, valve, agitator, etc.)
    - Industry
    - Material preferences
    - Certification requirements
    """
    
    # Application to product recommendations
    APPLICATION_RECOMMENDATIONS = {
        "pump": ["NA 758H", "NA 802", "NA 719", "NA 730", "NA 757"],
        "centrifugal_pump": ["NA 802", "NA 758H", "NA 740", "NA 747"],
        "plunger_pump": ["NA 781PP", "NA 745PP", "NA 757"],
        "valve": ["NA 701", "NA 707", "NA 715", "NA 752", "NA 781G"],
        "block_valve": ["NA 715", "NA 701"],
        "control_valve": ["NA 781G", "NA 781A", "NA 701", "NA 707"],
        "agitator": ["NA 758H", "NA 802", "NA 740", "NA 741"],
        "mixer": ["NA 758H", "NA 802", "NA 730"],
        "compressor": ["NA 752", "NA 751", "NA 754"],
        "blower": ["NA 730", "NA 733"],
        "reactor": ["NA 701T", "NA 706"],
        "flange": ["NA 610", "NA 706", "NA 709"],
        "boiler": ["NA 550", "NA 555", "NA 713"],
        "turbine": ["NA 711", "NA 710"],
    }
    
    # Industry-specific recommendations
    INDUSTRY_RECOMMENDATIONS = {
        "power_plant": ["NA 701", "NA 715", "NA 758H", "NA 802", "NA 705"],
        "refinery": ["NA 715", "NA 701", "NA 752", "NA 707"],
        "petrochemical": ["NA 715", "NA 752", "NA 701", "NA 758H"],
        "chemical": ["NA 758H", "NA 757", "NA 737", "NA 802"],
        "pulp_paper": ["NA 783", "NA 758H", "NA 802", "NA 740"],
        "sugar": ["NA 747", "NA 748", "NA 740"],
        "mining": ["NA 737", "NA 757", "NA 730", "NA 758CCC"],
        "food": ["NA 740 MA", "NA 781", "NA 730"],
        "pharmaceutical": ["NA 781", "NA 757", "NA 730"],
    }
    
    # Certification mappings
    CERTIFICATION_PRODUCTS = {
        "API 622": ["NA 715", "NA B-3 + 707"],
        "API 589": ["NA 715"],
        "API 607": ["NA 715"],
        "food_grade": ["NA 740 MA", "NA 730", "NA 781"],
        "low_emission": ["NA 715", "NA B-3 + 707", "NA SP-1"],
    }
    
    def __init__(self, catalog: Optional[ProductCatalogLoader] = None):
        self.catalog = catalog or get_product_catalog()
    
    def find_products(
        self,
        industry: Optional[str] = None,
        application: Optional[str] = None,
        equipment_type: Optional[str] = None,
        media: Optional[str] = None,
        operating_temp: Optional[float] = None,
        operating_pressure: Optional[float] = None,
        shaft_speed: Optional[float] = None,
        ph_value: Optional[float] = None,
        certifications: Optional[List[str]] = None,
        material_preference: Optional[str] = None,
        limit: int = 5,
    ) -> List[ProductMatch]:
        """
        Find matching products based on customer requirements.
        
        Args:
            industry: Industry sector (e.g., "power_plant", "refinery")
            application: Application type (e.g., "pump", "valve")
            equipment_type: Specific equipment (e.g., "centrifugal_pump")
            media: The media being sealed (e.g., "steam", "acid")
            operating_temp: Operating temperature in Celsius
            operating_pressure: Operating pressure in bar
            shaft_speed: Shaft speed in m/sec
            ph_value: pH of the media
            certifications: Required certifications
            material_preference: Preferred material
            limit: Maximum number of results
            
        Returns:
            List of ProductMatch objects sorted by relevance
        """
        matches: List[ProductMatch] = []
        
        for product in self.catalog.products.values():
            score = 0.0
            reasons = []
            
            # Check temperature compatibility
            if operating_temp is not None and product.specs:
                if product.specs.temperature_min is not None and product.specs.temperature_max is not None:
                    if product.specs.temperature_min <= operating_temp <= product.specs.temperature_max:
                        score += 25
                        reasons.append(f"Temperature compatible ({product.specs.temperature_min}°C to {product.specs.temperature_max}°C)")
                    else:
                        continue  # Temperature out of range
            
            # Check pressure compatibility
            if operating_pressure is not None and product.specs:
                max_pressure = max(
                    product.specs.pressure_static or 0,
                    product.specs.pressure_rotary or 0,
                    product.specs.pressure_reciprocating or 0
                )
                if max_pressure >= operating_pressure:
                    score += 20
                    reasons.append(f"Pressure compatible (up to {max_pressure} bar)")
                else:
                    score -= 10  # Reduce score but don't exclude
            
            # Check pH compatibility
            if ph_value is not None and product.specs:
                if product.specs.ph_min is not None and product.specs.ph_max is not None:
                    if product.specs.ph_min <= ph_value <= product.specs.ph_max:
                        score += 15
                        reasons.append(f"pH compatible ({product.specs.ph_min} to {product.specs.ph_max})")
            
            # Check industry match
            if industry:
                # Handle wizard values like "Oil & Gas / Refinery" - check all keywords
                industry_lower = industry.lower()
                industry_words = [w.strip() for w in industry_lower.replace('&', ' ').replace('/', ' ').split()]
                
                found_industry_match = False
                for industry_key, products in self.INDUSTRY_RECOMMENDATIONS.items():
                    # Check if any industry keyword matches (e.g., "refinery" in "oil & gas / refinery")
                    if any(industry_key in industry_lower or word in industry_key for word in industry_words):
                        if product.code in products:
                            score += 20
                            reasons.append(f"Recommended for {industry}")
                            found_industry_match = True
                            break
                
                if not found_industry_match:
                    # Check product's own industries list
                    if any(industry_lower in ind.lower() or any(w in ind.lower() for w in industry_words) for ind in product.industries):
                        score += 10
                        reasons.append(f"Used in {industry} industry")
            
            # Check application match
            if application or equipment_type:
                app_text = (equipment_type or application or "").lower()
                # Extract keywords from wizard values like "Valve Packing" → ["valve", "packing"]
                app_words = [w.strip() for w in app_text.replace('_', ' ').split()]
                
                found_app_match = False
                for app_key, products in self.APPLICATION_RECOMMENDATIONS.items():
                    # Check if any app keyword matches (e.g., "valve" in "Valve Packing")
                    if any(app_key in app_text or word in app_key or app_key in word for word in app_words):
                        if product.code in products:
                            score += 25
                            reasons.append(f"Recommended for {equipment_type or application}")
                            found_app_match = True
                            break
                
                if not found_app_match:
                    # Check product's own applications list
                    if any(any(w in app.lower() for w in app_words) for app in product.applications):
                        score += 15
                        reasons.append(f"Suitable for {equipment_type or application}")
            
            # Check media compatibility
            if media:
                media_lower = media.lower()
                if any(media_lower in m.lower() for m in product.service_media):
                    score += 15
                    reasons.append(f"Compatible with {media}")
            
            # Check certification requirements
            if certifications:
                for cert in certifications:
                    if cert in product.certifications:
                        score += 20
                        reasons.append(f"Has {cert} certification")
                    else:
                        # Check certification product mapping
                        cert_key = cert.lower().replace(' ', '_')
                        if cert_key in self.CERTIFICATION_PRODUCTS:
                            if product.code in self.CERTIFICATION_PRODUCTS[cert_key]:
                                score += 20
                                reasons.append(f"Has {cert} certification")
            
            # Check material preference
            if material_preference:
                if material_preference.lower() in product.material.lower():
                    score += 10
                    reasons.append(f"Contains {material_preference}")
            
            # Only include if there's some relevance
            if score > 0 and reasons:
                confidence = self._calculate_confidence(score, reasons)
                matches.append(ProductMatch(
                    product=product,
                    confidence=confidence,
                    score=score,
                    match_reasons=reasons
                ))
        
        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        
        return matches[:limit]
    
    def _calculate_confidence(self, score: float, reasons: List[str]) -> MatchConfidence:
        """Calculate confidence level based on score and match reasons."""
        if score >= 80 and len(reasons) >= 4:
            return MatchConfidence.EXACT
        elif score >= 50:
            return MatchConfidence.HIGH
        elif score >= 30:
            return MatchConfidence.MEDIUM
        else:
            return MatchConfidence.LOW
    
    def get_product_details(self, product_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific product."""
        product = self.catalog.get_product_by_code(product_code)
        if not product:
            return None
        
        return {
            "code": product.code,
            "name": product.name,
            "description": product.description,
            "category": product.category,
            "material": product.material,
            "features": product.features,
            "applications": product.applications,
            "industries": product.industries,
            "service_media": product.service_media,
            "certifications": product.certifications,
            "specifications": product.specs.to_dict() if product.specs else {},
            "available_forms": product.available_forms,
            "source_url": product.source_url,
            "searchable_text": product.to_searchable_text(),
        }
    
    def get_recommendations_for_selection(
        self,
        collected_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate product recommendations based on collected selection parameters.
        This is the main integration point with ProductSelectionAgent.
        
        Args:
            collected_params: Dictionary of parameters collected during selection:
                - industry: str
                - equipment_type: str
                - application_type: str
                - operating_temperature: float
                - operating_pressure: float
                - shaft_speed: float
                - media_type: str
                - media_description: str
                - ph_value: float
                - certifications_required: List[str]
                
        Returns:
            List of product recommendations with details
        """
        # Map selection parameters to retriever parameters
        industry = collected_params.get("industry")
        equipment = collected_params.get("equipment_type")
        application = collected_params.get("application_type")
        
        # Temperature
        temp = collected_params.get("operating_temperature")
        if isinstance(temp, str):
            # Parse temperature ranges like "100-200"
            try:
                if '-' in temp:
                    parts = temp.split('-')
                    temp = float(parts[1])  # Use max temp for matching
                else:
                    temp = float(temp)
            except:
                temp = None
        
        # Pressure
        pressure = collected_params.get("operating_pressure")
        if isinstance(pressure, str):
            try:
                pressure = float(pressure.replace(' bar', '').replace('bar', ''))
            except:
                pressure = None
        
        # Media
        media = collected_params.get("media_description") or collected_params.get("media_type")
        
        # pH
        ph = collected_params.get("ph_value")
        if isinstance(ph, str):
            try:
                ph = float(ph)
            except:
                ph = None
        
        # Certifications
        certs = collected_params.get("certifications_required", [])
        if isinstance(certs, str):
            certs = [certs]
        
        # Find matching products
        matches = self.find_products(
            industry=industry,
            application=application,
            equipment_type=equipment,
            media=media,
            operating_temp=temp,
            operating_pressure=pressure,
            ph_value=ph,
            certifications=certs,
            limit=5,
        )
        
        # Convert to recommendation format
        recommendations = []
        for match in matches:
            rec = match.to_dict()
            rec["recommendation_text"] = self._generate_recommendation_text(match)
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_recommendation_text(self, match: ProductMatch) -> str:
        """Generate human-readable recommendation text."""
        product = match.product
        
        text_parts = [
            f"**{product.code}** - {product.name}",
            f"\n{product.description}",
        ]
        
        if match.match_reasons:
            text_parts.append("\n**Why this product:**")
            for reason in match.match_reasons[:4]:
                text_parts.append(f"  - {reason}")
        
        if product.specs:
            specs_text = []
            if product.specs.temperature_max:
                specs_text.append(f"Temp: up to {product.specs.temperature_max}°C")
            if product.specs.pressure_static:
                specs_text.append(f"Pressure: {product.specs.pressure_static} bar")
            if specs_text:
                text_parts.append("\n**Specs:** " + ", ".join(specs_text))
        
        if product.certifications:
            text_parts.append(f"\n**Certifications:** {', '.join(product.certifications)}")
        
        return "\n".join(text_parts)
    
    def get_product_count(self) -> int:
        """Get total number of products in catalog."""
        return len(self.catalog.products)
    
    def get_categories(self) -> List[str]:
        """Get list of product categories."""
        categories = set()
        for product in self.catalog.products.values():
            categories.add(product.category)
        return sorted(list(categories))
    
    def get_industries(self) -> List[str]:
        """Get list of industries served."""
        industries = set()
        for product in self.catalog.products.values():
            industries.update(product.industries)
        return sorted(list(industries))


# Singleton instance
_retriever_instance: Optional[ProductCatalogRetriever] = None


def get_product_retriever() -> ProductCatalogRetriever:
    """Get or create the product retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ProductCatalogRetriever()
    return _retriever_instance


if __name__ == "__main__":
    # Test the retriever
    logging.basicConfig(level=logging.INFO)
    
    retriever = get_product_retriever()
    
    print(f"\nProduct catalog has {retriever.get_product_count()} products")
    print(f"Categories: {retriever.get_categories()}")
    
    # Test search
    print("\n--- Test: High temperature valve in power plant ---")
    matches = retriever.find_products(
        industry="power_plant",
        application="valve",
        operating_temp=500,
        operating_pressure=100,
    )
    
    for match in matches:
        print(f"\n{match.product.code}: {match.product.name}")
        print(f"  Score: {match.score}, Confidence: {match.confidence.value}")
        print(f"  Reasons: {', '.join(match.match_reasons)}")
