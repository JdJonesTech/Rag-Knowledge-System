"""
Product Catalog Retriever (Enhanced)
Provides intelligent, high-precision product search for the RAG system.
Bridges Product Selection Agent with actual structured product data.

Enhancements over v1:
- Weighted multi-signal scoring (temp, pressure, media families, certs, pH, shaft speed)
- Sealing-type routing (packing vs gasket vs insulation)
- Media-family normalization (hydrocarbons group, acids group, etc.)
- Spec-completeness penalty for products missing structured data
- Richer recommendation text with full structured specs
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
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
            "features": self.product.features,
            "industries": self.product.industries,
        }


# ---------------------------------------------------------------------------
# Media-family normalization
# ---------------------------------------------------------------------------
MEDIA_FAMILIES: Dict[str, List[str]] = {
    "hydrocarbons": [
        "oil", "gas", "fuel", "crude", "diesel", "kerosene", "gasoline",
        "naphtha", "petroleum", "hydrocarbon", "LPG", "LNG", "bitumen",
    ],
    "steam": [
        "steam", "superheated steam", "saturated steam", "boiler feed water",
    ],
    "water": [
        "water", "treated water", "cooling water", "demineralized water",
        "river water", "seawater", "brine",
    ],
    "acids": [
        "acid", "sulfuric", "hydrochloric", "nitric", "phosphoric",
        "acetic", "hcl", "h2so4", "hno3",
    ],
    "alkalis": [
        "alkali", "caustic", "caustic soda", "sodium hydroxide",
        "naoh", "lye", "ammonia",
    ],
    "solvents": [
        "solvent", "toluene", "benzene", "xylene", "acetone",
        "methanol", "ethanol",
    ],
    "gases_inert": [
        "nitrogen", "argon", "helium", "co2", "carbon dioxide",
        "inert gas",
    ],
    "gases_corrosive": [
        "h2s", "hydrogen sulfide", "sour gas", "chlorine", "cl2",
        "so2", "sulfur dioxide",
    ],
    "slurries": [
        "slurry", "slurries", "abrasive", "abrasives", "mud",
        "cement slurry", "mineral slurry", "pulp", "coal", "mineral",
        "suspended solids", "fibrous",
    ],
    "food_bev": [
        "food", "beverage", "juice", "milk", "beer", "wine", "sugar",
        "syrup", "chocolate",
    ],
    "pharma": [
        "pharmaceutical", "sterile", "usp", "api",
    ],
    "chemicals": [
        "chemical", "reagent", "amine", "glycol", "phenol",
    ],
}

# ---------------------------------------------------------------------------
# Sealing-type -> product category mapping
# ---------------------------------------------------------------------------
SEALING_TYPE_CATEGORIES: Dict[str, List[str]] = {
    "packing": [
        "compression packing", "graphite sealing products",
        "expanded ptfe products", "specialty lubricants",
        "low emission packing",
    ],
    "gasket": [
        "expanded ptfe products", "graphite sealing products",
    ],
    "insulation": [
        "insulation",
    ],
    "expansion_joint": [
        "expansion joint",
    ],
    "o_ring": [
        "industrial polymer products",
    ],
}


class ProductCatalogRetriever:
    """
    Intelligent product retriever for the RAG system.
    
    Matches customer requirements to products based on:
    - Operating conditions (temperature, pressure, pH, shaft speed)
    - Application type (pump, valve, agitator, etc.)
    - Industry
    - Material preferences
    - Certification requirements
    - Media compatibility (family-aware)
    - Sealing type routing (packing / gasket / insulation)
    """
    
    # Application to product recommendations (expert knowledge + scraped metadata)
    APPLICATION_RECOMMENDATIONS: Dict[str, List[str]] = {
        "pump": [
            "NA 758H", "NA 802", "NA 719", "NA 730", "NA 757",
            # From metadata: products listing pump applications
            "NA 441", "NA 442", "NA 702", "NA 702R", "NA 717",
            "NA 730SC", "NA 731", "NA 731CC", "NA 732", "NA 737", "NA 746",
            "NA 748", "NA 751M", "NA 751SR", "NA 753", "NA 755",
            "NA 758CCC", "NA 758HG", "NA 759", "NA 763RC",
            "NA 767", "NA 768", "NA 769", "NA 772", "NA 774",
            "NA 775", "NA 776", "NA 777", "NA 783", "NA 790", "NA 801",
        ],
        "centrifugal_pump": [
            "NA 802", "NA 758H", "NA 740", "NA 747",
            "NA 759", "NA 763", "NA 763G", "NA 763RC", "NA 20747",
        ],
        "plunger_pump": [
            "NA 781PP", "NA 745PP", "NA 757",
            "NA 745", "NA 760", "NA 763", "NA 763G", "NA 763RC",
        ],
        "valve": [
            "NA 701", "NA 707", "NA 715", "NA 752", "NA 781G",
            # From metadata: products listing valve applications
            "NA 441", "NA 442", "NA 600", "NA 620", "NA 701 + 707",
            "NA 702", "NA 702R", "NA 707SC", "NA 707T", "NA 708",
            "NA 710V", "NA 710VS", "NA 717", "NA 718", "NA 720",
            "NA 721", "NA 722", "NA 730SC", "NA 731", "NA 731CC",
            "NA 751M", "NA 751SR", "NA 754PL", "NA 755", "NA 768", "NA 769",
            "NA 772", "NA 776", "NA 778", "NA 781A", "NA 781L",
            "NA 801",
        ],
        "block_valve": ["NA 715", "NA 701"],
        "control_valve": ["NA 781G", "NA 781A", "NA 701", "NA 707", "NA 721"],
        "agitator": [
            "NA 758H", "NA 802", "NA 740", "NA 741",
            "NA 745", "NA 746", "NA 751M", "NA 751SR", "NA 755",
            "NA 759", "NA 760", "NA 763", "NA 763G", "NA 763RC",
            "NA 772", "NA 790",
        ],
        "mixer": [
            "NA 758H", "NA 802", "NA 730",
            "NA 745", "NA 746", "NA 751M", "NA 751SR", "NA 755",
            "NA 760", "NA 772", "NA 790",
        ],
        "compressor": ["NA 752", "NA 751", "NA 754", "NA 20747", "NA 716"],
        "blower": ["NA 730", "NA 733", "NA 20747", "NA 705"],
        "reactor": ["NA 701T", "NA 706", "NA 790"],
        "flange": ["NA 610", "NA 706", "NA 709", "NA 706TTX", "NA 703"],
        "gasket": [
            "NA 610", "NA 706", "NA 709", "NA 711", "NA 706M",
            "NA 706TTX", "NA 703", "NA 712", "NA 714",
        ],
        "boiler": ["NA 550", "NA 555", "NA 713", "NA 551", "NA 552", "NA 704", "NA 705"],
        "turbine": ["NA 711", "NA 710", "NA 556", "NA 557V"],
        "insulation": [
            "NA 550", "NA 555", "NA 550G", "NA 555V", "NA 559",
            "NA 551", "NA 552", "NA 556", "NA 557V", "NA 558",
        ],
        # New: catch-all for general/other industrial
        "other": [
            "NA 701T", "NA 733A", "NA 733KC", "NA 734", "NA 770", "NA 778",
            "NA 781L", "NA 783", "NA 784", "NA 785", "NA 802SR", "NA 9000",
            "NA 731CC", "NA 745PP",
        ],
    }
    
    # Industry-specific recommendations (expert knowledge + scraped metadata)
    INDUSTRY_RECOMMENDATIONS: Dict[str, List[str]] = {
        "power_plant": [
            "NA 701", "NA 715", "NA 758H", "NA 802", "NA 705",
            "NA 550", "NA 555", "NA 551", "NA 552", "NA 556",
            "NA 557V", "NA 704", "NA 711", "NA 713",
            # Valve mfr products also used in power
            "NA 707T", "NA 708", "NA 710V", "NA 710VS", "NA 721",
        ],
        "refinery": [
            "NA 715", "NA 701", "NA 752", "NA 707",
            "NA 716", "NA 720", "NA 722", "NA 753",
            # Valve mfr products also used in refinery
            "NA 707T", "NA 708", "NA 710V", "NA 710VS", "NA 721",
            "NA 768", "NA 769", "NA 772", "NA 776",
        ],
        "petrochemical": [
            "NA 715", "NA 752", "NA 701", "NA 758H",
            "NA 716", "NA 720", "NA 722",
        ],
        "chemical": [
            "NA 758H", "NA 757", "NA 737", "NA 802",
            # From metadata: products with Chemical Plants industry
            "NA 441", "NA 442", "NA 600", "NA 620", "NA 702",
            "NA 702R", "NA 717", "NA 718", "NA 721", "NA 722",
            "NA 730SC", "NA 731", "NA 745", "NA 746", "NA 748",
            "NA 751M", "NA 751SR", "NA 753", "NA 754PL", "NA 755",
            "NA 758CCC", "NA 758HG", "NA 759", "NA 760",
            "NA 763", "NA 763G", "NA 763RC", "NA 767", "NA 772",
            "NA 777", "NA 781A", "NA 781G", "NA 781L",
            "NA 784", "NA 785", "NA 790", "NA 801", "NA 9000",
            "NA 701T", "NA 707T", "NA 745PP",
        ],
        "pulp_paper": [
            "NA 783", "NA 758H", "NA 802", "NA 740",
            "NA 731CC", "NA 732", "NA 733A", "NA 733KC", "NA 734",
            "NA 753", "NA 759", "NA 763", "NA 763G", "NA 763RC",
            "NA 784", "NA 785", "NA 802SR",
        ],
        "sugar": [
            "NA 747", "NA 748", "NA 740",
            "NA 733A", "NA 733KC", "NA 734", "NA 802SR",
        ],
        "mining": [
            "NA 737", "NA 757", "NA 730", "NA 758CCC",
            "NA 733A", "NA 733KC", "NA 734",
        ],
        "food": [
            "NA 740", "NA 781", "NA 730",
            "NA 759", "NA 781L", "NA 781PP", "NA 714",
        ],
        "pharmaceutical": [
            "NA 781", "NA 757", "NA 730",
            "NA 759", "NA 781L", "NA 600",
        ],
        "steel": [
            "NA 715", "NA 701", "NA 701T", "NA 711",
            "NA 550", "NA 555", "NA 704",
        ],
        "cement": [
            "NA 550", "NA 555", "NA 550G", "NA 555V",
            "NA 551", "NA 552", "NA 558",
        ],
        # New: Valve Manufacturers (many products list this industry)
        "valve_manufacturer": [
            "NA 701", "NA 707", "NA 715", "NA 752",
            "NA 701 + 707", "NA 707SC", "NA 707T", "NA 708",
            "NA 710V", "NA 710VS", "NA 712", "NA 721", "NA 731CC",
            "NA 751M", "NA 751SR", "NA 754PL", "NA 755",
            "NA 758HG", "NA 768", "NA 769", "NA 770",
            "NA 772", "NA 776", "NA 778", "NA 781A", "NA 781G",
            "NA 781PP", "NA 801", "NA 802SR",
        ],
        # Catch-all: products with no specific industry
        "other": [
            "NA 20747", "NA 600", "NA 703", "NA 706TTX",
            "NA 774", "NA 775",
        ],
    }
    
    # Certification mappings — verified from jdjones.com product pages (Feb 2026)
    CERTIFICATION_PRODUCTS: Dict[str, List[str]] = {
        # Verified: NA 715 (API 622 3rd Ed), NA B-3 + 707 (API 622 3rd Ed), NA SP-1 (API 622 2nd+3rd Ed)
        "api 622": ["NA 715", "NA B-3 + 707", "NA SP-1"],
        # Verified: NA 715, NA 719 (7th Ed), NA B-3 + 707 (tested at Yarmouth), NA SP-1 (tested at Yarmouth)
        "api 589": ["NA 715", "NA 719", "NA B-3 + 707", "NA SP-1"],
        # Verified: NA 715, NA 719 (7th Ed), NA B-3 + 707 (tested), NA SP-1 (tested)
        "api 607": ["NA 715", "NA 719", "NA B-3 + 707", "NA SP-1"],
        # Verified: NA 715, NA B-3 + 707 (valve qualification)
        "api 624": ["NA 715", "NA B-3 + 707"],
        # Verified: NA 715, NA B-3 + 707 (valve qualification), NA SP-1 (Part 1, -196°C to 400°C, Class 1500)
        "iso 15848": ["NA 715", "NA B-3 + 707", "NA SP-1"],
        # Verified: NA 719 (ISO 10497:2010 from Yarmouth)
        "iso 10497": ["NA 719"],
        # Verified: NA 701 (Shell SPE MESC 77/312 Class B with NA 707), NA 715
        "shell spe": ["NA 701", "NA 715"],
        # Verified: NA 740 "Food Grade available", NA 740 MA, NA 747, NA 781 "Food Grade available", NA 714
        "food_grade": ["NA 740", "NA 740 MA", "NA 781", "NA 747", "NA 714"],
        # Verified: NA 740, NA 740 MA, NA 759, NA 781, NA 747, NA 714 (FDA/food contact)
        "fda": ["NA 740", "NA 740 MA", "NA 759", "NA 781", "NA 747", "NA 714"],
        # Verified: NA 715 (ultra low emission), NA B-3 + 707 (<100ppm), NA SP-1 (ultra low emission)
        "low_emission": ["NA 715", "NA B-3 + 707", "NA SP-1"],
        # Verified: NA 715 (API 589/607), NA 719 (API 589/607), NA B-3 + 707, NA SP-1
        "fire_safe": ["NA 715", "NA 719", "NA B-3 + 707", "NA SP-1"],
        # Verified: NA 715 (API 622), NA 701 (Shell SPE), NA B-3 + 707 (API 622), NA SP-1 (API 622)
        "fugitive_emissions": ["NA 715", "NA 701", "NA B-3 + 707", "NA SP-1"],
    }
    
    # ---------------------------------------------------------------------------
    # Scoring weights (out of a target max ~150 for a perfect match)
    # ---------------------------------------------------------------------------
    W_TEMPERATURE  = 25
    W_PRESSURE     = 20
    W_INDUSTRY     = 15
    W_APPLICATION  = 20
    W_MEDIA        = 15
    W_SEALING_TYPE = 15
    W_CERTIFICATION = 20
    W_PH           = 10
    W_SHAFT_SPEED  = 10
    W_MATERIAL     = 10
    PENALTY_NO_SPECS = -8   # Penalty if product has zero spec data
    
    def __init__(self, catalog: Optional[ProductCatalogLoader] = None):
        self.catalog = catalog or get_product_catalog()
    
    # ---------------------------------------------------------------------------
    # Normalization helpers
    # ---------------------------------------------------------------------------
    @staticmethod
    def _normalize_industry(raw: str) -> Set[str]:
        """Convert a free-text industry string into canonical tokens."""
        low = raw.lower()
        tokens: Set[str] = set()
        mapping = {
            "power": "power_plant", "refinery": "refinery", "oil": "refinery",
            "petrochemical": "petrochemical", "chemical": "chemical",
            "pulp": "pulp_paper", "paper": "pulp_paper",
            "sugar": "sugar", "mining": "mining", "mine": "mining",
            "food": "food", "beverage": "food",
            "pharma": "pharmaceutical", "steel": "steel",
            "cement": "cement", "marine": "marine",
            "valve": "valve_manufacturer",
        }
        for keyword, canon in mapping.items():
            if keyword in low:
                tokens.add(canon)
        if not tokens and "other" in low:
            tokens.add("other")
        return tokens
    
    @staticmethod
    def _normalize_application(raw: str) -> Set[str]:
        """Convert a free-text application to canonical tokens."""
        low = raw.lower()
        tokens: Set[str] = set()
        mapping = {
            "valve": "valve", "pump": "pump", "agitator": "agitator",
            "mixer": "mixer", "compressor": "compressor",
            "flange": "flange", "gasket": "gasket",
            "boiler": "boiler", "turbine": "turbine",
            "reactor": "reactor", "vessel": "reactor",
            "expansion joint": "expansion_joint",
            "blower": "blower", "centrifugal": "centrifugal_pump",
            "plunger": "plunger_pump", "reciprocating": "plunger_pump",
            "block valve": "block_valve", "isolation valve": "block_valve",
            "control valve": "control_valve",
            "insulation": "insulation",
            "heat exchanger": "gasket",
        }
        for keyword, canon in mapping.items():
            if keyword in low:
                tokens.add(canon)
        if not tokens and "other" in low:
            tokens.add("other")
        return tokens
    
    @staticmethod
    def _media_families_for_query(media_text: str) -> Set[str]:
        """Return the set of media family keys that match the query."""
        low = media_text.lower()
        matched: Set[str] = set()
        for family, keywords in MEDIA_FAMILIES.items():
            for kw in keywords:
                if kw in low:
                    matched.add(family)
                    break
        return matched
    
    @staticmethod
    def _product_media_families(product: Product) -> Set[str]:
        """Return the set of media families a product is compatible with."""
        combined = " ".join(product.service_media + product.applications + product.features)
        low = combined.lower()
        # Also include description
        low += " " + product.description.lower()
        matched: Set[str] = set()
        for family, keywords in MEDIA_FAMILIES.items():
            for kw in keywords:
                if kw in low:
                    matched.add(family)
                    break
        return matched
    
    @staticmethod
    def _product_has_specs(product: Product) -> bool:
        """Check whether a product has *any* meaningful spec data."""
        if not product.specs:
            return False
        s = product.specs
        return any([
            s.temperature_min is not None,
            s.temperature_max is not None,
            s.pressure_static is not None,
            s.pressure_rotary is not None,
            s.pressure_reciprocating is not None,
        ])
    
    # ---------------------------------------------------------------------------
    # Main search
    # ---------------------------------------------------------------------------
    def find_products(
        self,
        industry: Optional[str] = None,
        application: Optional[str] = None,
        equipment_type: Optional[str] = None,
        media: Optional[str] = None,
        sealing_type: Optional[str] = None,
        operating_temp: Optional[float] = None,
        operating_pressure: Optional[float] = None,
        shaft_speed: Optional[float] = None,
        ph_value: Optional[float] = None,
        certifications: Optional[List[str]] = None,
        material_preference: Optional[str] = None,
        limit: int = 5,
    ) -> List[ProductMatch]:
        """
        Find matching products using weighted multi-signal scoring.
        
        The scoring is additive across independent dimensions so that a product
        matching 5 criteria always outranks one matching only 2.
        """
        # Pre-compute query-side normalized tokens
        ind_tokens = self._normalize_industry(industry) if industry else set()
        app_tokens = self._normalize_application(
            equipment_type or application or ""
        ) if (equipment_type or application) else set()
        media_families = self._media_families_for_query(media) if media else set()
        
        # Normalize sealing type for category filtering
        sealing_key = None
        if sealing_type:
            sl = sealing_type.lower()
            if "gasket" in sl:
                sealing_key = "gasket"
            elif "packing" in sl:
                sealing_key = "packing"
            elif "insulation" in sl:
                sealing_key = "insulation"
            elif "expansion" in sl:
                sealing_key = "expansion_joint"
            elif "o-ring" in sl or "o ring" in sl:
                sealing_key = "o_ring"
        
        matches: List[ProductMatch] = []
        
        for product in self.catalog.products.values():
            score = 0.0
            reasons: List[str] = []
            disqualified = False
            
            # ------ Sealing-type filter (hard filter) ------
            if sealing_key and sealing_key in SEALING_TYPE_CATEGORIES:
                allowed_cats = SEALING_TYPE_CATEGORIES[sealing_key]
                if not any(product.category.lower().startswith(cat) or cat in product.category.lower()
                           for cat in allowed_cats):
                    continue  # Wrong product family entirely
                score += self.W_SEALING_TYPE
                reasons.append(f"Category matches {sealing_type}")
            
            # ------ Temperature ------
            if operating_temp is not None and product.specs:
                t_min = product.specs.temperature_min
                t_max = product.specs.temperature_max
                if t_min is not None and t_max is not None:
                    if t_min <= operating_temp <= t_max:
                        # Full score; bonus if product has wide margin
                        margin = (t_max - operating_temp) / max(t_max - t_min, 1)
                        score += self.W_TEMPERATURE
                        reasons.append(
                            f"Temperature compatible ({t_min:.0f}°C to {t_max:.0f}°C)"
                        )
                    else:
                        disqualified = True  # Hard disqualify on temperature
                        continue
                # If specs exist but temp fields are None, don't disqualify but no points
            
            # ------ Pressure ------
            if operating_pressure is not None and product.specs:
                max_p = max(
                    product.specs.pressure_static or 0,
                    product.specs.pressure_rotary or 0,
                    product.specs.pressure_reciprocating or 0,
                )
                if max_p > 0:
                    if max_p >= operating_pressure:
                        score += self.W_PRESSURE
                        reasons.append(f"Pressure compatible (up to {max_p:.0f} bar)")
                    else:
                        score -= self.W_PRESSURE * 0.5  # Penalty but don't exclude
                        reasons.append(f"Pressure marginally low ({max_p:.0f} bar vs {operating_pressure:.0f} required)")
            
            # ------ pH ------
            if ph_value is not None and product.specs:
                ph_min = product.specs.ph_min
                ph_max = product.specs.ph_max
                if ph_min is not None and ph_max is not None:
                    if ph_min <= ph_value <= ph_max:
                        score += self.W_PH
                        reasons.append(f"pH compatible ({ph_min}-{ph_max})")
                    else:
                        score -= self.W_PH * 0.5
            
            # ------ Shaft speed ------
            if shaft_speed is not None and product.specs:
                ss = product.specs.shaft_speed_rotary
                if ss is not None:
                    if ss >= shaft_speed:
                        score += self.W_SHAFT_SPEED
                        reasons.append(f"Shaft speed compatible ({ss} m/s)")
                    else:
                        score -= self.W_SHAFT_SPEED * 0.3
            
            # ------ Industry (two-tier: expert table + product metadata) ------
            if ind_tokens:
                industry_scored = False
                # Tier 1: Expert recommendation table
                for canon_ind in ind_tokens:
                    rec_list = self.INDUSTRY_RECOMMENDATIONS.get(canon_ind, [])
                    if product.code in rec_list:
                        score += self.W_INDUSTRY
                        reasons.append(f"Expert-recommended for {industry}")
                        industry_scored = True
                        break
                # Tier 2: Product's own industry metadata
                if not industry_scored:
                    prod_ind_lower = [i.lower() for i in product.industries]
                    for canon_ind in ind_tokens:
                        if any(canon_ind in pi or pi in canon_ind for pi in prod_ind_lower):
                            score += self.W_INDUSTRY * 0.6
                            reasons.append(f"Used in {industry} industry")
                            industry_scored = True
                            break
                    # Tier 3: fuzzy word overlap on raw industry text
                    if not industry_scored and industry:
                        ind_words = set(re.split(r'[\s/&,]+', industry.lower()))
                        for pi in prod_ind_lower:
                            pi_words = set(re.split(r'[\s/&,]+', pi))
                            if ind_words & pi_words:
                                score += self.W_INDUSTRY * 0.3
                                reasons.append(f"Partial industry match ({industry})")
                                break
            
            # ------ Application (two-tier) ------
            if app_tokens:
                app_scored = False
                for canon_app in app_tokens:
                    rec_list = self.APPLICATION_RECOMMENDATIONS.get(canon_app, [])
                    if product.code in rec_list:
                        score += self.W_APPLICATION
                        reasons.append(f"Expert-recommended for {equipment_type or application}")
                        app_scored = True
                        break
                if not app_scored:
                    prod_apps_lower = [a.lower() for a in product.applications]
                    for canon_app in app_tokens:
                        if any(canon_app in pa or pa in canon_app for pa in prod_apps_lower):
                            score += self.W_APPLICATION * 0.6
                            reasons.append(f"Suitable for {equipment_type or application}")
                            app_scored = True
                            break
                    if not app_scored:
                        # Check product name/description for application words
                        combined = (product.name + " " + product.description).lower()
                        for canon_app in app_tokens:
                            if canon_app in combined:
                                score += self.W_APPLICATION * 0.3
                                reasons.append(f"Related to {equipment_type or application}")
                                break
            
            # ------ Media compatibility (family-aware) ------
            if media_families:
                prod_families = self._product_media_families(product)
                overlap = media_families & prod_families
                if overlap:
                    score += self.W_MEDIA
                    reasons.append(f"Media compatible ({', '.join(overlap)})")
                else:
                    # Direct substring check as fallback
                    if media:
                        combined_media = " ".join(
                            product.service_media + product.applications + product.features
                        ).lower()
                        if media.lower() in combined_media:
                            score += self.W_MEDIA * 0.5
                            reasons.append(f"Media mention found ({media})")
            
            # ------ Certifications ------
            if certifications:
                cert_score = 0
                for cert in certifications:
                    cert_lower = cert.lower().strip()
                    if cert_lower in ("none", "none specific", "not sure", "none required"):
                        cert_score += self.W_CERTIFICATION * 0.2  # Neutral
                        break
                    # Check product certifications directly
                    if any(cert_lower in pc.lower() or pc.lower() in cert_lower
                           for pc in product.certifications):
                        cert_score += self.W_CERTIFICATION
                        reasons.append(f"Has {cert} certification")
                    else:
                        # Check expert certification table (normalize underscores)
                        cert_norm = cert_lower.replace("_", " ")
                        for ck, products_list in self.CERTIFICATION_PRODUCTS.items():
                            ck_norm = ck.replace("_", " ")
                            if (ck_norm in cert_norm or cert_norm in ck_norm
                                    or ck in cert_lower or cert_lower in ck):
                                if product.code in products_list:
                                    cert_score += self.W_CERTIFICATION
                                    reasons.append(f"Has {cert} certification (expert)")
                                    break
                score += min(cert_score, self.W_CERTIFICATION * 2)  # Cap at 2x
            
            # ------ Material preference ------
            if material_preference:
                mat_low = material_preference.lower()
                if mat_low in product.material.lower():
                    score += self.W_MATERIAL
                    reasons.append(f"Contains {material_preference}")
                elif mat_low in product.name.lower() or mat_low in product.description.lower():
                    score += self.W_MATERIAL * 0.5
                    reasons.append(f"Material related: {material_preference}")
            
            # ------ Spec-completeness penalty ------
            if not self._product_has_specs(product):
                score += self.PENALTY_NO_SPECS
            
            # ------ Threshold: only keep products with positive relevance ------
            if score > 0 and reasons:
                confidence = self._calculate_confidence(score, reasons)
                matches.append(ProductMatch(
                    product=product,
                    confidence=confidence,
                    score=round(score, 2),
                    match_reasons=reasons,
                ))
        
        # Sort by score descending
        matches.sort(key=lambda m: m.score, reverse=True)
        
        return matches[:limit]
    
    def _calculate_confidence(self, score: float, reasons: List[str]) -> MatchConfidence:
        """Calculate confidence level based on score and number of matching dimensions."""
        if score >= 80 and len(reasons) >= 4:
            return MatchConfidence.EXACT
        elif score >= 50 and len(reasons) >= 3:
            return MatchConfidence.HIGH
        elif score >= 25:
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
        
        Strategy:
        1. Try STRICT search first (hard filters on industry/application/media)
           — this ensures recommendations truly match the wizard criteria.
        2. Fall back to the softer find_products only if strict returns nothing
           (which should not happen when the wizard disables dead-end options).
        """
        # Map selection parameters to retriever parameters
        industry = collected_params.get("industry")
        equipment = collected_params.get("equipment_type")
        application = collected_params.get("application_type")
        sealing_type = collected_params.get("sealing_type")
        
        # Temperature
        temp = collected_params.get("operating_temperature") or collected_params.get("max_temperature")
        if isinstance(temp, str):
            try:
                if '-' in temp:
                    parts = temp.split('-')
                    temp = float(parts[1])
                else:
                    temp = float(temp)
            except (ValueError, IndexError):
                temp = None
        
        # Pressure
        pressure = collected_params.get("operating_pressure") or collected_params.get("max_pressure")
        if isinstance(pressure, str):
            try:
                pressure = float(pressure.replace(' bar', '').replace('bar', ''))
            except ValueError:
                pressure = None
        
        # Media
        media = (
            collected_params.get("media_description")
            or collected_params.get("media_type")
            or collected_params.get("media")
        )
        
        # pH
        ph = collected_params.get("ph_value")
        if isinstance(ph, str):
            try:
                ph = float(ph)
            except ValueError:
                ph = None
        
        # Shaft speed
        shaft = collected_params.get("shaft_speed")
        if isinstance(shaft, str):
            try:
                shaft = float(shaft)
            except ValueError:
                shaft = None
        
        # Certifications
        certs = collected_params.get("certifications_required", [])
        if isinstance(certs, str):
            certs = [certs]
        
        # Material
        material = collected_params.get("material_preference")
        
        # ---- PRIMARY: Strict search (hard filters) ----
        matches = self.find_products_strict(
            industry=industry,
            application=application or equipment,
            media=media,
            sealing_type=sealing_type,
            operating_temp=temp,
            operating_pressure=pressure,
            certifications=certs,
            limit=10,
        )
        
        # ---- FALLBACK 1: Strict with only cert filter (relax other hard filters) ----
        if not matches and certs:
            real_certs = [c for c in certs if "none" not in c.lower()]
            if real_certs:
                logger.info("Strict search returned no results, retrying with cert-only hard filter")
                matches = self.find_products_strict(
                    certifications=certs,
                    operating_temp=temp,
                    operating_pressure=pressure,
                    limit=10,
                )
        
        # ---- FALLBACK 2: Soft search if still nothing ----
        if not matches:
            logger.info("Strict search returned no results, falling back to soft scoring")
            matches = self.find_products(
                industry=industry,
                application=application,
                equipment_type=equipment,
                media=media,
                sealing_type=sealing_type,
                operating_temp=temp,
                operating_pressure=pressure,
                shaft_speed=shaft,
                ph_value=ph,
                certifications=certs,
                material_preference=material,
                limit=5,
            )
        
        # Convert to recommendation format
        recommendations = []
        for match in matches:
            rec = match.to_dict()
            rec["recommendation_text"] = self._generate_recommendation_text(match)
            # Enrich certifications from expert table (product.certifications
            # is often empty even when the product is known to be certified)
            explicit_certs = set(c.lower() for c in rec.get("certifications", []))
            expert_certs = []
            for cert_key, product_codes in self.CERTIFICATION_PRODUCTS.items():
                if match.product.code in product_codes:
                    # Format nicely: "api 622" → "API 622", "food_grade" → "Food Grade"
                    display = cert_key.replace("_", " ").upper() if len(cert_key) <= 7 else cert_key.replace("_", " ").title()
                    if display.lower() not in explicit_certs:
                        expert_certs.append(display)
            if expert_certs:
                rec["certifications"] = rec.get("certifications", []) + expert_certs
            recommendations.append(rec)
        
        return recommendations
    
    def _product_has_certification(self, product: Product, certs: List[str]) -> bool:
        """Check if a product has at least one of the requested certifications."""
        for cert in certs:
            cert_lower = cert.lower().strip()
            # Check explicit certifications
            if any(cert_lower in pc.lower() or pc.lower() in cert_lower
                   for pc in product.certifications):
                return True
            # Extract core cert ID (before parens)
            core_match = re.match(r'^([^(]+)', cert_lower)
            core_cert = core_match.group(1).strip() if core_match else cert_lower
            core_cert_norm = core_cert.replace("_", " ")
            # Check expert table  
            for ck, products_list in self.CERTIFICATION_PRODUCTS.items():
                ck_norm = ck.replace("_", " ")
                if (ck_norm == core_cert_norm
                        or ck_norm in core_cert_norm
                        or core_cert_norm in ck_norm):
                    if product.code in products_list:
                        return True
        return False
    
    def _generate_recommendation_text(self, match: ProductMatch) -> str:
        """Generate human-readable recommendation text with full structured specs."""
        product = match.product
        
        text_parts = [
            f"**{product.code}** - {product.name}",
            f"\n{product.description}",
        ]
        
        if match.match_reasons:
            text_parts.append("\n**Why this product:**")
            for reason in match.match_reasons[:6]:
                text_parts.append(f"  - {reason}")
        
        if product.specs:
            specs_lines = []
            if product.specs.temperature_min is not None and product.specs.temperature_max is not None:
                specs_lines.append(
                    f"Temperature: {product.specs.temperature_min:.0f}°C to {product.specs.temperature_max:.0f}°C"
                )
            if product.specs.pressure_static:
                specs_lines.append(f"Pressure (static): {product.specs.pressure_static:.0f} bar")
            if product.specs.pressure_rotary:
                specs_lines.append(f"Pressure (rotary): {product.specs.pressure_rotary:.0f} bar")
            if product.specs.pressure_reciprocating:
                specs_lines.append(f"Pressure (reciprocating): {product.specs.pressure_reciprocating:.0f} bar")
            if product.specs.shaft_speed_rotary:
                specs_lines.append(f"Shaft speed: {product.specs.shaft_speed_rotary:.0f} m/s")
            if product.specs.ph_min is not None and product.specs.ph_max is not None:
                specs_lines.append(f"pH: {product.specs.ph_min:.0f} to {product.specs.ph_max:.0f}")
            if specs_lines:
                text_parts.append("\n**Specifications:**")
                for sl in specs_lines:
                    text_parts.append(f"  - {sl}")
        
        if product.certifications:
            text_parts.append(f"\n**Certifications:** {', '.join(product.certifications)}")
        
        if product.material:
            text_parts.append(f"\n**Material:** {product.material}")
        
        return "\n".join(text_parts)
    
    def structured_product_summary(self, product_code: str) -> Optional[str]:
        """
        Generate a rich, structured summary of a product for LLM context.
        Leverages all available structured fields.
        """
        product = self.catalog.get_product_by_code(product_code)
        if not product:
            return None
        
        lines = [
            f"=== {product.code} - {product.name} ===",
            f"Category: {product.category}",
            f"Material: {product.material}" if product.material else None,
            f"Description: {product.description}",
        ]
        
        if product.specs:
            lines.append("\nTechnical Specifications:")
            s = product.specs
            if s.temperature_min is not None and s.temperature_max is not None:
                lines.append(f"  Temperature Range: {s.temperature_min:.0f}°C to {s.temperature_max:.0f}°C")
            if s.pressure_static:
                lines.append(f"  Max Pressure (static): {s.pressure_static:.0f} bar")
            if s.pressure_rotary:
                lines.append(f"  Max Pressure (rotary): {s.pressure_rotary:.0f} bar")
            if s.pressure_reciprocating:
                lines.append(f"  Max Pressure (reciprocating): {s.pressure_reciprocating:.0f} bar")
            if s.shaft_speed_rotary:
                lines.append(f"  Max Shaft Speed (rotary): {s.shaft_speed_rotary:.0f} m/s")
            if s.ph_min is not None and s.ph_max is not None:
                lines.append(f"  pH Range: {s.ph_min:.0f} to {s.ph_max:.0f}")
        
        if product.features:
            lines.append("\nKey Features:")
            for f in product.features[:8]:
                lines.append(f"  - {f}")
        
        if product.applications:
            clean_apps = [a for a in product.applications if a.lower() not in ("enquire now", "service media", "conditions")]
            if clean_apps:
                lines.append(f"\nApplications: {', '.join(clean_apps)}")
        
        if product.industries:
            lines.append(f"Industries: {', '.join(product.industries)}")
        
        if product.service_media:
            lines.append(f"Service Media: {', '.join(product.service_media)}")
        
        if product.certifications:
            lines.append(f"Certifications: {', '.join(product.certifications)}")
        
        if product.source_url:
            lines.append(f"URL: {product.source_url}")
        
        return "\n".join(l for l in lines if l is not None)
    
    # ---------------------------------------------------------------------------
    # Strict wizard search (hard-filter on wizard selections)
    # ---------------------------------------------------------------------------
    def find_products_strict(
        self,
        industry: Optional[str] = None,
        application: Optional[str] = None,
        media: Optional[str] = None,
        sealing_type: Optional[str] = None,
        operating_temp: Optional[float] = None,
        operating_pressure: Optional[float] = None,
        certifications: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[ProductMatch]:
        """
        Strict variant of find_products used by the wizard flow.
        
        Wizard parameters act as HARD filters:
        - Industry: product must be associated with the selected industry
        - Application: product must be suitable for the selected application type
        - Media: product must be compatible with the selected media
        - Temperature: product must cover the operating temperature
        - Sealing type: same hard filter as find_products
        
        Only products passing ALL hard filters are scored and returned.
        """
        # Pre-compute normalized tokens
        ind_tokens = self._normalize_industry(industry) if industry else set()
        app_tokens = self._normalize_application(application or "") if application else set()
        media_families = self._media_families_for_query(media) if media else set()
        
        # Derive sealing type from application (wizard doesn't have separate sealing step)
        sealing_key = None
        if sealing_type:
            sl = sealing_type.lower()
            for key in SEALING_TYPE_CATEGORIES:
                if key in sl:
                    sealing_key = key
                    break
        # Also infer from application
        if not sealing_key and application:
            al = application.lower()
            if "gasket" in al or "flange" in al or "heat exchanger" in al:
                sealing_key = "gasket"
            elif "insulation" in al:
                sealing_key = "insulation"
            elif any(kw in al for kw in ("valve", "pump", "agitator", "mixer",
                                          "compressor", "blower", "boiler",
                                          "turbine")):
                sealing_key = "packing"
        
        matches: List[ProductMatch] = []
        
        for product in self.catalog.products.values():
            score = 0.0
            reasons: List[str] = []
            
            # =========== HARD FILTERS ===========
            
            # --- Sealing type (hard — but exempt expert-assigned products) ---
            if sealing_key and sealing_key in SEALING_TYPE_CATEGORIES:
                allowed_cats = SEALING_TYPE_CATEGORIES[sealing_key]
                cat_match = any(
                    product.category.lower().startswith(cat) or cat in product.category.lower()
                    for cat in allowed_cats
                )
                # Check if product is explicitly in the expert table for this application
                expert_exempt = False
                if app_tokens:
                    for canon_app in app_tokens:
                        if product.code in self.APPLICATION_RECOMMENDATIONS.get(canon_app, []):
                            expert_exempt = True
                            break
                if not cat_match and not expert_exempt:
                    continue  # Wrong product family
                if cat_match:
                    score += self.W_SEALING_TYPE
                    reasons.append(f"Category matches sealing type")
                elif expert_exempt:
                    score += self.W_SEALING_TYPE * 0.5
                    reasons.append(f"Expert-assigned for application")
            
            # --- Temperature (hard) ---
            if operating_temp is not None and product.specs:
                t_min = product.specs.temperature_min
                t_max = product.specs.temperature_max
                if t_min is not None and t_max is not None:
                    if not (t_min <= operating_temp <= t_max):
                        continue  # Hard disqualify
                    margin = (t_max - operating_temp) / max(t_max - t_min, 1)
                    score += self.W_TEMPERATURE
                    reasons.append(f"Temperature compatible ({t_min:.0f}°C to {t_max:.0f}°C)")
                # If specs exist but no temp data, don't filter but no points
            
            # --- Industry (SOFT scoring — boosts rank but never excludes) ---
            if ind_tokens:
                industry_matched = False
                # Tier 1: Expert recommendation table
                for canon_ind in ind_tokens:
                    rec_list = self.INDUSTRY_RECOMMENDATIONS.get(canon_ind, [])
                    if product.code in rec_list:
                        score += self.W_INDUSTRY
                        reasons.append(f"Expert-recommended for {industry}")
                        industry_matched = True
                        break
                # Tier 2: Product's industry metadata
                if not industry_matched:
                    prod_ind_lower = [i.lower() for i in product.industries]
                    for canon_ind in ind_tokens:
                        if any(canon_ind in pi or pi in canon_ind for pi in prod_ind_lower):
                            score += self.W_INDUSTRY * 0.6
                            reasons.append(f"Used in {industry} industry")
                            industry_matched = True
                            break
                    # Tier 3: Fuzzy word overlap
                    if not industry_matched and industry:
                        ind_words = set(re.split(r'[\s/&,]+', industry.lower()))
                        for pi in prod_ind_lower:
                            pi_words = set(re.split(r'[\s/&,]+', pi))
                            if ind_words & pi_words:
                                score += self.W_INDUSTRY * 0.3
                                reasons.append(f"Partial industry match ({industry})")
                                industry_matched = True
                                break
                # No continue — industry is a soft factor, not a hard filter
            
            # --- Application (hard when specified) ---
            if app_tokens:
                app_matched = False
                for canon_app in app_tokens:
                    rec_list = self.APPLICATION_RECOMMENDATIONS.get(canon_app, [])
                    if product.code in rec_list:
                        score += self.W_APPLICATION
                        reasons.append(f"Expert-recommended for {application}")
                        app_matched = True
                        break
                if not app_matched:
                    prod_apps_lower = [a.lower() for a in product.applications]
                    for canon_app in app_tokens:
                        if any(canon_app in pa or pa in canon_app for pa in prod_apps_lower):
                            score += self.W_APPLICATION * 0.6
                            reasons.append(f"Suitable for {application}")
                            app_matched = True
                            break
                    if not app_matched:
                        combined = (product.name + " " + product.description).lower()
                        for canon_app in app_tokens:
                            if canon_app in combined:
                                score += self.W_APPLICATION * 0.3
                                reasons.append(f"Related to {application}")
                                app_matched = True
                                break
                if not app_matched:
                    continue  # Hard filter: no application match → skip
            
            # --- Media (SOFT scoring — boosts rank but never excludes) ---
            if media_families:
                prod_families = self._product_media_families(product)
                overlap = media_families & prod_families
                if overlap:
                    score += self.W_MEDIA
                    reasons.append(f"Media compatible ({', '.join(overlap)})")
                else:
                    # Direct substring fallback
                    if media:
                        combined_media = " ".join(
                            product.service_media + product.applications + product.features
                        ).lower()
                        if media.lower() in combined_media:
                            score += self.W_MEDIA * 0.5
                            reasons.append(f"Media mention found ({media})")
                    # No continue — media is a soft factor, not a hard filter
            
            # =========== SOFT SCORING ===========
            
            # --- Pressure ---
            if operating_pressure is not None and product.specs:
                max_p = max(
                    product.specs.pressure_static or 0,
                    product.specs.pressure_rotary or 0,
                    product.specs.pressure_reciprocating or 0,
                )
                if max_p > 0:
                    if max_p >= operating_pressure:
                        score += self.W_PRESSURE
                        reasons.append(f"Pressure compatible (up to {max_p:.0f} bar)")
                    else:
                        score -= self.W_PRESSURE * 0.3
                        reasons.append(f"Pressure marginal ({max_p:.0f} vs {operating_pressure:.0f} bar)")
            
            # --- Certifications (HARD filter when specific cert requested) ---
            if certifications:
                cert_matched = False
                for cert in certifications:
                    cert_lower = cert.lower().strip()
                    if cert_lower in ("none", "none specific", "not sure", "none required"):
                        cert_matched = True  # No cert required → always passes
                        break
                    # Check product's explicit certifications field
                    if any(cert_lower in pc.lower() or pc.lower() in cert_lower
                           for pc in product.certifications):
                        score += self.W_CERTIFICATION
                        reasons.append(f"Has {cert} certification")
                        cert_matched = True
                    else:
                        # Extract core cert identifier (before parenthetical, e.g.
                        # "API 622 (Fugitive Emissions)" → "api 622")
                        core_match = re.match(r'^([^(]+)', cert_lower)
                        core_cert = core_match.group(1).strip() if core_match else cert_lower
                        core_cert_norm = core_cert.replace("_", " ")
                        
                        # Check expert certification table
                        for ck, products_list in self.CERTIFICATION_PRODUCTS.items():
                            ck_norm = ck.replace("_", " ")
                            # Match core cert ID against table key (not the
                            # parenthetical part which could cause cross-matches)
                            if (ck_norm == core_cert_norm
                                    or ck_norm in core_cert_norm
                                    or core_cert_norm in ck_norm):
                                if product.code in products_list:
                                    score += self.W_CERTIFICATION
                                    reasons.append(f"Has {cert} (expert)")
                                    cert_matched = True
                                    break
                if not cert_matched:
                    continue  # Hard filter: no certification match → skip
            
            # --- Spec-completeness penalty ---
            if not self._product_has_specs(product):
                score += self.PENALTY_NO_SPECS
            
            # --- Only keep products with positive relevance ---
            if score > 0 and reasons:
                confidence = self._calculate_confidence(score, reasons)
                matches.append(ProductMatch(
                    product=product,
                    confidence=confidence,
                    score=round(score, 2),
                    match_reasons=reasons,
                ))
        
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:limit]
    
    # ---------------------------------------------------------------------------
    # Wizard option feasibility checker
    # ---------------------------------------------------------------------------
    # Mapping from wizard step fields to their option labels and the
    # parameter they map to in find_products_strict
    WIZARD_STEPS = [
        {
            "field": "application_type",
            "options": [
                "Valve",
                "Pump",
                "Flange / Gasket Joint",
                "Agitator / Mixer",
                "Compressor / Blower",
                "Boiler / Turbine",
                "Insulation",
                "Other Industrial",
            ],
        },
        {
            "field": "industry",
            "options": [
                "Oil & Gas / Refinery",
                "Chemical / Petrochemical",
                "Power Generation",
                "Pulp & Paper",
                "Cement",
                "Steel",
                "Mining / Minerals",
                "Sugar",
                "Food & Beverage",
                "Pharmaceutical",
                "Other",
            ],
        },
        {
            "field": "temperature_range",
            "options": [
                "Cryogenic (-200°C to -40°C)",
                "Low Temperature (-40°C to 50°C)",
                "Ambient (50°C to 200°C)",
                "High Temperature (200°C to 400°C)",
                "Very High Temperature (400°C+)",
            ],
        },
        {
            "field": "pressure_range",
            "options": [
                "Low (0-10 bar)",
                "Medium (10-50 bar)",
                "High (50-200 bar)",
                "Very High (200+ bar)",
            ],
        },
        {
            "field": "media_type",
            "options": [
                "Steam",
                "Water",
                "Hydrocarbons (Oil/Gas)",
                "Acids / Alkalis",
                "Solvents / Chemicals",
                "Slurries / Abrasives",
                "Other",
            ],
        },
        {
            "field": "required_certifications",
            "options": [
                "API 622 (Fugitive Emissions)",
                "API 589 (Fire Safe)",
                "FDA / Food Grade",
                "None Required",
            ],
        },
    ]
    
    @staticmethod
    def _parse_temp_from_option(option: str) -> Optional[float]:
        """Extract representative operating temp from wizard option text."""
        import re
        m = re.search(r'(-?\d+)\s*°?C?\s*(?:to|-)\s*(-?\d+)', option)
        if m:
            return float(m.group(2))  # Use max temp
        if 'cryogenic' in option.lower():
            return -40.0
        if '400' in option and '+' in option:
            return 500.0
        return None
    
    @staticmethod
    def _parse_pressure_from_option(option: str) -> Optional[float]:
        """Extract representative operating pressure from wizard option text."""
        import re
        m = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)', option)
        if m:
            return float(m.group(2))
        if '200' in option and '+' in option:
            return 300.0
        return None
    
    def _build_search_params_from_wizard(
        self, answers: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert partial wizard answers into search parameters."""
        params: Dict[str, Any] = {}
        
        if answers.get("application_type"):
            params["application"] = answers["application_type"]
        
        if answers.get("industry"):
            ind = answers["industry"]
            if ind != "Other":
                params["industry"] = ind
        
        if answers.get("temperature_range"):
            t = self._parse_temp_from_option(answers["temperature_range"])
            if t is not None:
                params["operating_temp"] = t
        
        if answers.get("pressure_range"):
            p = self._parse_pressure_from_option(answers["pressure_range"])
            if p is not None:
                params["operating_pressure"] = p
        
        if answers.get("media_type"):
            m = answers["media_type"]
            if m != "Other":
                params["media"] = m
        
        if answers.get("required_certifications"):
            certs = answers["required_certifications"]
            if isinstance(certs, list):
                real_certs = [c for c in certs if "none" not in c.lower()]
                if real_certs:
                    params["certifications"] = real_certs
        
        return params
    
    def get_valid_wizard_options(
        self,
        current_answers: Dict[str, Any],
        step_index: int,
    ) -> Dict[str, List[str]]:
        """
        Given the wizard's current answers, return which options for the
        *current* and all *future* steps would still yield at least 1 product.
        
        Returns a dict mapping field name -> list of disabled option labels.
        """
        disabled: Dict[str, List[str]] = {}
        
        # Build base params from already-answered steps
        base_params = self._build_search_params_from_wizard(current_answers)
        
        # For each remaining step (current and future), check each option
        for step in self.WIZARD_STEPS[step_index:]:
            field = step["field"]
            dead_options: List[str] = []
            
            for option in step["options"]:
                # Skip "Other" and "None Required" — these are always fine
                if option in ("Other", "None Required", "Other Industrial"):
                    continue
                
                # Build a hypothetical answer set
                test_answers = dict(current_answers)
                if field == "required_certifications":
                    test_answers[field] = [option]
                else:
                    test_answers[field] = option
                
                test_params = self._build_search_params_from_wizard(test_answers)
                
                # Run strict search with limit=1 — we only need to know if any match
                results = self.find_products_strict(
                    industry=test_params.get("industry"),
                    application=test_params.get("application"),
                    media=test_params.get("media"),
                    operating_temp=test_params.get("operating_temp"),
                    operating_pressure=test_params.get("operating_pressure"),
                    certifications=test_params.get("certifications"),
                    limit=1,
                )
                
                if len(results) == 0:
                    dead_options.append(option)
            
            if dead_options:
                disabled[field] = dead_options
        
        return disabled

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
    
    # Test 1: High temperature valve in power plant
    print("\n--- Test 1: High temperature valve in power plant ---")
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
    
    # Test 2: Pump packing for chemical plant with acid media
    print("\n--- Test 2: Pump packing for chemical plant with acid media ---")
    matches = retriever.find_products(
        industry="chemical",
        application="pump",
        media="acid",
        sealing_type="Packing (Dynamic Seal)",
        operating_temp=150,
    )
    for match in matches:
        print(f"\n{match.product.code}: {match.product.name}")
        print(f"  Score: {match.score}, Confidence: {match.confidence.value}")
        print(f"  Reasons: {', '.join(match.match_reasons)}")
    
    # Test 3: API 622 certified product
    print("\n--- Test 3: API 622 certified packing---")
    matches = retriever.find_products(
        certifications=["API 622"],
        sealing_type="packing",
    )
    for match in matches:
        print(f"\n{match.product.code}: {match.product.name}")
        print(f"  Score: {match.score}, Confidence: {match.confidence.value}")
        print(f"  Reasons: {', '.join(match.match_reasons)}")
