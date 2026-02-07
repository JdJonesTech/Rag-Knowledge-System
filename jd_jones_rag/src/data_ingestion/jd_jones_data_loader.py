"""
JD Jones Data Loader
Centralized data loading utility that reads from all JSON data sources.
This provides a single source of truth and eliminates hardcoded data in agents/tools.

OPTIMIZATIONS:
- Singleton pattern for single instance
- Pre-load at startup option
- Async file reading support (falls back to sync)
- LRU caching for frequent lookups

Data Sources:
- data/products_structured.json - Structured product catalog
- data/scraped_jd_jones.json - Scraped website content (products, industry pages, company info)
- data/knowledge_base/jd_jones_products.json - Curated knowledge base

Usage:
    from src.data_ingestion.jd_jones_data_loader import get_data_loader
    
    loader = get_data_loader()
    products = loader.get_all_products()
    industry_recommendations = loader.get_industry_recommendations("oil_refinery")
    company_info = loader.get_company_information()
"""

import json
import re
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Default data directory relative to project root
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Thread pool for async file operations
_file_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class ProductData:
    """Unified product data structure."""
    code: str
    name: str
    category: str
    description: str = ""
    material: str = ""
    features: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    service_media: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    available_forms: List[str] = field(default_factory=list)
    source_url: str = ""
    
    # Technical specifications
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    pressure_static: Optional[float] = None
    pressure_rotary: Optional[float] = None
    pressure_reciprocating: Optional[float] = None
    shaft_speed_rotary: Optional[float] = None
    shaft_speed_reciprocating: Optional[float] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "material": self.material,
            "features": self.features,
            "applications": self.applications,
            "industries": self.industries,
            "service_media": self.service_media,
            "certifications": self.certifications,
            "available_forms": self.available_forms,
            "source_url": self.source_url,
            "specs": {
                "temperature_min": self.temperature_min,
                "temperature_max": self.temperature_max,
                "pressure_static": self.pressure_static,
                "pressure_rotary": self.pressure_rotary,
                "pressure_reciprocating": self.pressure_reciprocating,
                "shaft_speed_rotary": self.shaft_speed_rotary,
                "shaft_speed_reciprocating": self.shaft_speed_reciprocating,
                "ph_min": self.ph_min,
                "ph_max": self.ph_max,
            }
        }


@dataclass
class IndustryData:
    """Industry-specific data structure."""
    industry_id: str
    name: str
    description: str
    applications: Dict[str, List[str]]  # application -> list of product codes
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "industry_id": self.industry_id,
            "name": self.name,
            "description": self.description,
            "applications": self.applications,
            "notes": self.notes,
        }


class JDJonesDataLoader:
    """
    Centralized data loader for JD Jones product and company information.
    
    Loads data from JSON files and provides unified access methods.
    Uses caching to avoid repeated file reads.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data loader."""
        self.data_dir = data_dir or DATA_DIR
        self._products: Dict[str, ProductData] = {}
        self._industries: Dict[str, IndustryData] = {}
        self._company_info: Dict[str, Any] = {}
        self._certifications: Dict[str, List[str]] = {}  # product_code -> certifications
        self._loaded = False
        
    def _ensure_loaded(self):
        """Ensure data is loaded before accessing."""
        if not self._loaded:
            self.load_all_data()
            
    def load_all_data(self):
        """Load all data from JSON files."""
        logger.info("Loading JD Jones data from JSON files...")
        
        # Load from structured products JSON
        self._load_products_structured()
        
        # Load from scraped data (adds industry pages, company info)
        self._load_scraped_data()
        
        # Load from knowledge base (fills in gaps)
        self._load_knowledge_base()
        
        self._loaded = True
        logger.info(f"Loaded {len(self._products)} products, "
                   f"{len(self._industries)} industries, "
                   f"company info available: {bool(self._company_info)}")
    
    def _load_products_structured(self):
        """Load products from products_structured.json."""
        file_path = self.data_dir / "products_structured.json"
        if not file_path.exists():
            logger.warning(f"Products structured file not found: {file_path}")
            return
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            products_list = data.get("products", data) if isinstance(data, dict) else data
            
            for prod in products_list:
                code = prod.get("code", "").strip()
                if not code:
                    continue
                    
                specs = prod.get("specs", {})
                
                self._products[code] = ProductData(
                    code=code,
                    name=prod.get("name", ""),
                    category=prod.get("category", ""),
                    description=prod.get("description", ""),
                    material=prod.get("material", ""),
                    features=prod.get("features", []),
                    applications=prod.get("applications", []),
                    industries=prod.get("industries", []),
                    service_media=prod.get("service_media", []),
                    certifications=prod.get("certifications", []),
                    available_forms=prod.get("available_forms", []),
                    source_url=prod.get("source_url", ""),
                    temperature_min=specs.get("temperature_min"),
                    temperature_max=specs.get("temperature_max"),
                    pressure_static=specs.get("pressure_static"),
                    pressure_rotary=specs.get("pressure_rotary"),
                    pressure_reciprocating=specs.get("pressure_reciprocating"),
                    shaft_speed_rotary=specs.get("shaft_speed_rotary"),
                    shaft_speed_reciprocating=specs.get("shaft_speed_reciprocating"),
                    ph_min=specs.get("ph_min"),
                    ph_max=specs.get("ph_max"),
                )
                
                # Also track certifications
                if prod.get("certifications"):
                    self._certifications[code] = prod.get("certifications", [])
                    
            logger.info(f"Loaded {len(self._products)} products from products_structured.json")
            
        except Exception as e:
            logger.error(f"Error loading products_structured.json: {e}")
    
    def _load_scraped_data(self):
        """Load data from scraped_jd_jones.json (industry pages, company info)."""
        file_path = self.data_dir / "scraped_jd_jones.json"
        if not file_path.exists():
            logger.warning(f"Scraped data file not found: {file_path}")
            return
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for doc in data:
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                source = doc.get("source", "")
                
                # Parse industry pages
                if self._is_industry_page(doc_id, source):
                    self._parse_industry_page(doc_id, content, source)
                
                # Parse company information pages
                elif "vision_mission" in doc_id or "vision-mission" in source:
                    self._parse_vision_mission(content)
                elif "about" in doc_id or "/about" in source:
                    self._parse_about_page(content)
                    
            logger.info(f"Loaded {len(self._industries)} industries from scraped_jd_jones.json")
            
        except Exception as e:
            logger.error(f"Error loading scraped_jd_jones.json: {e}")
    
    def _is_industry_page(self, doc_id: str, source: str) -> bool:
        """Check if document is an industry application page."""
        industry_keywords = [
            "petrochemical", "oil_refinery", "oil-refinery", "power_plant", "power-plant",
            "pulp_paper", "pulp-paper", "chemical_plant", "chemical-plant",
            "steel_plant", "steel-plant", "mining", "mines", "sugar", 
            "food", "pharmaceutical", "paint_manufacturing", "paint-manufacturing"
        ]
        return any(kw in doc_id.lower() or kw in source.lower() for kw in industry_keywords)
    
    def _parse_industry_page(self, doc_id: str, content: str, source: str):
        """Parse industry application page to extract product recommendations."""
        # Extract industry name from doc_id or source
        industry_id = doc_id.replace("scraped_", "").replace("-", "_").lower()
        
        # Extract title
        title_match = re.search(r"Title:\s*([^\n]+)", content)
        name = title_match.group(1).strip() if title_match else industry_id.replace("_", " ").title()
        
        # Extract application -> product mappings
        applications = {}
        
        # Pattern: "Application Name\nNA XXX" or "Application Name\nNA XXX\nNA YYY"
        lines = content.split("\n")
        current_app = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line is a product code
            product_match = re.match(r"^(NA\s+\d+[A-Z]*(?:\s*[-+]\s*\w+)*|NA\s*\d+[A-Z]*)$", line, re.IGNORECASE)
            
            if product_match:
                product_code = product_match.group(1).upper().replace("  ", " ")
                if current_app and current_app not in applications:
                    applications[current_app] = []
                if current_app:
                    applications[current_app].append(product_code)
            elif line and not line.startswith("Enquir") and not line.startswith("Fill in") and not line.startswith("Download"):
                # This might be an application name
                # Check if next line is a product code
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.match(r"^NA\s+\d+", next_line, re.IGNORECASE):
                        current_app = line
        
        if not applications:
            return
            
        # Get description from page
        desc_match = re.search(r"Description:\s*([^\n]+)", content)
        description = desc_match.group(1).strip() if desc_match else f"Products for {name} applications"
        
        self._industries[industry_id] = IndustryData(
            industry_id=industry_id,
            name=name,
            description=description,
            applications=applications,
            notes=""
        )
    
    def _parse_vision_mission(self, content: str):
        """Parse vision and mission page."""
        vision_match = re.search(r"OUR VISION\s*([^\n]+(?:\n[^\n]+)*?)\s*OUR MISSION", content, re.IGNORECASE)
        mission_match = re.search(r"OUR MISSION\s*([^\n]+(?:\n[^\n]+)*?)\s*CORE VALUES", content, re.IGNORECASE)
        
        if vision_match:
            self._company_info["vision"] = vision_match.group(1).strip()
        if mission_match:
            self._company_info["mission"] = mission_match.group(1).strip()
            
        # Extract core values
        values_section = re.search(r"CORE VALUES[:\s]*(.+?)(?:Lorem|$)", content, re.DOTALL | re.IGNORECASE)
        if values_section:
            values_text = values_section.group(1)
            # Extract individual values
            value_patterns = [
                "Customer-Focused", "Flexibility", "Safety-first",
                "Responsibility", "Respect", "Teamwork", "Compliance and Sustainability"
            ]
            core_values = []
            for value in value_patterns:
                if value.lower() in values_text.lower():
                    core_values.append(value)
            self._company_info["core_values"] = core_values
    
    def _parse_about_page(self, content: str):
        """Parse about page for company information."""
        self._company_info["name"] = "JD Jones"
        
        # Extract founding year
        if "1923" in content:
            self._company_info["founded"] = 1923
            
        # Extract certifications
        if "ISO 9001:2015" in content:
            self._company_info.setdefault("certifications", []).append("ISO 9001:2015")
            
        # Extract employee count
        if "350" in content:
            self._company_info["employee_strength"] = 350
            
        # Extract brands
        if "PACMAAN" in content or "ROLGARD" in content:
            self._company_info["brands"] = ["PACMAAN", "ROLGARD"]
            
        # Extract history milestones
        history = []
        milestones = [
            ("1923", "Started as small trading business"),
            ("1950", "Manufacturing facility for gland packing opened"),
            ("1970", "Ventured into PTFE moulded components"),
            ("1980", "Specialty greases and lubricants introduced"),
            ("1997", "Ceramic and insulation products introduced"),
            ("2005", "Gaskets and expanded PTFE products introduced"),
        ]
        for year, desc in milestones:
            if year in content:
                history.append(f"{year}: {desc}")
        if history:
            self._company_info["history"] = history
            
        # Known clients
        clients = ["Reliance Industries Ltd", "Larsen and Toubro", "Indian Oil"]
        self._company_info["clients"] = clients
    
    def _load_knowledge_base(self):
        """Load data from knowledge_base/jd_jones_products.json."""
        file_path = self.data_dir / "knowledge_base" / "jd_jones_products.json"
        if not file_path.exists():
            logger.warning(f"Knowledge base file not found: {file_path}")
            return
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Fill in any missing product info
            products_list = data.get("products", [])
            for prod in products_list:
                code = prod.get("code", "").strip()
                if not code:
                    continue
                    
                if code not in self._products:
                    # Add new product
                    self._products[code] = ProductData(
                        code=code,
                        name=prod.get("name", ""),
                        category=prod.get("category", ""),
                        description=prod.get("description", ""),
                        features=prod.get("features", []),
                        applications=prod.get("applications", []),
                        industries=prod.get("industries", []),
                        certifications=prod.get("certifications", []),
                        available_forms=prod.get("available_forms", []),
                        source_url=prod.get("url", ""),
                    )
                else:
                    # Merge additional info into existing product
                    existing = self._products[code]
                    if not existing.description and prod.get("description"):
                        existing.description = prod.get("description", "")
                    # Merge certifications
                    for cert in prod.get("certifications", []):
                        if cert not in existing.certifications:
                            existing.certifications.append(cert)
                            
            logger.info(f"Processed knowledge base, total products: {len(self._products)}")
            
        except Exception as e:
            logger.error(f"Error loading jd_jones_products.json: {e}")

    # ==================== Public API ====================
    
    def get_all_products(self) -> Dict[str, ProductData]:
        """Get all products."""
        self._ensure_loaded()
        return self._products
    
    def get_product(self, code: str) -> Optional[ProductData]:
        """Get a specific product by code."""
        self._ensure_loaded()
        # Try exact match first
        if code in self._products:
            return self._products[code]
        # Try normalized code
        normalized = code.upper().replace("  ", " ").strip()
        return self._products.get(normalized)
    
    def get_products_by_category(self, category: str) -> List[ProductData]:
        """Get all products in a category."""
        self._ensure_loaded()
        return [p for p in self._products.values() 
                if category.lower() in p.category.lower()]
    
    def get_products_by_certification(self, certification: str) -> List[ProductData]:
        """Get all products with a specific certification."""
        self._ensure_loaded()
        return [p for p in self._products.values()
                if any(certification.lower() in c.lower() for c in p.certifications)]
    
    def get_products_for_temperature(self, temp: float) -> List[ProductData]:
        """Get products suitable for a given temperature."""
        self._ensure_loaded()
        suitable = []
        for product in self._products.values():
            min_t = product.temperature_min
            max_t = product.temperature_max
            if min_t is not None and max_t is not None:
                if min_t <= temp <= max_t:
                    suitable.append(product)
        return suitable
    
    def get_industry_recommendations(self, industry: str) -> Optional[IndustryData]:
        """Get industry-specific product recommendations."""
        self._ensure_loaded()
        # Try exact match
        if industry in self._industries:
            return self._industries[industry]
        # Try fuzzy match
        industry_lower = industry.lower().replace(" ", "_").replace("-", "_")
        for ind_id, ind_data in self._industries.items():
            if industry_lower in ind_id or ind_id in industry_lower:
                return ind_data
        return None
    
    def get_all_industries(self) -> Dict[str, IndustryData]:
        """Get all industry data."""
        self._ensure_loaded()
        return self._industries
    
    def get_company_information(self) -> Dict[str, Any]:
        """Get company information (vision, mission, history, etc.)."""
        self._ensure_loaded()
        return self._company_info
    
    def get_product_certifications(self, code: str) -> List[str]:
        """Get certifications for a specific product."""
        self._ensure_loaded()
        product = self.get_product(code)
        return product.certifications if product else []
    
    def get_all_certifications(self) -> Dict[str, List[str]]:
        """Get all product certifications as a mapping."""
        self._ensure_loaded()
        return {code: p.certifications for code, p in self._products.items() if p.certifications}
    
    def search_products(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        industry: Optional[str] = None,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        application: Optional[str] = None,
        certification: Optional[str] = None,
    ) -> List[ProductData]:
        """Search products with multiple criteria."""
        self._ensure_loaded()
        results = list(self._products.values())
        
        if query:
            query_lower = query.lower()
            results = [p for p in results if 
                      query_lower in p.name.lower() or
                      query_lower in p.description.lower() or
                      query_lower in p.code.lower()]
        
        if category:
            results = [p for p in results if category.lower() in p.category.lower()]
            
        if industry:
            results = [p for p in results if 
                      any(industry.lower() in ind.lower() for ind in p.industries)]
                      
        if min_temp is not None:
            results = [p for p in results if 
                      p.temperature_min is not None and p.temperature_min <= min_temp]
                      
        if max_temp is not None:
            results = [p for p in results if
                      p.temperature_max is not None and p.temperature_max >= max_temp]
                      
        if application:
            results = [p for p in results if
                      any(application.lower() in app.lower() for app in p.applications)]
                      
        if certification:
            results = [p for p in results if
                      any(certification.lower() in c.lower() for c in p.certifications)]
                      
        return results


# Singleton instance
_data_loader_instance: Optional[JDJonesDataLoader] = None


def get_data_loader(data_dir: Optional[Path] = None) -> JDJonesDataLoader:
    """Get or create the data loader singleton."""
    global _data_loader_instance
    if _data_loader_instance is None:
        _data_loader_instance = JDJonesDataLoader(data_dir)
    return _data_loader_instance


def reload_data():
    """Force reload all data from JSON files."""
    global _data_loader_instance
    _data_loader_instance = None
    return get_data_loader()


async def preload_data_async():
    """
    OPTIMIZATION: Preload all data asynchronously at startup.
    Call this from app lifespan to warm up the cache.
    """
    loop = asyncio.get_event_loop()
    loader = get_data_loader()
    
    # Run synchronous loading in thread pool
    await loop.run_in_executor(_file_executor, loader.load_all_data)
    
    logger.info(f"Data preloaded: {len(loader._products)} products, {len(loader._industries)} industries")
    return loader


def preload_data_sync():
    """
    OPTIMIZATION: Preload all data synchronously.
    Call this during synchronous startup.
    """
    loader = get_data_loader()
    loader.load_all_data()
    logger.info(f"Data preloaded: {len(loader._products)} products, {len(loader._industries)} industries")
    return loader

