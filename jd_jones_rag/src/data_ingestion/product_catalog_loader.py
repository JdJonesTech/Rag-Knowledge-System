"""
Product Catalog Loader
Parses scraped JD Jones data and creates structured product database.
This bridges the gap between raw scraped data and the RAG system.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ProductSpecs:
    """Technical specifications for a product."""
    temperature_min: Optional[float] = None  # in Celsius
    temperature_max: Optional[float] = None
    pressure_static: Optional[float] = None  # in bar
    pressure_rotary: Optional[float] = None
    pressure_reciprocating: Optional[float] = None
    shaft_speed_rotary: Optional[float] = None  # in m/sec
    shaft_speed_reciprocating: Optional[float] = None
    ph_min: Optional[float] = None
    ph_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Product:
    """Structured product data."""
    code: str
    name: str
    description: str
    category: str
    material: str = ""
    features: List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    service_media: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    specs: Optional[ProductSpecs] = None
    source_url: str = ""
    available_forms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "material": self.material,
            "features": self.features,
            "applications": self.applications,
            "industries": self.industries,
            "service_media": self.service_media,
            "certifications": self.certifications,
            "available_forms": self.available_forms,
            "source_url": self.source_url,
        }
        if self.specs:
            result["specs"] = self.specs.to_dict()
        return result
    
    def to_searchable_text(self) -> str:
        """Generate rich text for vector embedding."""
        parts = [
            f"Product Code: {self.code}",
            f"Product Name: {self.name}",
            f"Category: {self.category}",
            f"Description: {self.description}",
        ]
        
        if self.material:
            parts.append(f"Materials: {self.material}")
        
        if self.features:
            parts.append("Features: " + ", ".join(self.features))
        
        if self.applications:
            parts.append("Applications: " + ", ".join(self.applications))
        
        if self.industries:
            parts.append("Industries: " + ", ".join(self.industries))
        
        if self.service_media:
            parts.append("Service Media: " + ", ".join(self.service_media))
        
        if self.certifications:
            parts.append("Certifications: " + ", ".join(self.certifications))
        
        if self.specs:
            spec_parts = []
            if self.specs.temperature_min is not None and self.specs.temperature_max is not None:
                spec_parts.append(f"Temperature: {self.specs.temperature_min}°C to {self.specs.temperature_max}°C")
            if self.specs.pressure_static:
                spec_parts.append(f"Pressure (static): {self.specs.pressure_static} bar")
            if self.specs.ph_min is not None and self.specs.ph_max is not None:
                spec_parts.append(f"pH: {self.specs.ph_min} to {self.specs.ph_max}")
            if spec_parts:
                parts.append("Specifications: " + "; ".join(spec_parts))
        
        return "\n".join(parts)


class ProductCatalogLoader:
    """Loads and parses JD Jones product data from scraped content."""
    
    # Product code patterns for JD Jones
    PRODUCT_CODE_PATTERN = re.compile(
        r'\b(NA\s*\d+[A-Z]*(?:\s*[-+]\s*\w+)*|'  # NA 701, NA 758H, NA 20747
        r'PACMAAN[®®]*\s*[ΝN][AΑ]\s*\d+|'        # PACMAAN NA 715
        r'ROLGARD\s*\d+)\b',                     # ROLGARD products
        re.IGNORECASE
    )
    
    # Category mapping based on URL patterns
    CATEGORY_MAP = {
        'compression-packing': 'Compression Packing',
        'expanded-ptfe': 'Expanded PTFE Products',
        'ptfe-products': 'Expanded PTFE Products',
        'graphite-sealing': 'Graphite Sealing Products',
        'graphite': 'Graphite Sealing Products',
        'insulation': 'Insulation',
        'ceramic': 'Insulation',
        'industrial-polymer': 'Industrial Polymer Products',
        'polymer': 'Industrial Polymer Products',
        'ptfe-rptfe': 'Industrial Polymer Products',
        'lubricant': 'Specialty Lubricants',
        'low-emission': 'Low Emission Packing',
    }
    
    # Industry keywords
    INDUSTRY_KEYWORDS = {
        'refinery': 'Oil Refineries',
        'petrochemical': 'Petrochemical Plants',
        'power plant': 'Power Plants',
        'power': 'Power Plants',
        'chemical': 'Chemical Plants',
        'steel': 'Steel Plants',
        'cement': 'Cement Plants',
        'pulp': 'Pulp & Paper Mills',
        'paper': 'Pulp & Paper Mills',
        'sugar': 'Sugar Plants',
        'fertilizer': 'Fertilizer Plants',
        'food': 'Food Processing',
        'brewery': 'Breweries & Distilleries',
        'distillery': 'Breweries & Distilleries',
        'paint': 'Paint Manufacturing',
        'mining': 'Mining',
        'mines': 'Mining',
        'valve': 'Valve Manufacturers',
    }
    
    # Certification patterns
    CERTIFICATION_PATTERNS = [
        (r'API\s*622', 'API 622'),
        (r'API\s*589', 'API 589'),
        (r'API\s*607', 'API 607'),
        (r'API\s*624', 'API 624'),
        (r'ISO\s*9001', 'ISO 9001:2015'),
        (r'ISO\s*15848', 'ISO 15848'),
        (r'FDA', 'FDA Approved'),
        (r'food\s*grade', 'Food Grade'),
        (r'Shell\s*approved', 'Shell Approved'),
        (r'Saudi\s*Aramco', 'Saudi Aramco Approved'),
        (r'ADNOC', 'ADNOC Approved'),
    ]
    
    def __init__(self, scraped_data_path: Optional[Path] = None):
        self.scraped_data_path = scraped_data_path or Path(__file__).parent.parent.parent / "data" / "scraped_jd_jones.json"
        self.products: Dict[str, Product] = {}
    
    def load_scraped_data(self) -> List[Dict[str, Any]]:
        """Load the scraped JSON data."""
        if not self.scraped_data_path.exists():
            logger.error(f"Scraped data not found: {self.scraped_data_path}")
            return []
        
        with open(self.scraped_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_product_code(self, text: str, url: str = "") -> Optional[str]:
        """Extract product code from text or URL."""
        # Try URL first for product-view pages
        if 'product-view' in url:
            url_parts = url.split('/')
            if url_parts:
                last_part = url_parts[-1]
                # Extract NA code from URL slug
                match = re.search(r'na[-_]?(\d+[a-z]*)', last_part, re.IGNORECASE)
                if match:
                    return f"NA {match.group(1).upper()}"
        
        # Try text content
        match = self.PRODUCT_CODE_PATTERN.search(text)
        if match:
            code = match.group(1)
            # Normalize code format
            code = re.sub(r'\s+', ' ', code.upper())
            code = re.sub(r'NA\s*', 'NA ', code)
            return code.strip()
        
        return None
    
    def extract_category(self, url: str, content: str) -> str:
        """Determine product category from URL and content."""
        url_lower = url.lower()
        content_lower = content.lower()
        
        for pattern, category in self.CATEGORY_MAP.items():
            if pattern in url_lower or pattern in content_lower:
                return category
        
        # Default based on keywords
        if 'packing' in content_lower:
            return 'Compression Packing'
        if 'gasket' in content_lower:
            return 'Graphite Sealing Products'
        if 'insulation' in content_lower or 'ceramic' in content_lower:
            return 'Insulation'
        
        return 'Compression Packing'  # Default
    
    def extract_specs(self, content: str) -> ProductSpecs:
        """Extract technical specifications from content."""
        specs = ProductSpecs()
        
        # Temperature extraction
        temp_match = re.search(
            r'temperature[:\s]*(-?\d+)[°º]?C?\s*to\s*(\+?\d+)[°º]?C?',
            content, re.IGNORECASE
        )
        if temp_match:
            specs.temperature_min = float(temp_match.group(1))
            specs.temperature_max = float(temp_match.group(2))
        
        # Pressure extraction
        pressure_static = re.search(r'(\d+)\s*bar\s*\(static\)', content, re.IGNORECASE)
        if pressure_static:
            specs.pressure_static = float(pressure_static.group(1))
        
        pressure_rotary = re.search(r'(\d+)\s*bar\s*\(rotary\)', content, re.IGNORECASE)
        if pressure_rotary:
            specs.pressure_rotary = float(pressure_rotary.group(1))
        
        pressure_recip = re.search(r'(\d+)\s*bar\s*\(reciprocating\)', content, re.IGNORECASE)
        if pressure_recip:
            specs.pressure_reciprocating = float(pressure_recip.group(1))
        
        # pH extraction
        ph_match = re.search(r'pH[:\s]*(\d+)\s*to\s*(\d+)', content, re.IGNORECASE)
        if ph_match:
            specs.ph_min = float(ph_match.group(1))
            specs.ph_max = float(ph_match.group(2))
        
        # Shaft speed
        speed_rotary = re.search(r'(\d+)\s*m/sec?\s*\(rotary\)', content, re.IGNORECASE)
        if speed_rotary:
            specs.shaft_speed_rotary = float(speed_rotary.group(1))
        
        return specs
    
    def extract_applications(self, content: str) -> List[str]:
        """Extract applications from content."""
        applications = []
        
        # Look for Applications section
        app_match = re.search(
            r'Applications?[:\s]*([^\n]+(?:\n[^\n]+)?)',
            content, re.IGNORECASE
        )
        if app_match:
            app_text = app_match.group(1)
            # Split by comma, newline, or 'and'
            apps = re.split(r'[,\n]|\band\b', app_text)
            applications = [a.strip() for a in apps if a.strip() and len(a.strip()) > 2]
        
        return applications[:10]  # Limit
    
    def extract_service_media(self, content: str) -> List[str]:
        """Extract service media conditions."""
        media = []
        
        media_match = re.search(
            r'Service Media[^:]*:[:\s]*([^\n]+(?:\n[^\n]+)?)',
            content, re.IGNORECASE
        )
        if media_match:
            media_text = media_match.group(1)
            media_items = re.split(r'[,\n]', media_text)
            media = [m.strip() for m in media_items if m.strip() and len(m.strip()) > 2]
        
        return media[:15]
    
    def extract_industries(self, content: str) -> List[str]:
        """Extract industries from content."""
        industries = set()
        content_lower = content.lower()
        
        for keyword, industry in self.INDUSTRY_KEYWORDS.items():
            if keyword in content_lower:
                industries.add(industry)
        
        return list(industries)
    
    def extract_certifications(self, content: str) -> List[str]:
        """Extract certifications from content."""
        certs = set()
        
        for pattern, cert_name in self.CERTIFICATION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                certs.add(cert_name)
        
        return list(certs)
    
    def extract_features(self, content: str) -> List[str]:
        """Extract product features."""
        features = []
        
        # Look for bullet points or feature-like statements
        feature_patterns = [
            r'(?:^|\n)\s*[-•]\s*([^\n]+)',
            r'Excellent\s+[^\n]+',
            r'High\s+(?:grade|quality|performance)[^\n]+',
            r'Ideal\s+(?:for|packing)[^\n]+',
            r'A\s+(?:superb|special|universal)[^\n]+',
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and len(match) > 10 and len(match) < 200:
                    features.append(match.strip())
        
        return features[:10]
    
    def extract_material(self, code: str, content: str) -> str:
        """Determine primary material from product code and content."""
        materials = []
        content_lower = content.lower()
        
        material_keywords = [
            ('graphite', 'Expanded Graphite'),
            ('ptfe', 'PTFE'),
            ('aramid', 'Aramid Fiber'),
            ('carbon', 'Carbon Fiber'),
            ('ceramic', 'Ceramic Fiber'),
            ('acrylic', 'Acrylic Fiber'),
            ('glass', 'Glass Fiber'),
            ('inconel', 'Inconel Reinforced'),
            ('kynol', 'Kynol Fiber'),
            ('flax', 'Flax'),
        ]
        
        for keyword, material in material_keywords:
            if keyword in content_lower:
                materials.append(material)
        
        return ', '.join(materials[:3]) if materials else 'Various'
    
    def extract_name(self, content: str, code: str) -> str:
        """Extract product name from content."""
        # Look for title pattern
        title_match = re.search(r'Title:\s*\r?\n?\s*([^\n]+)', content)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up title
            title = re.sub(r'\|\s*JD Jones.*$', '', title)
            title = re.sub(r'^\s*Top Quality\s*', '', title, flags=re.IGNORECASE)
            title = re.sub(r'^\s*Best Quality\s*', '', title, flags=re.IGNORECASE)
            return title.strip()
        
        # Try to construct from code
        return f"Product {code}"
    
    def extract_description(self, content: str) -> str:
        """Extract product description."""
        # Look for Description in meta
        desc_match = re.search(r'Description:\s*([^\n]+)', content)
        if desc_match:
            return desc_match.group(1).strip()
        
        # Look for first substantial paragraph
        paragraphs = content.split('\n')
        for p in paragraphs:
            p = p.strip()
            if len(p) > 50 and not p.startswith('Title:') and not p.startswith('URL:'):
                return p[:300]
        
        return "JD Jones industrial sealing product"
    
    def parse_product(self, doc: Dict[str, Any]) -> Optional[Product]:
        """Parse a single document into a Product."""
        content = doc.get('content', '')
        source_url = doc.get('source', '')
        
        # Only process product pages
        if 'product-view' not in source_url and 'product' not in doc.get('id', ''):
            return None
        
        # Extract product code
        code = self.extract_product_code(content, source_url)
        if not code:
            return None
        
        # Skip duplicates
        if code in self.products:
            return None
        
        product = Product(
            code=code,
            name=self.extract_name(content, code),
            description=self.extract_description(content),
            category=self.extract_category(source_url, content),
            material=self.extract_material(code, content),
            features=self.extract_features(content),
            applications=self.extract_applications(content),
            industries=self.extract_industries(content),
            service_media=self.extract_service_media(content),
            certifications=self.extract_certifications(content),
            specs=self.extract_specs(content),
            source_url=source_url,
        )
        
        return product
    
    def load_all_products(self) -> List[Product]:
        """Load and parse all products from scraped data."""
        logger.info("Loading product catalog from scraped data...")
        
        scraped_data = self.load_scraped_data()
        if not scraped_data:
            logger.error("No scraped data found")
            return []
        
        logger.info(f"Processing {len(scraped_data)} scraped documents...")
        
        for doc in scraped_data:
            product = self.parse_product(doc)
            if product:
                self.products[product.code] = product
        
        logger.info(f"Extracted {len(self.products)} unique products")
        return list(self.products.values())
    
    def get_product_by_code(self, code: str) -> Optional[Product]:
        """Get a specific product by code."""
        # Normalize code
        normalized = re.sub(r'\s+', ' ', code.upper()).strip()
        normalized = re.sub(r'^NA\s*', 'NA ', normalized)
        return self.products.get(normalized)
    
    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        industry: Optional[str] = None,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        min_pressure: Optional[float] = None,
    ) -> List[Product]:
        """Search products by various criteria."""
        results = []
        query_lower = query.lower()
        
        for product in self.products.values():
            # Category filter
            if category and category.lower() not in product.category.lower():
                continue
            
            # Industry filter
            if industry and not any(industry.lower() in ind.lower() for ind in product.industries):
                continue
            
            # Temperature filter
            if min_temp is not None and product.specs:
                if product.specs.temperature_max is None or product.specs.temperature_max < min_temp:
                    continue
            
            if max_temp is not None and product.specs:
                if product.specs.temperature_min is None or product.specs.temperature_min > max_temp:
                    continue
            
            # Pressure filter
            if min_pressure is not None and product.specs:
                static = product.specs.pressure_static or 0
                if static < min_pressure:
                    continue
            
            # Text search
            if query:
                searchable = product.to_searchable_text().lower()
                if query_lower not in searchable:
                    continue
            
            results.append(product)
        
        return results
    
    def export_to_json(self, output_path: Optional[Path] = None) -> Path:
        """Export products to a structured JSON file."""
        if output_path is None:
            output_path = self.scraped_data_path.parent / "products_structured.json"
        
        products_data = [p.to_dict() for p in self.products.values()]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(products_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(products_data)} products to {output_path}")
        return output_path


# Singleton instance for global access
_catalog_instance: Optional[ProductCatalogLoader] = None


def get_product_catalog() -> ProductCatalogLoader:
    """Get or create the product catalog singleton."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = ProductCatalogLoader()
        _catalog_instance.load_all_products()
    return _catalog_instance


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)
    
    loader = ProductCatalogLoader()
    products = loader.load_all_products()
    
    print(f"\nLoaded {len(products)} products")
    print("\nSample products:")
    for p in products[:5]:
        print(f"  - {p.code}: {p.name} ({p.category})")
        if p.specs:
            print(f"    Temp: {p.specs.temperature_min}°C to {p.specs.temperature_max}°C")
    
    # Export
    loader.export_to_json()
