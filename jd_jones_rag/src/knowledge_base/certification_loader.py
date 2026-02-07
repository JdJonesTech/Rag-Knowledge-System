"""
Certification Loader
Loads certification standards data from JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Certification:
    """Represents a certification standard."""
    standard: str
    title: str
    organization: str
    description: str
    latest_edition: str
    scope: List[str]
    test_requirements: Dict[str, Any]
    applicable_products: List[str]
    industries: List[str]
    related_standards: List[str]
    key_benefits: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "standard": self.standard,
            "title": self.title,
            "organization": self.organization,
            "description": self.description,
            "latest_edition": self.latest_edition,
            "scope": self.scope,
            "test_requirements": self.test_requirements,
            "applicable_products": self.applicable_products,
            "industries": self.industries,
            "related_standards": self.related_standards,
            "key_benefits": self.key_benefits
        }


class CertificationLoader:
    """Loads and manages certification standards data."""
    
    def __init__(self, certifications_dir: str = "data/certifications"):
        self.certifications_dir = Path(certifications_dir)
        self.certifications: Dict[str, Certification] = {}
        self._load_certifications()
    
    def _load_certifications(self):
        """Load all certification files."""
        if not self.certifications_dir.exists():
            logger.warning(f"Certifications directory not found: {self.certifications_dir}")
            return
        
        for json_file in self.certifications_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cert = Certification(
                    standard=data.get("standard", ""),
                    title=data.get("title", ""),
                    organization=data.get("organization", ""),
                    description=data.get("description", ""),
                    latest_edition=data.get("latest_edition", ""),
                    scope=data.get("scope", []),
                    test_requirements=data.get("test_requirements", {}),
                    applicable_products=data.get("applicable_products", []),
                    industries=data.get("industries", []),
                    related_standards=data.get("related_standards", []),
                    key_benefits=data.get("key_benefits", [])
                )
                
                self.certifications[cert.standard] = cert
                logger.info(f"Loaded certification: {cert.standard}")
                
            except Exception as e:
                logger.error(f"Error loading certification from {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.certifications)} certifications")
    
    def get_certification(self, standard: str) -> Optional[Certification]:
        """Get certification by standard name."""
        # Try exact match first
        if standard in self.certifications:
            return self.certifications[standard]
        
        # Try case-insensitive match
        for key, cert in self.certifications.items():
            if key.lower() == standard.lower():
                return cert
        
        return None
    
    def get_certifications_for_product(self, product_code: str) -> List[Certification]:
        """Get all certifications applicable to a product."""
        applicable = []
        for cert in self.certifications.values():
            if product_code in cert.applicable_products:
                applicable.append(cert)
        return applicable
    
    def get_certifications_for_industry(self, industry: str) -> List[Certification]:
        """Get all certifications relevant to an industry."""
        relevant = []
        industry_lower = industry.lower()
        for cert in self.certifications.values():
            for ind in cert.industries:
                if industry_lower in ind.lower():
                    relevant.append(cert)
                    break
        return relevant
    
    def get_all_certifications(self) -> List[Certification]:
        """Get all loaded certifications."""
        return list(self.certifications.values())
    
    def search_certifications(self, query: str) -> List[Certification]:
        """Search certifications by keyword."""
        query_lower = query.lower()
        results = []
        
        for cert in self.certifications.values():
            # Search in multiple fields
            searchable = " ".join([
                cert.standard,
                cert.title,
                cert.description,
                " ".join(cert.scope),
                " ".join(cert.key_benefits)
            ]).lower()
            
            if query_lower in searchable:
                results.append(cert)
        
        return results
    
    def get_test_requirements(self, standard: str) -> Optional[Dict[str, Any]]:
        """Get test requirements for a specific standard."""
        cert = self.get_certification(standard)
        if cert:
            return cert.test_requirements
        return None
    
    def format_certification_summary(self, standard: str) -> str:
        """Format a human-readable summary of a certification."""
        cert = self.get_certification(standard)
        if not cert:
            return f"Certification {standard} not found."
        
        summary = f"""
**{cert.standard}** - {cert.title}

**Organization:** {cert.organization}
**Edition:** {cert.latest_edition}

**Description:**
{cert.description}

**Scope:**
{chr(10).join('• ' + s for s in cert.scope)}

**Applicable Products:**
{', '.join(cert.applicable_products)}

**Industries:**
{', '.join(cert.industries)}

**Key Benefits:**
{chr(10).join('• ' + b for b in cert.key_benefits)}
"""
        return summary.strip()


# Global instance
_certification_loader: Optional[CertificationLoader] = None


def get_certification_loader() -> CertificationLoader:
    """Get or create global certification loader instance."""
    global _certification_loader
    if _certification_loader is None:
        _certification_loader = CertificationLoader()
    return _certification_loader
