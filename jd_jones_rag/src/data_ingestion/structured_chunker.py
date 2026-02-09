"""
Structured Product Chunker
Breaks raw scraped product pages into clean, structured, section-level chunks
with rich metadata for inverted-index-style filtering.

This addresses the problem where raw scraped pages are stored as single huge
blobs mixing navigation HTML, enquiry forms, and actual product content together,
resulting in poor retrieval quality.

Architecture:
- Each product page is split into multiple focused chunks:
  1. Product Overview (code, name, description)
  2. Technical Specifications (temperature, pressure, pH, shaft speed)
  3. Features & Benefits
  4. Applications & Service Media
  5. Certifications & Standards
- Each chunk has rich metadata enabling:
  - Direct product code lookup (inverted index on product_code)
  - Section-type filtering (specs, features, applications, etc.)
  - Category-based navigation (hierarchical index)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Boilerplate patterns to strip from scraped content
BOILERPLATE_PATTERNS = [
    r'Enquiry sent\nThank you for your query.*?(?=\n[A-Z]|\Z)',
    r'Fill in your details.*?(?=\n[A-Z]|\Z)',
    r'Only alphabets.*?(?=\n)',
    r'Please enter a valid.*?(?=\n)',
    r'Enter your company.*?(?=\n)',
    r'Cannot be blank\.?',
    r'Message cannot be blank\.?',
    r'ENQUIRE\s*NOW',
    r'DOWNLOAD',
    r'Configure\s*block',
    r'HOME\nOUR PRODUCTS\n.*?(?=\n×|\n[A-Z]{2}\s+\d)',
    r'×\n',
    r'\nEnquire Now\nDownload PDF\n',
    r'Fill in your details.*?ENQUIRE\s*NOW',
    r'Fill in your details.*?DOWNLOAD',
]


@dataclass
class StructuredChunk:
    """A clean, focused chunk of product information."""
    content: str
    section_type: str  # overview, specs, features, applications, certifications, service_media
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_embedded_text(self) -> str:
        """Generate text optimized for embedding."""
        parts = []
        code = self.metadata.get("product_code", "")
        name = self.metadata.get("product_name", "")
        
        if code:
            parts.append(f"Product Code: {code}")
        if name:
            parts.append(f"Product Name: {name}")
        
        parts.append(f"Section: {self.section_type}")
        parts.append(self.content)
        
        return "\n".join(parts)


class StructuredProductChunker:
    """
    Breaks raw scraped product pages into clean, structured chunks.
    
    This replaces the old approach of storing entire scraped pages as single
    documents, which resulted in poor retrieval because the chunks mixed
    navigation HTML, enquiry forms, and actual content.
    """
    
    def __init__(self):
        self.boilerplate_re = [
            re.compile(p, re.IGNORECASE | re.DOTALL) 
            for p in BOILERPLATE_PATTERNS
        ]
    
    def clean_content(self, raw_content: str) -> str:
        """Strip boilerplate, navigation, and form elements from scraped content."""
        content = raw_content
        
        # Remove boilerplate patterns
        for pattern in self.boilerplate_re:
            content = pattern.sub('', content)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove orphan short lines (likely UI elements)
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep lines that are substantial or are part of specs
            if len(stripped) > 3 or re.match(r'^[-\d+°]', stripped):
                cleaned_lines.append(stripped)
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_product_code(self, content: str, url: str = "") -> Optional[str]:
        """Extract the primary product code."""
        # Try URL first (most reliable for product-view pages)
        if 'product-view' in url:
            slug = url.rsplit('/', 1)[-1] if '/' in url else url
            match = re.search(r'na[-_]?(\d+[a-z]*)', slug, re.IGNORECASE)
            if match:
                return f"NA {match.group(1).upper()}"
        
        # Try content - look for "NA XXX -" pattern (product heading)
        heading = re.search(r'(NA\s*\d+[A-Z]*)\s*[-–]', content)
        if heading:
            code = heading.group(1)
            return re.sub(r'\s+', ' ', code.upper()).strip()
        
        # Fallback to text pattern
        match = re.search(r'\b(NA\s*\d+[A-Z]*)\b', content)
        if match:
            code = match.group(1)
            return re.sub(r'\s+', ' ', code.upper()).strip()
        
        return None
    
    def extract_product_name(self, content: str) -> str:
        """Extract the product name from content."""
        # Look for "NA XXX - Full Product Name" pattern
        name_match = re.search(
            r'NA\s*\d+[A-Z]*\s*[-–]\s*(.+?)(?:\n|$)',
            content
        )
        if name_match:
            return name_match.group(1).strip()
        
        # Fall back to Title
        title_match = re.search(r'Title:\s*\r?\n?\s*(.+?)(?:\n|$)', content)
        if title_match:
            title = title_match.group(1).strip()
            title = re.sub(r'\|\s*JD Jones.*$', '', title).strip()
            return title
        
        return "Unknown Product"
    
    def extract_description(self, content: str) -> str:
        """Extract product description."""
        desc_match = re.search(r'Description:\s*(.+?)(?:\n|URL:)', content, re.DOTALL)
        if desc_match:
            return desc_match.group(1).strip()
        return ""
    
    def extract_specs_section(self, content: str) -> Dict[str, str]:
        """Extract technical specifications as structured key-value pairs."""
        specs = {}
        
        # Temperature
        temp_match = re.search(
            r'[Tt]emperature\s*\n?\s*(-?\d+)\s*°?\s*C?\s*to\s*(\+?\d+)\s*°?\s*C?',
            content
        )
        if temp_match:
            specs["temperature"] = f"{temp_match.group(1)}°C to {temp_match.group(2)}°C"
        
        # Pressure (static)
        p_static = re.search(r'(\d+)\s*bar\s*\(static\)', content, re.IGNORECASE)
        if p_static:
            specs["pressure_static"] = f"{p_static.group(1)} bar (static)"
        
        # Pressure (rotary)
        p_rotary = re.search(r'(\d+)\s*bar\s*\(rotary\)', content, re.IGNORECASE)
        if p_rotary:
            specs["pressure_rotary"] = f"{p_rotary.group(1)} bar (rotary)"
        
        # Pressure (reciprocating)
        p_recip = re.search(r'(\d+)\s*bar\s*\(reciprocating\)', content, re.IGNORECASE)
        if p_recip:
            specs["pressure_reciprocating"] = f"{p_recip.group(1)} bar (reciprocating)"
        
        # Shaft Speed
        speed = re.search(r'(\d+)\s*m/sec?\s*\((?:rotary|reciprocating)\)', content, re.IGNORECASE)
        if speed:
            specs["shaft_speed"] = speed.group(0)
        
        # pH
        ph_match = re.search(r'pH\s*\n?\s*(\d+)\s*to\s*(\d+)', content, re.IGNORECASE)
        if ph_match:
            specs["ph_range"] = f"{ph_match.group(1)} to {ph_match.group(2)}"
        
        return specs
    
    def extract_features(self, content: str) -> List[str]:
        """Extract product features as a list."""
        features = []
        
        # Look for feature-like statements (typically after the product heading)
        # Features are usually the bullet-point-like text between the heading and specs
        lines = content.split('\n')
        in_feature_zone = False
        
        for line in lines:
            stripped = line.strip()
            
            # Start of features: after "NA XXX -" heading line
            if re.match(r'NA\s*\d+', stripped) and '-' in stripped:
                in_feature_zone = True
                continue
            
            # End of features: when we hit "Service Media" or "Temperature"
            if re.match(r'(Service Media|Temperature|Pressure|Shaft Speed|pH|Applications)', stripped, re.IGNORECASE):
                in_feature_zone = False
                continue
            
            if in_feature_zone and len(stripped) > 15:
                # Clean up common patterns
                cleaned = re.sub(r'^[-•*]\s*', '', stripped)
                if cleaned and not re.match(r'(Enquire|Download|Fill|ENQUIRE)', cleaned):
                    features.append(cleaned)
        
        return features[:10]
    
    def extract_service_media(self, content: str) -> str:
        """Extract service media section."""
        media_match = re.search(
            r'Service Media[^:]*:\s*\n?(.*?)(?=\nTemperature|\nPressure|\nShaft|\npH|\nApplications)',
            content, re.DOTALL | re.IGNORECASE
        )
        if media_match:
            return media_match.group(1).strip()
        return ""
    
    def extract_applications(self, content: str) -> str:
        """Extract applications section."""
        app_match = re.search(
            r'Applications?\s*\n(.*?)(?=\nEnquire|\nDownload|\nFill|\n×|\Z)',
            content, re.DOTALL | re.IGNORECASE
        )
        if app_match:
            return app_match.group(1).strip()
        return ""
    
    def extract_certifications(self, content: str) -> List[str]:
        """Extract certifications mentioned in the content."""
        certs = []
        cert_patterns = [
            (r'API\s*622', 'API 622'),
            (r'API\s*589', 'API 589'),
            (r'API\s*607', 'API 607'),
            (r'API\s*624', 'API 624'),
            (r'ISO\s*15848', 'ISO 15848'),
            (r'ISO\s*9001', 'ISO 9001'),
            (r'Shell\s+(?:approved|SPE|MESC)', 'Shell Approved'),
            (r'Saudi\s*Aramco', 'Saudi Aramco Approved'),
            (r'ADNOC', 'ADNOC Approved'),
            (r'FDA', 'FDA Approved'),
            (r'fugitive\s+emissions?\s+tests?', 'Fugitive Emission Tested'),
        ]
        
        for pattern, cert_name in cert_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                certs.append(cert_name)
        
        return certs
    
    def chunk_product_page(
        self,
        raw_content: str,
        source_url: str = "",
        access_level: str = "public",
        category: str = "product"
    ) -> List[StructuredChunk]:
        """
        Break a raw scraped product page into structured, focused chunks.
        
        Returns multiple chunks per product, each with rich metadata.
        """
        # Step 1: Clean the raw content
        cleaned = self.clean_content(raw_content)
        
        # Step 2: Extract product identifiers
        product_code = self.extract_product_code(cleaned, source_url)
        if not product_code:
            return []  # Can't identify product
        
        product_name = self.extract_product_name(cleaned)
        description = self.extract_description(raw_content)  # Use raw for meta description
        
        # Common metadata for all chunks of this product
        base_metadata = {
            "product_code": product_code,
            "product_name": product_name,
            "source_url": source_url,
            "access_level": access_level,
            "category": category,
        }
        
        chunks = []
        
        # Chunk 1: Product Overview
        overview_text = f"{product_name}\n{description}"
        features = self.extract_features(cleaned)
        if features:
            overview_text += "\n\nKey Features:\n" + "\n".join(f"- {f}" for f in features)
        
        chunks.append(StructuredChunk(
            content=overview_text,
            section_type="overview",
            metadata={**base_metadata, "section": "overview"},
        ))
        
        # Chunk 2: Technical Specifications
        specs = self.extract_specs_section(cleaned)
        if specs:
            specs_text = "Technical Specifications:\n"
            specs_text += "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in specs.items())
            
            chunks.append(StructuredChunk(
                content=specs_text,
                section_type="specifications",
                metadata={
                    **base_metadata,
                    "section": "specifications",
                    **{f"spec_{k}": v for k, v in specs.items()},
                },
            ))
        
        # Chunk 3: Service Media & Conditions
        service_media = self.extract_service_media(cleaned)
        if service_media:
            chunks.append(StructuredChunk(
                content=f"Service Media and Conditions:\n{service_media}",
                section_type="service_media",
                metadata={**base_metadata, "section": "service_media"},
            ))
        
        # Chunk 4: Applications
        applications = self.extract_applications(cleaned)
        if applications:
            chunks.append(StructuredChunk(
                content=f"Applications:\n{applications}",
                section_type="applications",
                metadata={**base_metadata, "section": "applications"},
            ))
        
        # Chunk 5: Certifications
        certifications = self.extract_certifications(cleaned)
        if certifications:
            chunks.append(StructuredChunk(
                content="Certifications and Standards:\n" + ", ".join(certifications),
                section_type="certifications",
                metadata={
                    **base_metadata,
                    "section": "certifications",
                    "certifications": ", ".join(certifications),
                },
            ))
        
        # Chunk 6: Full cleaned product page (for general queries)
        # This is a comprehensive chunk with all info, but CLEANED
        full_text = f"""Product Code: {product_code}
Product Name: {product_name}
Description: {description}

"""
        if features:
            full_text += "Features:\n" + "\n".join(f"- {f}" for f in features) + "\n\n"
        if service_media:
            full_text += f"Service Media: {service_media}\n\n"
        if specs:
            full_text += "Specifications:\n" + "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in specs.items()) + "\n\n"
        if applications:
            full_text += f"Applications: {applications}\n\n"
        if certifications:
            full_text += f"Certifications: {', '.join(certifications)}\n"
        
        chunks.append(StructuredChunk(
            content=full_text.strip(),
            section_type="full_product",
            metadata={
                **base_metadata,
                "section": "full_product",
                **({"certifications": ", ".join(certifications)} if certifications else {}),
            },
        ))
        
        logger.debug(
            f"Chunked {product_code} into {len(chunks)} structured chunks"
        )
        
        return chunks
    
    def chunk_scraped_data(
        self,
        scraped_docs: List[Dict[str, Any]]
    ) -> Tuple[List[StructuredChunk], List[Dict[str, Any]]]:
        """
        Process all scraped documents, separating product pages into structured
        chunks and passing through non-product pages.
        
        Returns:
            Tuple of (product_chunks, non_product_docs)
        """
        product_chunks = []
        non_product_docs = []
        
        for doc in scraped_docs:
            source_url = doc.get("source", "")
            doc_id = doc.get("id", "")
            content = doc.get("content", "")
            
            if 'product-view' in source_url or 'product_view' in doc_id:
                chunks = self.chunk_product_page(
                    raw_content=content,
                    source_url=source_url,
                    access_level=doc.get("access_level", "public"),
                )
                product_chunks.extend(chunks)
            else:
                non_product_docs.append(doc)
        
        logger.info(
            f"Structured chunker: {len(product_chunks)} product chunks from "
            f"{sum(1 for d in scraped_docs if 'product-view' in d.get('source', '') or 'product_view' in d.get('id', ''))} "
            f"product pages, {len(non_product_docs)} non-product docs"
        )
        
        return product_chunks, non_product_docs


# Convenience function
def get_structured_chunker() -> StructuredProductChunker:
    """Get a structured product chunker instance."""
    return StructuredProductChunker()
