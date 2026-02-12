#!/usr/bin/env python3
"""
Standalone Product Catalog Generator
Creates products_structured.json without requiring full dependencies.

This is a simplified version for Phase 1 of the E1 task.

Usage:
    python scripts/generate_product_catalog_standalone.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_product_code(text: str, url: str = "") -> Optional[str]:
    """Extract product code from text or URL."""
    # Product code pattern
    pattern = re.compile(
        r'\b(NA\s*\d+[A-Z]*(?:\s*[-+]\s*\w+)*|'
        r'PACMAAN[®®]*\s*[ΝN][AΑ]\s*\d+|'
        r'ROLGARD\s*\d+)\b',
        re.IGNORECASE
    )
    
    # Try URL first for product-view pages
    if 'product-view' in url:
        url_parts = url.split('/')
        if url_parts:
            last_part = url_parts[-1]
            match = re.search(r'na[-_]?(\d+[a-z]*)', last_part, re.IGNORECASE)
            if match:
                return f"NA {match.group(1).upper()}"
    
    # Try text content
    match = pattern.search(text)
    if match:
        code = match.group(1)
        code = re.sub(r'\s+', ' ', code.upper())
        code = re.sub(r'NA\s*', 'NA ', code)
        return code.strip()
    
    return None


def extract_specs(content: str) -> Dict[str, Any]:
    """Extract technical specifications."""
    specs = {}
    
    # Temperature
    temp_match = re.search(
        r'temperature[:\s]*(-?\d+)[°º]?C?\s*to\s*(\+?\d+)[°º]?C?',
        content, re.IGNORECASE
    )
    if temp_match:
        specs['temperature_min'] = float(temp_match.group(1))
        specs['temperature_max'] = float(temp_match.group(2))
    
    # Pressure
    pressure_static = re.search(r'(\d+)\s*bar\s*\(static\)', content, re.IGNORECASE)
    if pressure_static:
        specs['pressure_static'] = float(pressure_static.group(1))
    
    return specs


def extract_applications(content: str) -> List[str]:
    """Extract applications."""
    applications = []
    app_match = re.search(
        r'Applications?[:\s]*([^\n]+(?:\n[^\n]+)?)',
        content, re.IGNORECASE
    )
    if app_match:
        app_text = app_match.group(1)
        apps = re.split(r'[,\n]|\band\b', app_text)
        applications = [a.strip() for a in apps if a.strip() and len(a.strip()) > 2]
    return applications[:10]


def extract_certifications(content: str) -> List[str]:
    """Extract certifications."""
    certs = set()
    cert_patterns = [
        (r'API\s*622', 'API 622'),
        (r'API\s*589', 'API 589'),
        (r'API\s*607', 'API 607'),
        (r'ISO\s*9001', 'ISO 9001:2015'),
        (r'ISO\s*15848', 'ISO 15848'),
        (r'FDA', 'FDA Approved'),
    ]
    
    for pattern, cert_name in cert_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            certs.add(cert_name)
    
    return list(certs)


def extract_name(content: str, code: str) -> str:
    """Extract product name."""
    title_match = re.search(r'Title:\s*\r?\n?\s*([^\n]+)', content)
    if title_match:
        title = title_match.group(1).strip()
        title = re.sub(r'\|\s*JD Jones.*$', '', title)
        return title.strip()
    return f"Product {code}"


def extract_description(content: str) -> str:
    """Extract description."""
    desc_match = re.search(r'Description:\s*([^\n]+)', content)
    if desc_match:
        return desc_match.group(1).strip()
    
    paragraphs = content.split('\n')
    for p in paragraphs:
        p = p.strip()
        if len(p) > 50 and not p.startswith('Title:') and not p.startswith('URL:'):
            return p[:300]
    
    return "JD Jones industrial sealing product"


def extract_category(url: str, content: str) -> str:
    """Determine category."""
    category_map = {
        'compression-packing': 'Compression Packing',
        'expanded-ptfe': 'Expanded PTFE Products',
        'graphite': 'Graphite Sealing Products',
        'insulation': 'Insulation',
        'polymer': 'Industrial Polymer Products',
    }
    
    url_lower = url.lower()
    for pattern, category in category_map.items():
        if pattern in url_lower:
            return category
    
    return 'Compression Packing'


def main():
    """Generate products_structured.json."""
    print("=" * 60)
    print("Standalone Product Catalog Generator")
    print("=" * 60)
    print()
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    scraped_file = data_dir / "scraped_jd_jones.json"
    output_file = data_dir / "products_structured.json"
    
    print(f"Looking for scraped data: {scraped_file}")
    
    if not scraped_file.exists():
        print(f"\nERROR: Scraped data file not found!")
        print(f"Expected: {scraped_file}")
        print("\nPlease ensure the scraped data file exists.")
        return 1
    
    # Load scraped data
    print("Loading scraped data...")
    with open(scraped_file, 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)
    
    print(f"Found {len(scraped_data)} documents")
    
    # Process products
    products = {}
    for doc in scraped_data:
        content = doc.get('content', '')
        source_url = doc.get('source', '')
        
        # Only process product pages
        if 'product-view' not in source_url and 'product' not in doc.get('id', ''):
            continue
        
        code = extract_product_code(content, source_url)
        if not code or code in products:
            continue
        
        specs = extract_specs(content)
        
        product = {
            'code': code,
            'name': extract_name(content, code),
            'description': extract_description(content),
            'category': extract_category(source_url, content),
            'material': '',
            'features': [],
            'applications': extract_applications(content),
            'industries': [],
            'service_media': [],
            'certifications': extract_certifications(content),
            'available_forms': [],
            'source_url': source_url,
            'specs': specs
        }
        
        products[code] = product
    
    print(f"\nExtracted {len(products)} unique products")
    
    # Show samples
    print("\nSample products:")
    for i, (code, product) in enumerate(list(products.items())[:5], 1):
        print(f"  {i}. {code}: {product['name']}")
        specs = product.get('specs', {})
        if specs.get('temperature_min') and specs.get('temperature_max'):
            print(f"     Temp: {specs['temperature_min']}°C to {specs['temperature_max']}°C")
    
    # Export
    print(f"\nExporting to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(list(products.values()), f, indent=2, ensure_ascii=False)
    
    file_size = output_file.stat().st_size
    print(f"✓ File created: {file_size:,} bytes")
    print()
    print("=" * 60)
    print("SUCCESS: products_structured.json generated!")
    print("=" * 60)
    print()
    print("Phase 1 Complete!")
    print(f"Total products: {len(products)}")
    print(f"Output file: {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
