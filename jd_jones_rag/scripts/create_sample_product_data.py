#!/usr/bin/env python3
"""
Create Sample Product Data
Generates a sample products_structured.json with 104 products for E1 task testing.

This creates realistic sample data when scraped data is not available.
"""

import json
from pathlib import Path
import random


def generate_sample_products(count=104):
    """Generate sample product data."""
    products = []
    
    # Product code prefixes
    codes = [f"NA {700 + i}" for i in range(1, count + 1)]
    
    categories = [
        "Compression Packing",
        "Expanded PTFE Products",
        "Graphite Sealing Products",
        "Insulation",
        "Industrial Polymer Products",
        "Specialty Lubricants"
    ]
    
    applications = [
        "Pumps", "Valves", "Agitators", "Mixers", "Rotary Equipment",
        "Static Sealing", "Dynamic Sealing", "High Pressure Applications"
    ]
    
    industries = [
        "Oil Refineries", "Petrochemical Plants", "Power Plants",
        "Chemical Plants", "Steel Plants", "Pulp & Paper Mills"
    ]
    
    certifications_pool = [
        "API 622", "API 607", "ISO 15848", "FDA Approved",
        "ISO 9001:2015", "TA-Luft", "Shell Approved"
    ]
    
    for i, code in enumerate(codes, 1):
        # Vary completeness to create realistic audit scenarios
        has_temp = random.random() > 0.1  # 90% have temperature
        has_pressure = random.random() > 0.15  # 85% have pressure
        has_apps = random.random() > 0.05  # 95% have applications
        has_desc = random.random() > 0.02  # 98% have description
        has_certs = random.random() > 0.3  # 70% have certifications
        
        product = {
            "code": code,
            "name": f"Product {code} - Industrial Sealing Solution",
            "description": f"High-performance sealing product for industrial applications. Suitable for various operating conditions." if has_desc else "",
            "category": random.choice(categories),
            "material": random.choice(["PTFE", "Graphite", "Aramid Fiber", "Carbon Fiber", "Various"]),
            "features": [
                "High temperature resistance",
                "Chemical resistant",
                "Low friction coefficient"
            ] if random.random() > 0.2 else [],
            "applications": random.sample(applications, k=random.randint(1, 3)) if has_apps else [],
            "industries": random.sample(industries, k=random.randint(1, 2)),
            "service_media": ["Water", "Steam", "Chemicals", "Hydrocarbons"][:random.randint(0, 4)],
            "certifications": random.sample(certifications_pool, k=random.randint(1, 3)) if has_certs else [],
            "available_forms": ["Braided", "Twisted", "Die-formed"][:random.randint(1, 3)],
            "source_url": f"https://www.jdjones.com/product-view/{code.lower().replace(' ', '-')}",
            "specs": {
                "temperature_min": -200 + random.randint(0, 100) if has_temp else None,
                "temperature_max": 200 + random.randint(0, 450) if has_temp else None,
                "pressure_static": random.choice([50, 100, 150, 200, 250, 300]) if has_pressure else None,
                "pressure_rotary": random.choice([20, 40, 60, 80, 100]) if random.random() > 0.5 else None,
                "pressure_reciprocating": random.choice([30, 50, 70, 100]) if random.random() > 0.6 else None,
                "shaft_speed_rotary": random.choice([5, 10, 15, 20]) if random.random() > 0.7 else None,
                "shaft_speed_reciprocating": None,
                "ph_min": random.choice([0, 1, 2, 3]) if random.random() > 0.8 else None,
                "ph_max": random.choice([12, 13, 14]) if random.random() > 0.8 else None
            }
        }
        
        products.append(product)
    
    return products


def main():
    """Generate sample products_structured.json."""
    print("=" * 60)
    print("Sample Product Data Generator")
    print("=" * 60)
    print()
    
    # Generate products
    print("Generating 104 sample products...")
    products = generate_sample_products(104)
    
    # Output path
    data_dir = Path(__file__).parent.parent / "data"
    output_file = data_dir / "products_structured.json"
    
    # Create directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    
    file_size = output_file.stat().st_size
    print(f"✓ File created: {file_size:,} bytes")
    
    # Statistics
    print("\nData Statistics:")
    print(f"  Total products: {len(products)}")
    
    with_temp = sum(1 for p in products if p['specs'].get('temperature_max'))
    with_pressure = sum(1 for p in products if p['specs'].get('pressure_static'))
    with_apps = sum(1 for p in products if p['applications'])
    with_certs = sum(1 for p in products if p['certifications'])
    with_desc = sum(1 for p in products if len(p['description']) > 20)
    
    print(f"  With temperature: {with_temp} ({with_temp/len(products)*100:.1f}%)")
    print(f"  With pressure: {with_pressure} ({with_pressure/len(products)*100:.1f}%)")
    print(f"  With applications: {with_apps} ({with_apps/len(products)*100:.1f}%)")
    print(f"  With certifications: {with_certs} ({with_certs/len(products)*100:.1f}%)")
    print(f"  With description: {with_desc} ({with_desc/len(products)*100:.1f}%)")
    
    print()
    print("=" * 60)
    print("SUCCESS: Sample data generated!")
    print("=" * 60)
    print()
    print("Phase 1 Complete!")
    print(f"Output: {output_file}")
    print("\nReady for Phase 2: Create audit script")
    print()


if __name__ == "__main__":
    main()
