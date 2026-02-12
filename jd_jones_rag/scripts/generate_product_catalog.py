#!/usr/bin/env python3
"""
Generate Product Catalog Script
Creates products_structured.json from the ProductCatalogLoader.

This script is used for Phase 1 of the E1 task to generate the structured
product catalog file that will be audited.

Usage:
    python scripts/generate_product_catalog.py
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.product_catalog_loader import ProductCatalogLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate the products_structured.json file."""
    print("=" * 60)
    print("Product Catalog Generation Script")
    print("=" * 60)
    print()
    
    # Initialize the loader
    logger.info("Initializing ProductCatalogLoader...")
    loader = ProductCatalogLoader()
    
    # Check if scraped data exists
    scraped_data_path = loader.scraped_data_path
    logger.info(f"Looking for scraped data at: {scraped_data_path}")
    
    if not scraped_data_path.exists():
        logger.error(f"Scraped data file not found: {scraped_data_path}")
        logger.error("Please ensure data/scraped_jd_jones.json exists before running this script.")
        print()
        print("ERROR: Scraped data file not found!")
        print(f"Expected location: {scraped_data_path}")
        print()
        print("This file should contain the scraped product data from the JD Jones website.")
        print("Please check if the file exists or needs to be generated first.")
        sys.exit(1)
    
    # Load all products
    logger.info("Loading products from scraped data...")
    products = loader.load_all_products()
    
    if not products:
        logger.error("No products were loaded from the scraped data.")
        sys.exit(1)
    
    print(f"\n✓ Successfully loaded {len(products)} products")
    
    # Show sample products
    print("\nSample products:")
    for i, product in enumerate(products[:5], 1):
        print(f"  {i}. {product.code}: {product.name}")
        if product.specs:
            temp_range = f"{product.specs.temperature_min}°C to {product.specs.temperature_max}°C" if product.specs.temperature_min and product.specs.temperature_max else "N/A"
            print(f"     Category: {product.category}, Temp: {temp_range}")
    
    # Export to JSON
    logger.info("Exporting products to products_structured.json...")
    output_path = loader.export_to_json()
    
    print(f"\n✓ Successfully exported to: {output_path}")
    
    # Verify the file was created
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"  File size: {file_size:,} bytes")
        
        # Load and verify JSON structure
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  Products in file: {len(data)}")
        print()
        print("=" * 60)
        print("SUCCESS: products_structured.json has been generated!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Review the generated file at:", output_path)
        print("  2. Proceed to Phase 2: Create the audit script")
        print()
    else:
        logger.error("Failed to create the output file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
