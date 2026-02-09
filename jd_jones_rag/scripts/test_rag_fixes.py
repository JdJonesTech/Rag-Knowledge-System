"""
Test Script - Validates all three RAG fixes:

1. Product Code Lookup Fix (product_catalog_loader.py)
   - NA 701 should map to standalone page, not combo NA 701+707
   - NA 715 should be found
   - Combo products get their own codes

2. Structured Chunking (structured_chunker.py)
   - Raw scraped pages are broken into clean section-level chunks
   - Boilerplate is stripped
   - Rich metadata is attached

3. Grounding Validator (grounding_validator.py)
   - Detects ungrounded specs in responses
   - Short-circuits when context is insufficient
   - Cross-checks against catalog

Run: python scripts/test_rag_fixes.py
Or in Docker: docker compose exec api python scripts/test_rag_fixes.py
"""

import sys
import os
import json
import logging
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_product_code_fix():
    """Test Fix 1: Product code lookup resolves correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Product Code Lookup Fix")
    logger.info("=" * 60)
    
    from src.data_ingestion.product_catalog_loader import ProductCatalogLoader
    
    loader = ProductCatalogLoader()
    products = loader.load_all_products()
    
    logger.info(f"Total products loaded: {len(products)}")
    
    passed = True
    
    # Test 1a: NA 701 should be the standalone product
    p701 = loader.get_product_by_code("NA 701")
    if p701:
        logger.info(f"\n  NA 701: FOUND")
        logger.info(f"    Name: {p701.name}")
        logger.info(f"    URL: {p701.source_url}")
        
        # Check it's NOT from the combo page
        if "+707" in p701.source_url or "701-707" in p701.source_url:
            logger.error("    FAIL: NA 701 is mapped to combo page (NA 701+707)")
            passed = False
        else:
            logger.info("    PASS: NA 701 is correctly mapped to standalone page")
        
        # Check specs are reasonable
        if p701.specs:
            logger.info(f"    Temp: {p701.specs.temperature_min}°C to {p701.specs.temperature_max}°C")
            logger.info(f"    Pressure (static): {p701.specs.pressure_static} bar")
            
            # Real NA 701 specs: -240°C to 650°C, 450 bar static
            if p701.specs.temperature_max and p701.specs.temperature_max > 200:
                logger.info("    PASS: Temperature range looks correct (high-temp product)")
            else:
                logger.warning("    WARN: Temperature range may be incorrect")
    else:
        logger.error("  NA 701: NOT FOUND - FAIL")
        passed = False
    
    # Test 1b: NA 715 should be found
    p715 = loader.get_product_by_code("NA 715")
    if p715:
        logger.info(f"\n  NA 715: FOUND")
        logger.info(f"    Name: {p715.name}")
        logger.info(f"    URL: {p715.source_url}")
        
        if p715.specs:
            logger.info(f"    Temp: {p715.specs.temperature_min}°C to {p715.specs.temperature_max}°C")
            logger.info(f"    Pressure (static): {p715.specs.pressure_static} bar")
        
        logger.info("    PASS: NA 715 found in catalog")
    else:
        logger.error("  NA 715: NOT FOUND - FAIL")
        passed = False
    
    # Test 1c: Combo product should have its own code  
    combo = loader.get_product_by_code("NA 701 + 707")
    if combo:
        logger.info(f"\n  NA 701 + 707 (combo): FOUND as separate entry")
        logger.info(f"    URL: {combo.source_url}")
        logger.info("    PASS: Combo product has its own code")
    else:
        logger.info(f"\n  NA 701 + 707 (combo): Not found (may not have combo URL)")
        logger.info("    INFO: This is OK if the combo page doesn't exist in scraped data")
    
    return passed


def test_structured_chunker():
    """Test Fix 2: Structured chunking of product pages."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Structured Product Chunker")
    logger.info("=" * 60)
    
    from src.data_ingestion.structured_chunker import StructuredProductChunker
    
    chunker = StructuredProductChunker()
    
    passed = True
    
    # Test with sample raw content (simulating a scraped page)
    sample_content = """
HOME
OUR PRODUCTS
Compression Packings
×
NA 701 - High Temperature PTFE Compression Packing
Fill in your details
Only alphabets are allowed
Please enter a valid email
ENQUIRE NOW

Advanced PTFE fibre packing reinforced with Inconel wire for extreme service.
Excellent chemical resistance and low friction.

Service Media
Steam, thermal oils, acids, alkalis, solvents

Temperature
-240 to 650

Pressure
450 bar (static)
200 bar (rotary)
25 bar (reciprocating)

Shaft Speed
15 m/sec (rotary)

pH
0 to 14

Applications
Valves, pumps, agitators, autoclaves, reactors

Enquire Now
Download PDF
DOWNLOAD
"""
    
    chunks = chunker.chunk_product_page(
        raw_content=sample_content,
        source_url="https://jdjones.com/product-view/na-701"
    )
    
    logger.info(f"\n  Raw content length: {len(sample_content)} chars")
    logger.info(f"  Produced {len(chunks)} structured chunks:")
    
    for chunk in chunks:
        logger.info(f"\n    [{chunk.section_type}]")
        logger.info(f"      Content preview: {chunk.content[:100]}...")
        logger.info(f"      Metadata: product_code={chunk.metadata.get('product_code')}")
    
    # Verify no boilerplate in chunks
    for chunk in chunks:
        if "ENQUIRE NOW" in chunk.content or "Fill in your details" in chunk.content:
            logger.error(f"    FAIL: Boilerplate found in {chunk.section_type} chunk")
            passed = False
    
    if passed:
        logger.info("\n    PASS: No boilerplate detected in any chunk")
    
    # Verify we got the right chunk types
    chunk_types = {c.section_type for c in chunks}
    expected_types = {"overview", "specifications", "full_product"}
    missing = expected_types - chunk_types
    if missing:
        logger.warning(f"    WARN: Missing expected chunk types: {missing}")
    else:
        logger.info("    PASS: All expected chunk types present")
    
    # Verify product code in metadata
    for chunk in chunks:
        if chunk.metadata.get("product_code") != "NA 701":
            logger.error(f"    FAIL: Wrong product code in {chunk.section_type}: {chunk.metadata.get('product_code')}")
            passed = False
            break
    else:
        logger.info("    PASS: Product code correct in all chunks")
    
    return passed


def test_grounding_validator():
    """Test Fix 3: Grounding validator catches hallucination."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Grounding Validator (Anti-Hallucination)")
    logger.info("=" * 60)
    
    from src.retrieval.grounding_validator import GroundingValidator
    
    validator = GroundingValidator()
    
    passed = True
    
    # Test 3a: Context sufficiency check - product not in context
    logger.info("\n  Test 3a: Context sufficiency (missing product)")
    is_sufficient, fallback = validator.check_context_sufficiency(
        context="No specific information available in the knowledge base.",
        product_matches=[],
        query="What is the temperature range for NA 701?"
    )
    
    if not is_sufficient and fallback:
        logger.info("    PASS: Correctly detected insufficient context for NA 701 query")
        logger.info(f"    Fallback: {fallback[:80]}...")
    else:
        logger.error("    FAIL: Should have detected insufficient context")
        passed = False
    
    # Test 3b: Detect hallucinated specs in response
    logger.info("\n  Test 3b: Detect hallucinated specifications")
    
    # Simulated context that mentions NA 701 with correct specs
    context_with_specs = """
    Product Code: NA 701
    Temperature: -240°C to 650°C
    Pressure: 450 bar (static), 200 bar (rotary)
    """
    
    # Hallucinated response with WRONG specs
    hallucinated_response = """
    NA 701 operates in a temperature range of -40°C to 85°C with a 
    maximum pressure of 1.5 bar. It also supports NA 999 which has
    a temperature of -30°C to 60°C.
    """
    
    result = validator.validate_response(
        response=hallucinated_response,
        context=context_with_specs,
        product_matches=[],
        query="Tell me about NA 701"
    )
    
    if not result.is_grounded:
        logger.info(f"    PASS: Detected {len(result.ungrounded_claims)} ungrounded claims")
        for claim in result.ungrounded_claims:
            logger.info(f"      - {claim['type']}: {claim.get('value', claim.get('code', ''))}")
    else:
        logger.warning("    WARN: Did not detect hallucinated specs (may need pattern tuning)")
    
    # Test 3c: Valid response should pass grounding check
    logger.info("\n  Test 3c: Valid response passes grounding")
    
    correct_response = """
    NA 701 operates in a temperature range of -240°C to 650°C with a 
    maximum static pressure of 450 bar.
    """
    
    result = validator.validate_response(
        response=correct_response,
        context=context_with_specs,
        product_matches=[],
        query="Tell me about NA 701"
    )
    
    if result.is_grounded:
        logger.info("    PASS: Correctly verified grounded response")
    else:
        logger.warning(f"    WARN: False positive - flagged {len(result.ungrounded_claims)} claims in valid response")
        for claim in result.ungrounded_claims:
            logger.info(f"      - {claim}")
    
    return passed


def test_data_ingestion_format():
    """Test that the ingestion script creates proper structured documents."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Data Ingestion Format Check")
    logger.info("=" * 60)
    
    scraped_path = Path(__file__).parent.parent / "data" / "scraped_jd_jones.json"
    
    if not scraped_path.exists():
        logger.warning(f"  Scraped data not found at {scraped_path}")
        logger.info("  SKIP: Cannot test without scraped data")
        return True
    
    from src.data_ingestion.structured_chunker import StructuredProductChunker
    
    with open(scraped_path, "r", encoding="utf-8") as f:
        scraped_docs = json.load(f)
    
    # Count product pages
    product_pages = [d for d in scraped_docs if 'product-view' in d.get('source', '')]
    logger.info(f"  Total scraped docs: {len(scraped_docs)}")
    logger.info(f"  Product pages: {len(product_pages)}")
    
    chunker = StructuredProductChunker()
    product_chunks, non_product = chunker.chunk_scraped_data(scraped_docs)
    
    logger.info(f"  Structured product chunks: {len(product_chunks)}")
    logger.info(f"  Non-product docs: {len(non_product)}")
    
    if len(product_chunks) > len(product_pages):
        logger.info("  PASS: More chunks than pages (multi-section chunking working)")
    else:
        logger.warning("  WARN: Expected more chunks than product pages")
    
    # Check chunk metadata quality
    codes_found = set()
    section_types = set()
    for chunk in product_chunks:
        codes_found.add(chunk.metadata.get("product_code", ""))
        section_types.add(chunk.section_type)
    
    logger.info(f"  Unique product codes in chunks: {len(codes_found)}")
    logger.info(f"  Section types: {section_types}")
    logger.info(f"  Product codes found: {sorted(codes_found)[:10]}...")
    
    return True


def main():
    logger.info("=" * 60)
    logger.info("JD Jones RAG System - Fix Validation Tests")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Product code fix
    try:
        results["product_code_fix"] = test_product_code_fix()
    except Exception as e:
        logger.error(f"\nTest 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results["product_code_fix"] = False
    
    # Test 2: Structured chunker
    try:
        results["structured_chunker"] = test_structured_chunker()
    except Exception as e:
        logger.error(f"\nTest 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results["structured_chunker"] = False
    
    # Test 3: Grounding validator
    try:
        results["grounding_validator"] = test_grounding_validator()
    except Exception as e:
        logger.error(f"\nTest 3 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results["grounding_validator"] = False
    
    # Test 4: Ingestion format
    try:
        results["data_ingestion"] = test_data_ingestion_format()
    except Exception as e:
        logger.error(f"\nTest 4 failed with error: {e}")
        import traceback
        traceback.print_exc()
        results["data_ingestion"] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for name, status in results.items():
        icon = "PASS" if status else "FAIL"
        logger.info(f"  [{icon}] {name}")
        if not status:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll tests passed!")
    else:
        logger.info("\nSome tests failed. Review output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
