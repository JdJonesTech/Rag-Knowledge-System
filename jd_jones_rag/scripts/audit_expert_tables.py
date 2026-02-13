import json
import os
import sys
import ast
from typing import List, Dict, Set, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_product_catalog(filepath: Path) -> List[Dict]:
    """Load the structured product catalog JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Product catalog not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in product catalog at {filepath}")
        sys.exit(1)

def extract_expert_tables_from_source(filepath: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse the source code of ProductCatalogRetriever to extract
    APPLICATION_RECOMMENDATIONS and INDUSTRY_RECOMMENDATIONS
    without importing the module (avoiding dependencies).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Error parsing source file {filepath}: {e}")
        sys.exit(1)

    extracted_data = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ProductCatalogRetriever':
            for item in node.body:
                # Handle standard assignments: vars = ...
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            if target.id in ['APPLICATION_RECOMMENDATIONS', 'INDUSTRY_RECOMMENDATIONS']:
                                try:
                                    val = ast.literal_eval(item.value)
                                    extracted_data[target.id] = val
                                except Exception as e:
                                    print(f"Failed to evaluate {target.id}: {e}")

                # Handle annotated assignments: vars: Type = ...
                elif isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                         if item.target.id in ['APPLICATION_RECOMMENDATIONS', 'INDUSTRY_RECOMMENDATIONS']:
                            if item.value: # Ensure there is a value assigned
                                try:
                                    val = ast.literal_eval(item.value)
                                    extracted_data[item.target.id] = val
                                except Exception as e:
                                    print(f"Failed to evaluate {item.target.id}: {e}")
    
    return extracted_data

def audit_expert_tables():
    """
    Audit the consistency of expert tables against the actual product catalog.
    """
    print("Starting Expert Table Consistency Audit (AST Mode)...")
    print("=" * 60)

    # 1. Load Data
    catalog_path = project_root / 'data' / 'products_structured.json'
    products = load_product_catalog(catalog_path)
    
    # Create a set of valid product codes from the catalog
    valid_product_codes = {p['code'].strip() for p in products if 'code' in p}
    print(f"Loaded {len(valid_product_codes)} unique product codes from catalog.")

    # 2. Extract Expert Tables via AST
    source_path = project_root / 'src' / 'data_ingestion' / 'product_catalog_retriever.py'
    tables = extract_expert_tables_from_source(source_path)
    
    app_recs = tables.get('APPLICATION_RECOMMENDATIONS', {})
    ind_recs = tables.get('INDUSTRY_RECOMMENDATIONS', {})

    if not app_recs or not ind_recs:
        print("Error: Failed to extract recommendation tables from source code.")
        sys.exit(1)

    print(f"Extracted {len(app_recs)} Application categories.")
    print(f"Extracted {len(ind_recs)} Industry categories.")

    # ---------------------------------------------------------
    # Check 1: Phantom Product Codes (In tables but not in catalog)
    # ---------------------------------------------------------
    print("\n[Check 1] Phantom Product Solution (Codes in tables but NOT in catalog)")
    print("-" * 60)
    
    phantoms_found = False
    
    # Check Application Recommendations
    for app, codes in app_recs.items():
        for code in codes:
            if code.strip() not in valid_product_codes:
                print(f"[FAIL] Phantom code '{code}' found in Application table under '{app}'")
                phantoms_found = True

    # Check Industry Recommendations
    for ind, codes in ind_recs.items():
        for code in codes:
            if code.strip() not in valid_product_codes:
                print(f"[FAIL] Phantom code '{code}' found in Industry table under '{ind}'")
                phantoms_found = True

    if not phantoms_found:
        print("[PASS] No phantom product codes found.")

    # ---------------------------------------------------------
    # Check 2: Duplicates within Tables
    # ---------------------------------------------------------
    print("\n[Check 2] Duplicates within Tables")
    print("-" * 60)
    
    duplicates_found = False
    
    for app, codes in app_recs.items():
        if len(codes) != len(set(codes)):
            seen = set()
            dupes = [x for x in codes if x in seen or seen.add(x)]
            print(f"[FAIL] Duplicate codes in Application '{app}': {dupes}")
            duplicates_found = True

    for ind, codes in ind_recs.items():
        if len(codes) != len(set(codes)):
            seen = set()
            dupes = [x for x in codes if x in seen or seen.add(x)]
            print(f"[FAIL] Duplicate codes in Industry '{ind}': {dupes}")
            duplicates_found = True

    if not duplicates_found:
        print("[PASS] No duplicates found within table entries.")

    # ---------------------------------------------------------
    # Check 3: Orphaned Products (In catalog but not in ANY table)
    # ---------------------------------------------------------
    print("\n[Check 3] Orphaned Products Coverage")
    print("-" * 60)
    
    referenced_in_apps = set()
    for codes in app_recs.values():
        referenced_in_apps.update(c.strip() for c in codes)
        
    referenced_in_inds = set()
    for codes in ind_recs.values():
        referenced_in_inds.update(c.strip() for c in codes)

    orphans_found = False
    
    # Products missing from Application tables
    missing_from_apps = valid_product_codes - referenced_in_apps
    if missing_from_apps:
        print(f"[WARN] {len(missing_from_apps)} products not referenced in any Application table:")
        for code in sorted(missing_from_apps):
            print(f"  - {code}")
        orphans_found = True
    else:
        print("[PASS] All products referenced in at least one Application table.")

    print("") 

    # Products missing from Industry tables
    missing_from_inds = valid_product_codes - referenced_in_inds
    if missing_from_inds:
        print(f"[WARN] {len(missing_from_inds)} products not referenced in any Industry table:")
        for code in sorted(missing_from_inds):
            print(f"  - {code}")
        orphans_found = True
    else:
        print("[PASS] All products referenced in at least one Industry table.")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Audit Summary")
    print("-" * 60)
    
    status = "SUCCESS"
    if phantoms_found or duplicates_found:
        status = "FAILURE (Data Integrity Issues)"
    elif orphans_found:
        status = "WARNING (Make sure all products are covered)"
        
    print(f"Final Status: {status}")
    print(f"Total Products in Catalog: {len(valid_product_codes)}")
    print(f"Unique Products in App Tables: {len(referenced_in_apps)}")
    print(f"Unique Products in Ind Tables: {len(referenced_in_inds)}")

if __name__ == "__main__":
    audit_expert_tables()
