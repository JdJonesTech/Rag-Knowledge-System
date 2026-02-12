#!/usr/bin/env python3
"""
Product Catalog Data Completeness Audit Script
E1 Task - Phase 2

This script performs a comprehensive audit of the products_structured.json file
to verify data completeness and quality for all 104 products.

Validation Checks:
    1. Product code (not empty)
    2. Name (not empty)
    3. Category (not empty)
    4. Max temperature (not None/0)
    5. Max pressure (not None/0)
    6. At least 1 application listed
    7. Description (length > 20 chars)
    8. Certifications presence

Output:
    - Detailed audit report saved to data/output/product_audit_report.txt
    - Summary statistics for each field
    - List of products with missing/incomplete data
    - Products without certifications

Usage:
    python scripts/audit_product_data.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import defaultdict


class ProductAuditor:
    """Audits product catalog data for completeness and quality."""
    
    def __init__(self, products_file: Path):
        """Initialize the auditor with the products file path."""
        self.products_file = products_file
        self.products: List[Dict[str, Any]] = []
        self.audit_results: Dict[str, Any] = {}
        self.issues: List[Dict[str, Any]] = []
        
    def load_products(self) -> bool:
        """Load products from JSON file."""
        try:
            with open(self.products_file, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
            print(f"✓ Loaded {len(self.products)} products from {self.products_file.name}")
            return True
        except FileNotFoundError:
            print(f"✗ ERROR: File not found: {self.products_file}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ ERROR: Invalid JSON: {e}")
            return False
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return False
    
    def validate_product_code(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has a valid code."""
        code = product.get('code', '').strip()
        if not code:
            return False, "Product code is empty or missing"
        return True, "OK"
    
    def validate_name(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has a valid name."""
        name = product.get('name', '').strip()
        if not name:
            return False, "Product name is empty or missing"
        return True, "OK"
    
    def validate_category(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has a valid category."""
        category = product.get('category', '').strip()
        if not category:
            return False, "Category is empty or missing"
        return True, "OK"
    
    def validate_max_temperature(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has max temperature specified."""
        specs = product.get('specs', {})
        if not specs:
            return False, "No specs section found"
        
        temp_max = specs.get('temperature_max')
        if temp_max is None:
            return False, "Max temperature is None"
        if temp_max == 0:
            return False, "Max temperature is 0"
        
        return True, f"OK ({temp_max}°C)"
    
    def validate_max_pressure(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has max pressure specified."""
        specs = product.get('specs', {})
        if not specs:
            return False, "No specs section found"
        
        # Check static pressure as the primary pressure indicator
        pressure = specs.get('pressure_static')
        if pressure is None:
            return False, "Max pressure (static) is None"
        if pressure == 0:
            return False, "Max pressure is 0"
        
        return True, f"OK ({pressure} bar)"
    
    def validate_applications(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has at least one application."""
        applications = product.get('applications', [])
        if not applications or len(applications) == 0:
            return False, "No applications listed"
        
        # Filter out empty strings
        valid_apps = [app for app in applications if app and app.strip()]
        if not valid_apps:
            return False, "Applications list is empty or contains only empty strings"
        
        return True, f"OK ({len(valid_apps)} applications)"
    
    def validate_description(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has a description with length > 20 chars."""
        description = product.get('description', '').strip()
        if not description:
            return False, "Description is empty or missing"
        if len(description) <= 20:
            return False, f"Description too short ({len(description)} chars, need >20)"
        
        return True, f"OK ({len(description)} chars)"
    
    def check_certifications(self, product: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if product has certifications (not a validation failure, just tracking)."""
        certifications = product.get('certifications', [])
        if not certifications or len(certifications) == 0:
            return False, "No certifications"
        
        valid_certs = [cert for cert in certifications if cert and cert.strip()]
        if not valid_certs:
            return False, "Certifications list is empty"
        
        return True, f"{len(valid_certs)} certifications"
    
    def audit_single_product(self, product: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Audit a single product and return results."""
        code = product.get('code', f'UNKNOWN_{index}')
        
        result = {
            'index': index + 1,
            'code': code,
            'name': product.get('name', 'N/A'),
            'validations': {},
            'has_issues': False,
            'issue_count': 0
        }
        
        # Run all validations
        validations = [
            ('product_code', self.validate_product_code),
            ('name', self.validate_name),
            ('category', self.validate_category),
            ('max_temperature', self.validate_max_temperature),
            ('max_pressure', self.validate_max_pressure),
            ('applications', self.validate_applications),
            ('description', self.validate_description),
            ('certifications', self.check_certifications)
        ]
        
        for field_name, validator in validations:
            is_valid, message = validator(product)
            result['validations'][field_name] = {
                'valid': is_valid,
                'message': message
            }
            
            # Track issues (certifications is informational, not a failure)
            if not is_valid and field_name != 'certifications':
                result['has_issues'] = True
                result['issue_count'] += 1
        
        return result
    
    def run_audit(self) -> Dict[str, Any]:
        """Run complete audit on all products."""
        print("\n" + "=" * 60)
        print("RUNNING PRODUCT DATA AUDIT")
        print("=" * 60)
        print()
        
        results = []
        products_with_issues = []
        products_without_certs = []
        
        # Field-level statistics
        field_stats = {
            'product_code': {'valid': 0, 'invalid': 0},
            'name': {'valid': 0, 'invalid': 0},
            'category': {'valid': 0, 'invalid': 0},
            'max_temperature': {'valid': 0, 'invalid': 0},
            'max_pressure': {'valid': 0, 'invalid': 0},
            'applications': {'valid': 0, 'invalid': 0},
            'description': {'valid': 0, 'invalid': 0},
            'certifications': {'valid': 0, 'invalid': 0}
        }
        
        # Audit each product
        for i, product in enumerate(self.products):
            result = self.audit_single_product(product, i)
            results.append(result)
            
            # Update statistics
            for field, validation in result['validations'].items():
                if validation['valid']:
                    field_stats[field]['valid'] += 1
                else:
                    field_stats[field]['invalid'] += 1
            
            # Track problematic products
            if result['has_issues']:
                products_with_issues.append(result)
            
            if not result['validations']['certifications']['valid']:
                products_without_certs.append(result)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(self.products)} products...")
        
        print(f"  Processed {len(self.products)}/{len(self.products)} products...")
        print("\n✓ Audit complete!")
        
        # Compile results
        self.audit_results = {
            'total_products': len(self.products),
            'field_stats': field_stats,
            'detailed_results': results,
            'products_with_issues': products_with_issues,
            'products_without_certs': products_without_certs,
            'audit_timestamp': datetime.now().isoformat()
        }
        
        return self.audit_results
    
    def generate_report(self, output_file: Path) -> None:
        """Generate detailed audit report and save to file."""
        if not self.audit_results:
            print("✗ No audit results available. Run audit first.")
            return
        
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("PRODUCT CATALOG DATA COMPLETENESS AUDIT REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Products File: {self.products_file}")
        lines.append(f"Total Products: {self.audit_results['total_products']}")
        lines.append("=" * 80)
        lines.append("")
        
        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        total = self.audit_results['total_products']
        issues = len(self.audit_results['products_with_issues'])
        no_certs = len(self.audit_results['products_without_certs'])
        
        lines.append(f"Total Products Audited:        {total}")
        lines.append(f"Products with Issues:          {issues} ({issues/total*100:.1f}%)")
        lines.append(f"Products without Issues:       {total - issues} ({(total-issues)/total*100:.1f}%)")
        lines.append(f"Products without Certifications: {no_certs} ({no_certs/total*100:.1f}%)")
        lines.append("")
        
        # Field-by-Field Statistics
        lines.append("FIELD COMPLETENESS STATISTICS")
        lines.append("-" * 80)
        lines.append(f"{'Field':<25} {'Valid':<10} {'Invalid':<10} {'% Complete':<15}")
        lines.append("-" * 80)
        
        field_stats = self.audit_results['field_stats']
        for field, stats in field_stats.items():
            valid = stats['valid']
            invalid = stats['invalid']
            pct = (valid / total * 100) if total > 0 else 0
            
            field_display = field.replace('_', ' ').title()
            lines.append(f"{field_display:<25} {valid:<10} {invalid:<10} {pct:>6.1f}%")
        
        lines.append("")
        
        # Products with Missing Fields
        lines.append("PRODUCTS WITH MISSING OR INCOMPLETE DATA")
        lines.append("-" * 80)
        
        if self.audit_results['products_with_issues']:
            lines.append(f"Found {len(self.audit_results['products_with_issues'])} products with issues:")
            lines.append("")
            
            for result in self.audit_results['products_with_issues']:
                lines.append(f"Product #{result['index']}: {result['code']}")
                lines.append(f"  Name: {result['name']}")
                lines.append(f"  Issues ({result['issue_count']}):")
                
                for field, validation in result['validations'].items():
                    if not validation['valid'] and field != 'certifications':
                        field_display = field.replace('_', ' ').title()
                        lines.append(f"    ✗ {field_display}: {validation['message']}")
                
                lines.append("")
        else:
            lines.append("✓ No products with missing required fields!")
            lines.append("")
        
        # Products without Certifications
        lines.append("PRODUCTS WITHOUT CERTIFICATIONS")
        lines.append("-" * 80)
        
        if self.audit_results['products_without_certs']:
            lines.append(f"Found {len(self.audit_results['products_without_certs'])} products without certifications:")
            lines.append("")
            
            # Group by category for better readability
            by_category = defaultdict(list)
            for result in self.audit_results['products_without_certs']:
                # Find the product to get category
                product = next((p for p in self.products if p.get('code') == result['code']), None)
                category = product.get('category', 'Unknown') if product else 'Unknown'
                by_category[category].append(result)
            
            for category, products in sorted(by_category.items()):
                lines.append(f"\n{category}:")
                for result in products:
                    lines.append(f"  - {result['code']}: {result['name']}")
            
            lines.append("")
        else:
            lines.append("✓ All products have certifications!")
            lines.append("")
        
        # Detailed Product Breakdown (Summary)
        lines.append("DETAILED PRODUCT VALIDATION SUMMARY")
        lines.append("-" * 80)
        lines.append(f"{'#':<5} {'Code':<15} {'Name':<35} {'Issues':<10}")
        lines.append("-" * 80)
        
        for result in self.audit_results['detailed_results']:
            index = result['index']
            code = result['code'][:14]
            name = result['name'][:34]
            issues = result['issue_count']
            status = "✓ OK" if issues == 0 else f"✗ {issues}"
            
            lines.append(f"{index:<5} {code:<15} {name:<35} {status:<10}")
        
        lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        
        if len(self.audit_results['products_with_issues']) > 0:
            lines.append("1. Review and complete missing required fields for products listed above")
            lines.append("2. Ensure all products have temperature and pressure specifications")
            lines.append("3. Add application information for products missing this data")
            lines.append("4. Verify and expand product descriptions where needed")
        
        if len(self.audit_results['products_without_certs']) > 0:
            lines.append("5. Consider adding certifications for products that lack them")
            lines.append("6. Verify if products truly lack certifications or if data is missing")
        
        if len(self.audit_results['products_with_issues']) == 0:
            lines.append("✓ All required fields are complete!")
            lines.append("  Consider adding certifications to products that don't have them.")
        
        lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        report_content = "\n".join(lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n✓ Report saved to: {output_file}")
        print(f"  Report size: {len(report_content):,} characters")
        
        return report_content


def main():
    """Main execution function."""
    print("=" * 60)
    print("Product Catalog Data Completeness Audit")
    print("E1 Task - Phase 2")
    print("=" * 60)
    print()
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    products_file = project_root / "data" / "products_structured.json"
    output_file = project_root / "data" / "output" / "product_audit_report.txt"
    
    print(f"Input:  {products_file}")
    print(f"Output: {output_file}")
    print()
    
    # Initialize auditor
    auditor = ProductAuditor(products_file)
    
    # Load products
    if not auditor.load_products():
        print("\n✗ Failed to load products. Exiting.")
        return 1
    
    # Run audit
    results = auditor.run_audit()
    
    # Generate report
    print("\nGenerating detailed report...")
    auditor.generate_report(output_file)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Products:           {results['total_products']}")
    print(f"Products with Issues:     {len(results['products_with_issues'])}")
    print(f"Products without Certs:   {len(results['products_without_certs'])}")
    print()
    
    # Field completeness
    print("Field Completeness:")
    for field, stats in results['field_stats'].items():
        pct = (stats['valid'] / results['total_products'] * 100)
        status = "✓" if pct >= 95 else "⚠" if pct >= 80 else "✗"
        field_display = field.replace('_', ' ').title()
        print(f"  {status} {field_display:<25} {pct:>6.1f}%")
    
    print()
    print("=" * 60)
    
    if len(results['products_with_issues']) == 0:
        print("✓ SUCCESS: All products have complete required data!")
    else:
        print(f"⚠ WARNING: {len(results['products_with_issues'])} products need attention")
    
    print("=" * 60)
    print()
    print(f"📄 Full report: {output_file}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
