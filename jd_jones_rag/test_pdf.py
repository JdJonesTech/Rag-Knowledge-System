"""Test PDF generation inside Docker container."""
from src.documents.pdf_generator import get_pdf_generator

generator = get_pdf_generator()

products = [
    {
        "code": "NA 710",
        "name": "Flexible pure graphite compressed pressure seal ring reinforced with AISI 304 wire net",
        "size": {"od": "299", "id": "279", "th": "20"},
        "dimension_unit": "mm",
        "material": "2520456100323010",
        "material_code": "GRAPHITE",
        "material_grade": "Pure Graphite",
        "quantity": 2,
        "unit": "Nos.",
        "unit_price": 2219,
        "rings_per_set": "-",
    },
    {
        "code": "NA 707",
        "name": "Flexible pure graphite compressed gland packing ring set in endless form",
        "size": {"od": "80", "id": "54", "th": "13"},
        "dimension_unit": "mm",
        "material": "2520456100324810",
        "material_code": "GRAPHITE",
        "material_grade": "Pure Graphite",
        "quantity": 40,
        "unit": "Nos.",
        "unit_price": 995,
        "rings_per_set": "10",
    },
    {
        "code": "NA 710",
        "name": "Flexible pure graphite pressure seal ring",
        "size": {"od": "266", "id": "244", "th": "20"},
        "dimension_unit": "mm",
        "material": "2520456100223000",
        "material_code": "GRAPHITE",
        "material_grade": "Pure Graphite",
        "quantity": 2,
        "unit": "Nos.",
        "unit_price": 725,
        "rings_per_set": "-",
    },
    {
        "code": "NA 707",
        "name": "Flexible pure graphite gland packing ring set",
        "size": {"od": "72.5", "id": "50.7", "th": "9.5"},
        "dimension_unit": "mm",
        "material": "2520456100224800",
        "material_code": "GRAPHITE",
        "material_grade": "Pure Graphite",
        "quantity": 4,
        "unit": "Set",
        "unit_price": 844,
        "rings_per_set": "10",
    },
    {
        "code": "NA 710",
        "name": "Flexible pure graphite pressure seal ring",
        "size": {"od": "260", "id": "242", "th": "20"},
        "dimension_unit": "mm",
        "material": "2520456110323010",
        "material_code": "GRAPHITE",
        "material_grade": "Pure Graphite",
        "quantity": 2,
        "unit": "Nos.",
        "unit_price": 586,
        "rings_per_set": "-",
    },
]

doc = generator.generate_quotation(
    customer_name="Mr. R. K. Behura",
    customer_email="ratikantabehura@iffco.in",
    products=products,
    notes="",
    validity_days=90,
    customer_company="IIFCO, PARADIP",
    customer_designation="Sr. Mgr. (Power)",
    customer_address="Odisha 754120",
    rfq_number="252002261012",
    rfq_date="15.10.2025",
    due_date="23.10.2025 at 12:00 Hours",
)

print(f"SUCCESS: Generated PDF: {doc.filename}")
print(f"Path: {doc.file_path}")
print(f"Format: {doc.format}")

# Check file size
import os
fsize = os.path.getsize(doc.file_path)
print(f"File size: {fsize} bytes")

# Test 2: Mixed units (mm and inch)
print("\n--- Test 2: Mixed dimension units ---")
mixed_products = [
    {
        "code": "NA 701",
        "name": "Pure Graphite Packing",
        "size": {"od": "100", "id": "80", "th": "10"},
        "dimension_unit": "mm",
        "material_code": "GRAPHITE",
        "material_grade": "Standard",
        "quantity": 5,
        "unit": "Nos.",
        "unit_price": 500,
        "rings_per_set": "-",
    },
    {
        "code": "NA 702",
        "name": "Graphite with PTFE Corners",
        "size": {"od": "4.5", "id": "3.0", "th": "0.5"},
        "dimension_unit": "inch",
        "material_code": "GRAPHITE/PTFE",
        "material_grade": "High Purity",
        "quantity": 10,
        "unit": "Set",
        "unit_price": 750,
        "rings_per_set": "8",
    },
]

doc2 = generator.generate_quotation(
    customer_name="Test Customer",
    customer_email="test@example.com",
    products=mixed_products,
    notes="Testing mixed dimension units",
    validity_days=30,
    customer_company="Test Corp",
)

print(f"SUCCESS: Generated mixed-unit PDF: {doc2.filename}")
print(f"Path: {doc2.file_path}")
print(f"Format: {doc2.format}")
fsize2 = os.path.getsize(doc2.file_path)
print(f"File size: {fsize2} bytes")
