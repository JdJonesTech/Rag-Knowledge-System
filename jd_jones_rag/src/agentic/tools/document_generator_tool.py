"""
Document Generator Tool
Generates professional documents from templates and data.
Includes validation against industry standards.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

from src.agentic.tools.base_tool import BaseTool, ToolResult, ToolStatus


class DocumentType(str, Enum):
    """Types of documents that can be generated."""
    QUOTATION = "quotation"
    TECHNICAL_DATASHEET = "technical_datasheet"
    CERTIFICATE = "certificate"
    SPECIFICATION_SHEET = "specification_sheet"
    PROPOSAL = "proposal"
    ORDER_CONFIRMATION = "order_confirmation"


class DocumentGeneratorTool(BaseTool):
    """
    Tool for generating professional documents.
    
    Capabilities:
    - Generate quotations with accurate pricing
    - Create technical datasheets
    - Generate specification sheets
    - Create proposals
    - Validate against standards before generation
    """
    
    # Document templates (simplified; in production, use actual template engine)
    TEMPLATES = {
        DocumentType.QUOTATION: {
            "title": "QUOTATION",
            "fields": ["quote_number", "customer", "products", "pricing", "validity", "terms"],
            "requires_validation": True
        },
        DocumentType.TECHNICAL_DATASHEET: {
            "title": "TECHNICAL DATA SHEET",
            "fields": ["product_name", "specifications", "applications", "certifications", "limitations"],
            "requires_validation": True
        },
        DocumentType.SPECIFICATION_SHEET: {
            "title": "PRODUCT SPECIFICATION",
            "fields": ["product_id", "dimensions", "materials", "performance", "standards"],
            "requires_validation": True
        },
        DocumentType.PROPOSAL: {
            "title": "TECHNICAL PROPOSAL",
            "fields": ["customer", "requirements", "solution", "products", "pricing", "timeline"],
            "requires_validation": True
        }
    }
    
    def __init__(self):
        """Initialize document generator."""
        super().__init__(
            name="document_generator",
            description="""
            Generates professional documents:
            - Quotations with pricing
            - Technical datasheets
            - Specification sheets
            - Technical proposals
            
            Validates data against industry standards before generation.
            """
        )
        
        self.generated_documents: List[Dict[str, Any]] = []
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """Generate a document."""
        try:
            doc_type_str = parameters.get("document_type", "quotation")
            
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown document type: {doc_type_str}"
                )
            
            # Route to appropriate generator
            if doc_type == DocumentType.QUOTATION:
                return await self._generate_quotation(parameters)
            
            elif doc_type == DocumentType.TECHNICAL_DATASHEET:
                return await self._generate_datasheet(parameters)
            
            elif doc_type == DocumentType.SPECIFICATION_SHEET:
                return await self._generate_spec_sheet(parameters)
            
            elif doc_type == DocumentType.PROPOSAL:
                return await self._generate_proposal(parameters)
            
            else:
                return await self._generate_generic(doc_type, parameters)
                
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _generate_quotation(self, parameters: Dict[str, Any]) -> ToolResult:
        """Generate a quotation document."""
        quote_number = f"QUO-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        customer = parameters.get("customer", {})
        products = parameters.get("products", [])
        validity_days = parameters.get("validity_days", 30)
        
        # Calculate totals
        line_items = []
        subtotal = 0
        
        for product in products:
            unit_price = product.get("unit_price", 0)
            quantity = product.get("quantity", 1)
            line_total = unit_price * quantity
            subtotal += line_total
            
            line_items.append({
                "product_id": product.get("product_id"),
                "description": product.get("description", ""),
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total
            })
        
        # Apply discounts
        discount_percent = 0
        if subtotal > 100000:
            discount_percent = 15
        elif subtotal > 50000:
            discount_percent = 10
        elif subtotal > 10000:
            discount_percent = 5
        
        discount_amount = subtotal * (discount_percent / 100)
        total = subtotal - discount_amount
        
        # Validate against certifications
        validation_warnings = await self._validate_for_document(products, parameters)
        
        document = {
            "document_id": quote_number,
            "document_type": "quotation",
            "customer": customer,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "valid_until": (datetime.now().replace(day=datetime.now().day + validity_days)).strftime("%Y-%m-%d"),
            "line_items": line_items,
            "subtotal": subtotal,
            "discount_percent": discount_percent,
            "discount_amount": discount_amount,
            "total": total,
            "currency": parameters.get("currency", "USD"),
            "terms": {
                "payment": "Net 30",
                "delivery": parameters.get("delivery_terms", "Ex-Works Kolkata"),
                "validity": f"{validity_days} days"
            },
            "notes": parameters.get("notes", ""),
            "validation_warnings": validation_warnings,
            "generated_at": datetime.now().isoformat()
        }
        
        self.generated_documents.append(document)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "document_id": quote_number,
                "document_type": "quotation",
                "document": document,
                "download_url": f"/api/documents/{quote_number}",
                "validation_warnings": validation_warnings
            }
        )
    
    async def _generate_datasheet(self, parameters: Dict[str, Any]) -> ToolResult:
        """Generate a technical datasheet."""
        product_id = parameters.get("product_id", "UNKNOWN")
        doc_id = f"TDS-{product_id}-{datetime.now().strftime('%Y%m%d')}"
        
        document = {
            "document_id": doc_id,
            "document_type": "technical_datasheet",
            "product_id": product_id,
            "product_name": parameters.get("product_name", product_id),
            "revision": parameters.get("revision", "1.0"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sections": {
                "description": parameters.get("description", ""),
                "applications": parameters.get("applications", []),
                "specifications": {
                    "temperature_range": parameters.get("temperature_range", ""),
                    "pressure_rating": parameters.get("pressure_rating", ""),
                    "chemical_resistance": parameters.get("chemical_resistance", []),
                    "dimensions": parameters.get("dimensions", {})
                },
                "certifications": parameters.get("certifications", []),
                "installation": parameters.get("installation_notes", ""),
                "limitations": parameters.get("limitations", []),
                "ordering_info": parameters.get("ordering_info", "")
            },
            "generated_at": datetime.now().isoformat()
        }
        
        self.generated_documents.append(document)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "document_id": doc_id,
                "document_type": "technical_datasheet",
                "document": document,
                "download_url": f"/api/documents/{doc_id}"
            }
        )
    
    async def _generate_spec_sheet(self, parameters: Dict[str, Any]) -> ToolResult:
        """Generate a specification sheet."""
        product_id = parameters.get("product_id", "UNKNOWN")
        doc_id = f"SPEC-{product_id}-{datetime.now().strftime('%Y%m%d')}"
        
        document = {
            "document_id": doc_id,
            "document_type": "specification_sheet",
            "product_id": product_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "specifications": parameters.get("specifications", {}),
            "materials": parameters.get("materials", {}),
            "performance_data": parameters.get("performance_data", {}),
            "applicable_standards": parameters.get("standards", []),
            "test_results": parameters.get("test_results", {}),
            "generated_at": datetime.now().isoformat()
        }
        
        self.generated_documents.append(document)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "document_id": doc_id,
                "document_type": "specification_sheet",
                "document": document,
                "download_url": f"/api/documents/{doc_id}"
            }
        )
    
    async def _generate_proposal(self, parameters: Dict[str, Any]) -> ToolResult:
        """Generate a technical proposal."""
        proposal_id = f"PROP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        document = {
            "document_id": proposal_id,
            "document_type": "proposal",
            "customer": parameters.get("customer", {}),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sections": {
                "executive_summary": parameters.get("summary", ""),
                "requirements": parameters.get("requirements", []),
                "proposed_solution": parameters.get("solution", ""),
                "products": parameters.get("products", []),
                "technical_details": parameters.get("technical_details", {}),
                "pricing_summary": parameters.get("pricing", {}),
                "timeline": parameters.get("timeline", ""),
                "terms_conditions": parameters.get("terms", "Standard terms apply")
            },
            "validity_days": parameters.get("validity_days", 30),
            "generated_at": datetime.now().isoformat()
        }
        
        self.generated_documents.append(document)
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "document_id": proposal_id,
                "document_type": "proposal",
                "document": document,
                "download_url": f"/api/documents/{proposal_id}"
            }
        )
    
    async def _generate_generic(self, doc_type: DocumentType, parameters: Dict[str, Any]) -> ToolResult:
        """Generate a generic document."""
        doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"
        
        document = {
            "document_id": doc_id,
            "document_type": doc_type.value,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "content": parameters,
            "generated_at": datetime.now().isoformat()
        }
        
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            data={
                "document_id": doc_id,
                "document_type": doc_type.value,
                "document": document
            }
        )
    
    async def _validate_for_document(
        self,
        products: List[Dict[str, Any]],
        parameters: Dict[str, Any]
    ) -> List[str]:
        """Validate products against requirements before generating document."""
        warnings = []
        
        required_certs = parameters.get("required_certifications", [])
        required_temp = parameters.get("required_temperature")
        required_pressure = parameters.get("required_pressure")
        
        for product in products:
            product_certs = product.get("certifications", [])
            
            # Check certifications
            for cert in required_certs:
                if cert.upper() not in [c.upper() for c in product_certs]:
                    warnings.append(
                        f"Product {product.get('product_id')} may not have {cert} certification. "
                        f"Please verify before finalizing quote."
                    )
        
        # Add timestamp warning if certs might be outdated
        if required_certs:
            warnings.append(
                "Note: Please verify all certifications are current before finalizing this document."
            )
        
        return warnings
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": ["quotation", "technical_datasheet", "specification_sheet", "proposal", "certificate"]
                },
                "customer": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "contact_name": {"type": "string"},
                        "email": {"type": "string"},
                        "address": {"type": "string"}
                    }
                },
                "products": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "description": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "number"}
                        }
                    }
                },
                "product_id": {"type": "string"},
                "specifications": {"type": "object"},
                "certifications": {"type": "array", "items": {"type": "string"}},
                "required_certifications": {"type": "array", "items": {"type": "string"}},
                "validity_days": {"type": "integer"},
                "currency": {"type": "string"},
                "notes": {"type": "string"}
            }
        }
