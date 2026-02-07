"""
Document Generation API Router
Provides endpoints for generating and downloading PDF documents.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# Request/Response Models
class QuotationRequest(BaseModel):
    """Request model for quotation generation."""
    customer_name: str
    customer_email: EmailStr
    products: List[Dict[str, Any]]  # [{code, name, quantity, unit_price}]
    notes: Optional[str] = ""
    validity_days: int = 30
    terms: Optional[str] = "Standard terms and conditions apply"


class DatasheetRequest(BaseModel):
    """Request model for datasheet generation."""
    product_code: str
    product_name: Optional[str] = None
    include_certifications: bool = True


class ProposalRequest(BaseModel):
    """Request model for proposal generation."""
    customer_name: str
    customer_email: EmailStr
    project_name: str
    products: List[Dict[str, Any]]
    scope_of_work: str
    delivery_timeline: str
    notes: Optional[str] = ""


class DocumentResponse(BaseModel):
    """Response model for document generation."""
    doc_id: str
    doc_type: str
    title: str
    filename: str
    download_url: str
    format: str
    created_at: str


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[Dict[str, Any]]
    count: int


# Endpoints
@router.post("/quotation", response_model=DocumentResponse)
async def generate_quotation(request: QuotationRequest):
    """
    Generate a quotation PDF for a customer.
    
    Includes product pricing, taxes, and payment terms.
    """
    from src.documents.pdf_generator import get_pdf_generator
    
    try:
        generator = get_pdf_generator()
        
        doc = generator.generate_quotation(
            customer_name=request.customer_name,
            customer_email=request.customer_email,
            products=request.products,
            notes=request.notes or "",
            validity_days=request.validity_days,
            terms=request.terms or "Standard terms and conditions apply"
        )
        
        return DocumentResponse(
            doc_id=doc.doc_id,
            doc_type=doc.doc_type.value,
            title=doc.title,
            filename=doc.filename,
            download_url=f"/documents/download/{doc.doc_id}",
            format=doc.format,
            created_at=doc.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating quotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasheet", response_model=DocumentResponse)
async def generate_datasheet(request: DatasheetRequest):
    """
    Generate a product datasheet PDF.
    
    Includes technical specifications, applications, and certifications.
    """
    from src.documents.pdf_generator import get_pdf_generator
    from src.data_ingestion.product_catalog_retriever import get_product_retriever
    from src.knowledge_base.certification_loader import get_certification_loader
    
    try:
        generator = get_pdf_generator()
        retriever = get_product_retriever()
        cert_loader = get_certification_loader()
        
        # Find product in catalog
        product_data = retriever.get_product_details(request.product_code)
        
        if not product_data:
            raise HTTPException(
                status_code=404,
                detail=f"Product {request.product_code} not found in catalog"
            )
        
        # Get specifications - handle both formats
        specs = product_data.get("specifications", {})
        
        # Format temperature range from min/max
        temp_range = "N/A"
        if specs.get("temperature_min") is not None and specs.get("temperature_max") is not None:
            temp_range = f"{specs['temperature_min']}°C to {specs['temperature_max']}°C"
        elif specs.get("temperature_range"):
            temp_range = specs.get("temperature_range")
        
        # Format pressure ranges
        pressure_parts = []
        if specs.get("pressure_static"):
            pressure_parts.append(f"{specs['pressure_static']} bar (static)")
        if specs.get("pressure_rotary"):
            pressure_parts.append(f"{specs['pressure_rotary']} bar (rotary)")
        if specs.get("pressure_reciprocating"):
            pressure_parts.append(f"{specs['pressure_reciprocating']} bar (reciprocating)")
        pressure_range = ", ".join(pressure_parts) if pressure_parts else specs.get("pressure_range", "N/A")
        
        # Format pH range
        ph_range = "N/A"
        if specs.get("ph_min") is not None and specs.get("ph_max") is not None:
            ph_range = f"{specs['ph_min']} to {specs['ph_max']}"
        elif specs.get("ph_range"):
            ph_range = specs.get("ph_range")
        
        # Format shaft speed
        shaft_speed = "N/A"
        if specs.get("shaft_speed_rotary"):
            shaft_speed = f"{specs['shaft_speed_rotary']} m/sec (rotary)"
        
        specifications = {
            "Temperature Range": temp_range,
            "Pressure Range": pressure_range,
            "pH Range": ph_range,
            "Shaft Speed": shaft_speed,
            "Material": product_data.get("material", "N/A"),
            "Density": specs.get("density", "N/A"),
        }
        
        # Get certifications if requested
        certifications = product_data.get("certifications", [])
        if request.include_certifications:
            try:
                certs = cert_loader.get_certifications_for_product(request.product_code)
                certifications = [c.standard for c in certs]
            except Exception:
                pass  # Use product certifications if loader fails
        
        doc = generator.generate_datasheet(
            product_code=product_data["code"],
            product_name=request.product_name or product_data.get("name", request.product_code),
            specifications=specifications,
            certifications=certifications,
            applications=product_data.get("applications", []),
            materials={"Base Material": product_data.get("material")} if product_data.get("material") else {}
        )
        
        return DocumentResponse(
            doc_id=doc.doc_id,
            doc_type=doc.doc_type.value,
            title=doc.title,
            filename=doc.filename,
            download_url=f"/documents/download/{doc.doc_id}",
            format=doc.format,
            created_at=doc.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating datasheet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{doc_id}")
async def download_document(doc_id: str):
    """
    Download a generated document by its ID.
    """
    from src.documents.pdf_generator import get_pdf_generator
    
    generator = get_pdf_generator()
    file_path = generator.get_document(doc_id)
    
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".md": "text/markdown"
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type
    )


@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """
    List all generated documents.
    """
    from src.documents.pdf_generator import get_pdf_generator
    
    generator = get_pdf_generator()
    documents = generator.list_generated_documents()
    
    return DocumentListResponse(
        documents=documents,
        count=len(documents)
    )


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    Delete a generated document.
    """
    from src.documents.pdf_generator import get_pdf_generator
    
    generator = get_pdf_generator()
    file_path = generator.get_document(doc_id)
    
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        os.remove(str(file_path))
        return {"status": "deleted", "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/certifications")
async def list_certifications():
    """
    List all available certifications.
    """
    from src.knowledge_base.certification_loader import get_certification_loader
    
    loader = get_certification_loader()
    certifications = loader.get_all_certifications()
    
    return {
        "certifications": [c.to_dict() for c in certifications],
        "count": len(certifications)
    }


@router.get("/certifications/{standard}")
async def get_certification(standard: str):
    """
    Get details for a specific certification standard.
    """
    from src.knowledge_base.certification_loader import get_certification_loader
    
    loader = get_certification_loader()
    cert = loader.get_certification(standard)
    
    if not cert:
        raise HTTPException(status_code=404, detail=f"Certification {standard} not found")
    
    return cert.to_dict()


@router.get("/certifications/product/{product_code}")
async def get_product_certifications(product_code: str):
    """
    Get all certifications applicable to a product.
    """
    from src.knowledge_base.certification_loader import get_certification_loader
    
    loader = get_certification_loader()
    certifications = loader.get_certifications_for_product(product_code)
    
    return {
        "product_code": product_code,
        "certifications": [c.to_dict() for c in certifications],
        "count": len(certifications)
    }
